use super::*;

#[test]
fn gaussian_hybrid_three_stage_row_fft() {
    use crate::application_support::ckks::encode_rns;
    use crate::ckks::{CkksChainState, PreparedHybridRotationContext, RnsCkksEvaluationKeys};
    use crate::grafting::{
        decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
    };
    use crate::ring::{
        centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, Modulus,
        ModulusBasis,
    };
    use crate::rlwe::ErrorDistribution;
    use num_traits::ToPrimitive;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    let profile = crate::ckks::CkksParameterProfile::new(
        "hybrid-three-stage-correctness",
        4096,
        &[
            0x3ffffffdf0001,
            0x7fffba001,
            0x7fffb0001,
            0x7ffff6001,
            0x7fffa4001,
        ],
        2.0_f64.powi(35),
        crate::ckks::CkksParameterClass::Research,
        None,
    );
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let chain = profile.modulus_chain();
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();

    assert!(chain.max_level() >= 3);

    let shape = Fft2Shape::new(2, 8);
    let tiles_per_ciphertext = slots / shape.elements();

    let tile: Vec<Complex64> = (0..16)
        .map(|i| Complex64::new((i + 1) as f64 / 32.0, (i % 3) as f64 / 64.0))
        .collect();

    let input: Vec<Complex64> = (0..slots).map(|i| tile[i % 16]).collect();

    let secret: Vec<i8> = (0..degree)
        .map(|i| match i % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();

    let distribution = ErrorDistribution::DiscreteGaussian { sigma: 3.19 };

    let top_basis = chain.top().clone();
    let top_plan = RnsNttPlan::new(top_basis.moduli().to_vec(), degree);

    let plaintext = encode_rns(&input, &embedding, &top_basis, scale);

    let initial: Vec<RnsCkksCiphertext> = (0..4_u64)
        .map(|i| {
            let mut rng = ChaCha20Rng::seed_from_u64(0xA013_E000 + i);

            let encrypted = encrypt_rns_raw_with_distribution_ntt_rng(
                &plaintext,
                2,
                distribution,
                &secret,
                &top_plan,
                &mut rng,
            );

            RnsCkksCiphertext::new(encrypted, CkksChainState::top(&chain, scale), &chain)
        })
        .collect();

    let auxiliary_basis = ModulusBasis::new(vec![Modulus::new(114_689), Modulus::new(786_433)]);

    let mut key_material = Vec::new();

    for (stage_index, span) in [8_usize, 4, 2].into_iter().enumerate() {
        let q = chain.level(stage_index).clone();

        let mut qp_moduli = q.moduli().to_vec();
        qp_moduli.extend_from_slice(auxiliary_basis.moduli());

        let qp_plan = RnsNttPlan::new(qp_moduli, degree);

        let diagonals = PackedFft2DifStageDiagonals::new(
            shape,
            PackedFft2Axis::Rows,
            span,
            FftDirection::Forward,
        );

        let rotation = diagonals.rotation();
        let left_exponent = rotation_exponent_left(degree, rotation);
        let right_exponent = rotation_exponent_right(degree, rotation);

        let make_key = |exponent: usize, seed: u64| {
            let mut rng = ChaCha20Rng::seed_from_u64(seed);

            PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                BoundedGadgetLayout::new(q.clone(), 8),
                auxiliary_basis.clone(),
                &secret,
                exponent,
                distribution,
                2,
                &qp_plan,
                &mut rng,
            )
        };

        let left_key = make_key(left_exponent, 0xA013_E100 + 2 * stage_index as u64);

        let right_key = make_key(right_exponent, 0xA013_E101 + 2 * stage_index as u64);

        key_material.push((left_key, right_key, qp_plan));
    }

    let mut rotations = PreparedHybridRotationContext::new(&chain);

    for (stage_index, (left, right, plan)) in key_material.iter().enumerate() {
        rotations.insert(stage_index, left, plan);

        if right.exponent() != left.exponent() {
            rotations.insert(stage_index, right, plan);
        }
    }

    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let staged = execute_packed_local_rows_cp_hybrid_staged(
        initial.clone(),
        (shape, tiles_per_ciphertext),
        &evaluator,
        &rotations,
        &embedding,
        &chain,
        2,
    );

    let mut manual = initial;

    for span in [8_usize, 4, 2] {
        let level = manual[0].level();

        let q_plan = RnsNttPlan::new(manual[0].basis().moduli().to_vec(), degree);

        let prepared_q = PreparedRnsNttPlan::new(&q_plan);

        let diagonals = PackedFft2DifStageDiagonals::new(
            shape,
            PackedFft2Axis::Rows,
            span,
            FftDirection::Forward,
        );

        let prepared_stage = PreparedRepeatedPackedFft2DifStage::new(
            &manual[0],
            &diagonals,
            tiles_per_ciphertext,
            &embedding,
            &chain,
            &q_plan,
        );

        manual = execute_packed_local_row_stage_cp_hybrid_batch(
            &manual,
            &prepared_stage,
            &evaluator,
            &rotations,
            &chain,
            &prepared_q,
            1,
        );

        assert_eq!(manual[0].level(), level + 1);
    }

    assert_eq!(staged, manual);
    assert_eq!(staged.len(), 4);

    let mut expected = input.clone();

    for span in [8_usize, 4, 2] {
        let diagonals = PackedFft2DifStageDiagonals::new(
            shape,
            PackedFft2Axis::Rows,
            span,
            FftDirection::Forward,
        );

        expected =
            execute_repeated_packed_fft2_dif_stage_pp(&expected, &diagonals, tiles_per_ciphertext);
    }

    for (index, result) in staged.iter().enumerate() {
        assert_eq!(result.level(), 3);

        let result_plan = RnsNttPlan::new(result.basis().moduli().to_vec(), degree);

        let decrypted = decrypt_rns_raw_with_ntt(result.rlwe(), &secret, &result_plan);

        let modulus = composite_modulus_big(decrypted.basis());

        let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
            .into_iter()
            .map(|value| {
                centered_representative_big(&value, &modulus)
                    .to_f64()
                    .expect("CKKS coefficient fits f64")
                    / result.scale()
            })
            .collect();

        let actual = embedding.coefficients_to_slots(&coefficients);

        let mut max_abs = 0.0_f64;
        let mut error_sq = 0.0_f64;
        let mut reference_sq = 0.0_f64;

        for (a, e) in actual.iter().zip(expected.iter()) {
            let error = (*a - *e).norm();
            max_abs = max_abs.max(error);
            error_sq += error * error;
            reference_sq += e.norm_sqr();
        }

        let rel_l2 = (error_sq / reference_sq).sqrt();

        println!(
            "M1_3F_CASE={} level={} max_abs={:.9e} rel_l2={:.9e}",
            index,
            result.level(),
            max_abs,
            rel_l2
        );

        assert!(
            max_abs < 5.0e-3 && rel_l2 < 5.0e-3,
            "three-stage Gaussian hybrid FFT numerical error"
        );
    }

    println!("M1_3F_THREE_STAGE_HYBRID_ROW_STATUS=PASS");
}

#[test]
fn gaussian_hybrid_four_stage_row_fft() {
    use crate::application_support::ckks::encode_rns;
    use crate::ckks::{CkksChainState, PreparedHybridRotationContext, RnsCkksEvaluationKeys};
    use crate::grafting::{
        decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
    };
    use crate::ring::{
        centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, Modulus,
        ModulusBasis,
    };
    use crate::rlwe::ErrorDistribution;
    use num_traits::ToPrimitive;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    let profile = crate::ckks::CkksParameterProfile::new(
        "hybrid-four-stage-correctness",
        4096,
        &[
            0x3ffffffdf0001,
            0x7fffba001,
            0x7fffb0001,
            0x7ffff6001,
            0x7fffa4001,
        ],
        2.0_f64.powi(35),
        crate::ckks::CkksParameterClass::Research,
        None,
    );
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let chain = profile.modulus_chain();
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();

    assert!(chain.max_level() >= 4);

    let shape = Fft2Shape::new(2, 16);
    let tiles_per_ciphertext = slots / shape.elements();

    let tile: Vec<Complex64> = (0..32)
        .map(|i| Complex64::new((i + 1) as f64 / 32.0, (i % 3) as f64 / 64.0))
        .collect();

    let input: Vec<Complex64> = (0..slots).map(|i| tile[i % 32]).collect();

    let secret: Vec<i8> = (0..degree)
        .map(|i| match i % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();

    let distribution = ErrorDistribution::DiscreteGaussian { sigma: 3.19 };

    let top_basis = chain.top().clone();
    let top_plan = RnsNttPlan::new(top_basis.moduli().to_vec(), degree);

    let plaintext = encode_rns(&input, &embedding, &top_basis, scale);

    let initial: Vec<RnsCkksCiphertext> = (0..4_u64)
        .map(|i| {
            let mut rng = ChaCha20Rng::seed_from_u64(0xA013_E000 + i);

            let encrypted = encrypt_rns_raw_with_distribution_ntt_rng(
                &plaintext,
                2,
                distribution,
                &secret,
                &top_plan,
                &mut rng,
            );

            RnsCkksCiphertext::new(encrypted, CkksChainState::top(&chain, scale), &chain)
        })
        .collect();

    let auxiliary_basis = ModulusBasis::new(vec![Modulus::new(114_689), Modulus::new(786_433)]);

    let mut key_material = Vec::new();

    for (stage_index, span) in [16_usize, 8, 4, 2].into_iter().enumerate() {
        let q = chain.level(stage_index).clone();

        let mut qp_moduli = q.moduli().to_vec();
        qp_moduli.extend_from_slice(auxiliary_basis.moduli());

        let qp_plan = RnsNttPlan::new(qp_moduli, degree);

        let diagonals = PackedFft2DifStageDiagonals::new(
            shape,
            PackedFft2Axis::Rows,
            span,
            FftDirection::Forward,
        );

        let rotation = diagonals.rotation();
        let left_exponent = rotation_exponent_left(degree, rotation);
        let right_exponent = rotation_exponent_right(degree, rotation);

        let make_key = |exponent: usize, seed: u64| {
            let mut rng = ChaCha20Rng::seed_from_u64(seed);

            PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                BoundedGadgetLayout::new(q.clone(), 8),
                auxiliary_basis.clone(),
                &secret,
                exponent,
                distribution,
                2,
                &qp_plan,
                &mut rng,
            )
        };

        let left_key = make_key(left_exponent, 0xA013_E100 + 2 * stage_index as u64);

        let right_key = make_key(right_exponent, 0xA013_E101 + 2 * stage_index as u64);

        key_material.push((left_key, right_key, qp_plan));
    }

    let mut rotations = PreparedHybridRotationContext::new(&chain);

    for (stage_index, (left, right, plan)) in key_material.iter().enumerate() {
        rotations.insert(stage_index, left, plan);

        if right.exponent() != left.exponent() {
            rotations.insert(stage_index, right, plan);
        }
    }

    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let staged = execute_packed_local_rows_cp_hybrid_staged(
        initial.clone(),
        (shape, tiles_per_ciphertext),
        &evaluator,
        &rotations,
        &embedding,
        &chain,
        2,
    );

    let mut manual = initial;

    for span in [16_usize, 8, 4, 2] {
        let level = manual[0].level();

        let q_plan = RnsNttPlan::new(manual[0].basis().moduli().to_vec(), degree);

        let prepared_q = PreparedRnsNttPlan::new(&q_plan);

        let diagonals = PackedFft2DifStageDiagonals::new(
            shape,
            PackedFft2Axis::Rows,
            span,
            FftDirection::Forward,
        );

        let prepared_stage = PreparedRepeatedPackedFft2DifStage::new(
            &manual[0],
            &diagonals,
            tiles_per_ciphertext,
            &embedding,
            &chain,
            &q_plan,
        );

        manual = execute_packed_local_row_stage_cp_hybrid_batch(
            &manual,
            &prepared_stage,
            &evaluator,
            &rotations,
            &chain,
            &prepared_q,
            1,
        );

        assert_eq!(manual[0].level(), level + 1);
    }

    assert_eq!(staged, manual);
    assert_eq!(staged.len(), 4);

    let mut expected = input.clone();

    for span in [16_usize, 8, 4, 2] {
        let diagonals = PackedFft2DifStageDiagonals::new(
            shape,
            PackedFft2Axis::Rows,
            span,
            FftDirection::Forward,
        );

        expected =
            execute_repeated_packed_fft2_dif_stage_pp(&expected, &diagonals, tiles_per_ciphertext);
    }

    for (index, result) in staged.iter().enumerate() {
        assert_eq!(result.level(), 4);

        let result_plan = RnsNttPlan::new(result.basis().moduli().to_vec(), degree);

        let decrypted = decrypt_rns_raw_with_ntt(result.rlwe(), &secret, &result_plan);

        let modulus = composite_modulus_big(decrypted.basis());

        let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
            .into_iter()
            .map(|value| {
                centered_representative_big(&value, &modulus)
                    .to_f64()
                    .expect("CKKS coefficient fits f64")
                    / result.scale()
            })
            .collect();

        let actual = embedding.coefficients_to_slots(&coefficients);

        let mut max_abs = 0.0_f64;
        let mut error_sq = 0.0_f64;
        let mut reference_sq = 0.0_f64;

        for (a, e) in actual.iter().zip(expected.iter()) {
            let error = (*a - *e).norm();
            max_abs = max_abs.max(error);
            error_sq += error * error;
            reference_sq += e.norm_sqr();
        }

        let rel_l2 = (error_sq / reference_sq).sqrt();

        println!(
            "M1_3G_CASE={} level={} max_abs={:.9e} rel_l2={:.9e}",
            index,
            result.level(),
            max_abs,
            rel_l2
        );

        assert!(
            max_abs < 5.0e-3 && rel_l2 < 5.0e-3,
            "four-stage Gaussian hybrid FFT numerical error"
        );
    }

    println!("M1_3G_FOUR_STAGE_HYBRID_ROW_STATUS=PASS");
}

#[test]
fn gaussian_hybrid_six_stage_row_fft() {
    use crate::application_support::ckks::encode_rns;
    use crate::ckks::{CkksChainState, PreparedHybridRotationContext, RnsCkksEvaluationKeys};
    use crate::grafting::{
        decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
    };
    use crate::ring::{
        centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, Modulus,
        ModulusBasis,
    };
    use crate::rlwe::ErrorDistribution;
    use num_traits::ToPrimitive;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    let profile = crate::ckks::CkksParameterProfile::new(
        "hybrid-six-stage-correctness",
        4096,
        &[
            0x3ffffffdf0001,
            0x7fffba001,
            0x7fffb0001,
            0x7ffff6001,
            0x7fffa4001,
            0x7fff80001,
            0x7fff7e001,
        ],
        2.0_f64.powi(35),
        crate::ckks::CkksParameterClass::Research,
        None,
    );
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let chain = profile.modulus_chain();
    let embedding = CkksCanonicalEmbedding::new(degree);
    let slots = embedding.slot_count();

    assert!(chain.max_level() >= 6);

    let shape = Fft2Shape::new(2, 64);
    let tiles_per_ciphertext = slots / shape.elements();

    let tile: Vec<Complex64> = (0..128)
        .map(|i| Complex64::new((i + 1) as f64 / 32.0, (i % 3) as f64 / 64.0))
        .collect();

    let input: Vec<Complex64> = (0..slots).map(|i| tile[i % 128]).collect();

    let secret: Vec<i8> = (0..degree)
        .map(|i| match i % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();

    let distribution = ErrorDistribution::DiscreteGaussian { sigma: 3.19 };

    let top_basis = chain.top().clone();
    let top_plan = RnsNttPlan::new(top_basis.moduli().to_vec(), degree);

    let plaintext = encode_rns(&input, &embedding, &top_basis, scale);

    let initial: Vec<RnsCkksCiphertext> = (0..4_u64)
        .map(|i| {
            let mut rng = ChaCha20Rng::seed_from_u64(0xA013_E000 + i);

            let encrypted = encrypt_rns_raw_with_distribution_ntt_rng(
                &plaintext,
                2,
                distribution,
                &secret,
                &top_plan,
                &mut rng,
            );

            RnsCkksCiphertext::new(encrypted, CkksChainState::top(&chain, scale), &chain)
        })
        .collect();

    let auxiliary_basis = ModulusBasis::new(vec![Modulus::new(114_689), Modulus::new(786_433)]);

    let mut key_material = Vec::new();

    for (stage_index, span) in [64_usize, 32, 16, 8, 4, 2].into_iter().enumerate() {
        let q = chain.level(stage_index).clone();

        let mut qp_moduli = q.moduli().to_vec();
        qp_moduli.extend_from_slice(auxiliary_basis.moduli());

        let qp_plan = RnsNttPlan::new(qp_moduli, degree);

        let diagonals = PackedFft2DifStageDiagonals::new(
            shape,
            PackedFft2Axis::Rows,
            span,
            FftDirection::Forward,
        );

        let rotation = diagonals.rotation();
        let left_exponent = rotation_exponent_left(degree, rotation);
        let right_exponent = rotation_exponent_right(degree, rotation);

        let make_key = |exponent: usize, seed: u64| {
            let mut rng = ChaCha20Rng::seed_from_u64(seed);

            PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                BoundedGadgetLayout::new(q.clone(), 8),
                auxiliary_basis.clone(),
                &secret,
                exponent,
                distribution,
                2,
                &qp_plan,
                &mut rng,
            )
        };

        let left_key = make_key(left_exponent, 0xA013_E100 + 2 * stage_index as u64);

        let right_key = make_key(right_exponent, 0xA013_E101 + 2 * stage_index as u64);

        key_material.push((left_key, right_key, qp_plan));
    }

    let mut rotations = PreparedHybridRotationContext::new(&chain);

    for (stage_index, (left, right, plan)) in key_material.iter().enumerate() {
        rotations.insert(stage_index, left, plan);

        if right.exponent() != left.exponent() {
            rotations.insert(stage_index, right, plan);
        }
    }

    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let staged = execute_packed_local_rows_cp_hybrid_staged(
        initial.clone(),
        (shape, tiles_per_ciphertext),
        &evaluator,
        &rotations,
        &embedding,
        &chain,
        2,
    );

    let mut cache = PackedRowPreparationContext::new();

    let first = execute_packed_local_rows_cp_hybrid_staged_cached(
        initial.clone(),
        (shape, tiles_per_ciphertext),
        &evaluator,
        &rotations,
        &embedding,
        &chain,
        (2, &mut cache),
    );

    assert_eq!(first, staged);
    assert_eq!(cache.hybrid_preparations, 6);
    assert_eq!(cache.hybrid_hits, 0);

    let second = execute_packed_local_rows_cp_hybrid_staged_cached(
        initial.clone(),
        (shape, tiles_per_ciphertext),
        &evaluator,
        &rotations,
        &embedding,
        &chain,
        (2, &mut cache),
    );

    assert_eq!(second, staged);
    assert_eq!(cache.hybrid_preparations, 6);
    assert_eq!(cache.hybrid_hits, 6);

    println!(
        "M1_4D_HYBRID_CACHE_STATUS=PASS preparations={} hits={}",
        cache.hybrid_preparations, cache.hybrid_hits,
    );

    let mut manual = initial;

    for span in [64_usize, 32, 16, 8, 4, 2] {
        let level = manual[0].level();

        let q_plan = RnsNttPlan::new(manual[0].basis().moduli().to_vec(), degree);

        let prepared_q = PreparedRnsNttPlan::new(&q_plan);

        let diagonals = PackedFft2DifStageDiagonals::new(
            shape,
            PackedFft2Axis::Rows,
            span,
            FftDirection::Forward,
        );

        let prepared_stage = PreparedRepeatedPackedFft2DifStage::new(
            &manual[0],
            &diagonals,
            tiles_per_ciphertext,
            &embedding,
            &chain,
            &q_plan,
        );

        manual = execute_packed_local_row_stage_cp_hybrid_batch(
            &manual,
            &prepared_stage,
            &evaluator,
            &rotations,
            &chain,
            &prepared_q,
            1,
        );

        assert_eq!(manual[0].level(), level + 1);
    }

    assert_eq!(staged, manual);
    assert_eq!(staged.len(), 4);

    let mut expected = input.clone();

    for span in [64_usize, 32, 16, 8, 4, 2] {
        let diagonals = PackedFft2DifStageDiagonals::new(
            shape,
            PackedFft2Axis::Rows,
            span,
            FftDirection::Forward,
        );

        expected =
            execute_repeated_packed_fft2_dif_stage_pp(&expected, &diagonals, tiles_per_ciphertext);
    }

    for (index, result) in staged.iter().enumerate() {
        assert_eq!(result.level(), 6);

        let result_plan = RnsNttPlan::new(result.basis().moduli().to_vec(), degree);

        let decrypted = decrypt_rns_raw_with_ntt(result.rlwe(), &secret, &result_plan);

        let modulus = composite_modulus_big(decrypted.basis());

        let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
            .into_iter()
            .map(|value| {
                centered_representative_big(&value, &modulus)
                    .to_f64()
                    .expect("CKKS coefficient fits f64")
                    / result.scale()
            })
            .collect();

        let actual = embedding.coefficients_to_slots(&coefficients);

        let mut max_abs = 0.0_f64;
        let mut error_sq = 0.0_f64;
        let mut reference_sq = 0.0_f64;

        for (a, e) in actual.iter().zip(expected.iter()) {
            let error = (*a - *e).norm();
            max_abs = max_abs.max(error);
            error_sq += error * error;
            reference_sq += e.norm_sqr();
        }

        let rel_l2 = (error_sq / reference_sq).sqrt();

        println!(
            "M1_3H_CASE={} level={} max_abs={:.9e} rel_l2={:.9e}",
            index,
            result.level(),
            max_abs,
            rel_l2
        );

        assert!(
            max_abs < 5.0e-3 && rel_l2 < 5.0e-3,
            "six-stage Gaussian hybrid FFT numerical error"
        );
    }

    println!("M1_3H_SIX_STAGE_HYBRID_ROW_STATUS=PASS");
}

#[test]
fn degree_65536_hybrid_qp_preflight() {
    use crate::grafting::BoundedGadgetLayout;
    use crate::ring::{Modulus, ModulusBasis, RnsNttPlan};

    const DEGREE: usize = 65_536;
    const P0: u64 = 0x7ffffffbee0001;
    const P1: u64 = 0x7ffffffbe40001;

    let auxiliary = ModulusBasis::new(vec![Modulus::new(P0), Modulus::new(P1)]);

    let p_plan = RnsNttPlan::new(auxiliary.moduli().to_vec(), DEGREE);

    assert_eq!(p_plan.degree(), DEGREE);
    assert_eq!(p_plan.moduli(), auxiliary.moduli());

    let configurations = [("1K", 20, 2, 4), ("UHD", 24, 2, 6), ("8K", 26, 2, 7)];

    for (name, depth, guard, first_level) in configurations {
        let profile = crate::ckks::research_profile_65536_for_levels(depth, guard);
        let chain = profile.modulus_chain();

        for level in [first_level, first_level + 5] {
            let q = chain.level(level).clone();

            for modulus in q.moduli() {
                assert_ne!(modulus.value(), P0);
                assert_ne!(modulus.value(), P1);
            }

            let mut qp_moduli = q.moduli().to_vec();
            qp_moduli.extend_from_slice(auxiliary.moduli());

            let qp_plan = RnsNttPlan::new(qp_moduli, DEGREE);

            assert_eq!(qp_plan.degree(), DEGREE);
            assert_eq!(qp_plan.moduli().len(), q.len() + auxiliary.len(),);

            for base_log in [32, 40, 55] {
                let layout = BoundedGadgetLayout::new(q.clone(), base_log);

                let digits = layout.digit_count();

                let key_bytes =
                    2_u128 * DEGREE as u128 * qp_plan.moduli().len() as u128 * digits as u128 * 8;

                println!(
                    "M1_5C_QP_CASE=PASS workload={} level={} base={}                      q_limbs={} p_limbs={} digits={}                      estimated_key_mib={:.2}",
                    name,
                    level,
                    base_log,
                    q.len(),
                    auxiliary.len(),
                    digits,
                    key_bytes as f64 / 1_048_576.0,
                );
            }
        }
    }

    println!("M1_5C_DEGREE_65536_QP_STATUS=PASS");
}

#[test]
#[ignore = "allocates a production-sized Gaussian hybrid key"]
fn degree_65536_single_hybrid_key_smoke() {
    use crate::ckks::{rotation_exponent_left, PreparedHybridRnsGaloisKey};
    use crate::grafting::BoundedGadgetLayout;
    use crate::ring::{Modulus, ModulusBasis, RnsNttPlan};
    use crate::rlwe::ErrorDistribution;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    const N: usize = 65_536;
    let level: usize = std::env::var("FHE_HYBRID_TEST_LEVEL")
        .ok()
        .map(|value| value.parse().expect("invalid test level"))
        .unwrap_or(9);

    assert!([4, 9].contains(&level));
    let base_log: u32 = std::env::var("FHE_HYBRID_TEST_BASE_LOG")
        .ok()
        .map(|value| value.parse().expect("invalid gadget base"))
        .unwrap_or(32);

    assert!(
        [32, 40, 55].contains(&base_log),
        "unsupported experimental gadget base"
    );

    let profile = crate::ckks::research_profile_65536_for_levels(20, 2);
    let chain = profile.modulus_chain();

    let q = chain.level(level).clone();

    let p = ModulusBasis::new(vec![
        Modulus::new(0x7ffffffbee0001),
        Modulus::new(0x7ffffffbe40001),
    ]);

    let mut qp_moduli = q.moduli().to_vec();
    qp_moduli.extend_from_slice(p.moduli());

    let qp_plan = RnsNttPlan::new(qp_moduli, N);
    let layout = BoundedGadgetLayout::new(q, base_log);
    let expected_digits = layout.digit_count();

    assert_eq!(
        expected_digits,
        match (level, base_log) {
            (4, 32) => 31,
            (4, 40) => 25,
            (4, 55) => 19,
            (9, 32) => 23,
            (9, 40) => 18,
            (9, 55) => 14,
            _ => unreachable!(),
        },
    );

    let secret: Vec<i8> = (0..N)
        .map(|index| match index % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        })
        .collect();

    let stage_span = match level {
        4 => 64_usize,
        9 => 2_usize,
        _ => unreachable!(),
    };
    let rotation_steps = stage_span / 2;

    let exponent = rotation_exponent_left(N, rotation_steps);

    // Deterministic research fixture only.
    // Production key generation must use fresh OS entropy.
    let mut rng = ChaCha20Rng::seed_from_u64(0x5150_3635_3533_3609);

    let started = std::time::Instant::now();

    let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
        layout,
        p.clone(),
        &secret,
        exponent,
        ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
        2,
        &qp_plan,
        &mut rng,
    );

    let elapsed_ms = started.elapsed().as_millis();

    assert_eq!(key.degree(), N);
    assert_eq!(key.exponent(), exponent);
    assert_eq!(key.digit_count(), expected_digits);
    assert_eq!(key.auxiliary_basis(), &p);

    // Independent sparse-polynomial automorphism oracle.
    // The expected result is computed from X^N = -1 and
    // X^i -> X^(i * exponent), without calling the rotation code.
    use crate::ckks::{CkksChainState, PreparedHybridRotationContext};
    use crate::grafting::{decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng};
    use crate::ring::{
        centered_representative_big, composite_modulus_big, reconstruct_coefficients_big,
        Polynomial, RnsPolynomial,
    };
    use num_bigint::BigInt;
    use num_traits::{Signed, ToPrimitive};

    let sparse = [
        (0_usize, 1_i64 << 32),
        (1, 2_i64 << 32),
        (17, 3_i64 << 32),
        (N / 2 + 7, 4_i64 << 32),
        (N - 1, 5_i64 << 32),
    ];

    let mut message = vec![0_i64; N];

    for (index, coefficient) in sparse {
        message[index] = coefficient;
    }

    let plaintext = RnsPolynomial::from_residues(
        chain
            .level(level)
            .moduli()
            .iter()
            .copied()
            .map(|modulus| {
                Polynomial::new(
                    modulus,
                    message
                        .iter()
                        .map(|&value| value.rem_euclid(modulus.value() as i64) as u64)
                        .collect(),
                )
            })
            .collect(),
    );

    let q_plan = RnsNttPlan::new(chain.level(level).moduli().to_vec(), N);

    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x5150_3635_3533_3610);

    let encrypted = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
        &secret,
        &q_plan,
        &mut encryption_rng,
    );

    let ciphertext = RnsCkksCiphertext::new(
        encrypted,
        CkksChainState::new(&chain, level, profile.initial_scale()),
        &chain,
    );

    let right_exponent = crate::ckks::rotation_exponent_right(N, rotation_steps);

    assert_ne!(exponent, right_exponent);

    let mut right_rng = ChaCha20Rng::seed_from_u64(0x5150_3635_3533_3611);

    let right_started = std::time::Instant::now();

    let right_key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
        BoundedGadgetLayout::new(chain.level(level).clone(), base_log),
        p.clone(),
        &secret,
        right_exponent,
        ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
        2,
        &qp_plan,
        &mut right_rng,
    );

    let right_keygen_ms = right_started.elapsed().as_millis();
    assert_eq!(right_key.degree(), N);
    assert_eq!(right_key.digit_count(), expected_digits);

    let mut rotation_context = PreparedHybridRotationContext::new(&chain);

    rotation_context.insert(level, &key, &qp_plan);
    rotation_context.insert(level, &right_key, &qp_plan);

    let rotation_start = std::time::Instant::now();
    let rotated = rotation_context.rotate_left(&ciphertext, rotation_steps);
    let rotation_ms = rotation_start.elapsed().as_millis();

    assert_eq!(rotated.level(), level);
    assert_eq!(rotated.state(), ciphertext.state());

    let decrypted = decrypt_rns_raw_with_ntt(rotated.rlwe(), &secret, &q_plan);

    let modulus = composite_modulus_big(decrypted.basis());
    let reconstructed = reconstruct_coefficients_big(&decrypted);

    assert_eq!(reconstructed.len(), N);

    // Compute the expected negacyclic automorphism independently.
    let mut expected = vec![0_i64; N];

    for (index, coefficient) in sparse {
        let mapped = (index * exponent) % (2 * N);

        if mapped < N {
            expected[mapped] += coefficient;
        } else {
            expected[mapped - N] -= coefficient;
        }
    }

    let mut max_abs_error = BigInt::from(0);
    let mut error_count = 0usize;

    for (index, residue) in reconstructed.iter().enumerate() {
        let actual = centered_representative_big(residue, &modulus);

        let error = (actual - BigInt::from(expected[index])).abs();

        if error > BigInt::from(0) {
            error_count += 1;
        }

        if error > max_abs_error {
            max_abs_error = error;
        }
    }

    // Exploratory coefficient-domain tolerance.
    // Must be reviewed against actual noise and CKKS precision.
    let tolerance = BigInt::from(1_i64 << 20);

    println!(
        "M1_5D_ROTATION_METRICS degree={} level={}          base={} p_limbs={} keygen_ms={} rotation_ms={}          max_abs_coeff_error={} nonzero_error_coeffs={}          tolerance={}",
        N,
        level,
        base_log,
        p.len(),
        elapsed_ms,
        rotation_ms,
        max_abs_error,
        error_count,
        tolerance,
    );

    assert!(
        max_abs_error <= tolerance,
        "degree-65536 hybrid rotation exceeded coefficient tolerance"
    );

    println!("M1_5D_ROTATION_ORACLE_STATUS=PASS");

    let right_started = std::time::Instant::now();
    let right_rotated = rotation_context.rotate_right(&ciphertext, rotation_steps);
    let right_rotation_ms = right_started.elapsed().as_millis();

    assert_eq!(right_rotated.level(), level);
    assert_eq!(right_rotated.state(), ciphertext.state());

    let right_decrypted = decrypt_rns_raw_with_ntt(right_rotated.rlwe(), &secret, &q_plan);

    let right_modulus = composite_modulus_big(right_decrypted.basis());
    let right_coefficients = reconstruct_coefficients_big(&right_decrypted);

    let mut right_expected = vec![0_i64; N];

    for (index, coefficient) in sparse {
        let mapped = (index * right_exponent) % (2 * N);

        if mapped < N {
            right_expected[mapped] += coefficient;
        } else {
            right_expected[mapped - N] -= coefficient;
        }
    }

    let mut right_max_error = BigInt::from(0);
    let mut right_nonzero_errors = 0usize;

    for (index, residue) in right_coefficients.iter().enumerate() {
        let actual = centered_representative_big(residue, &right_modulus);

        let error = (actual - BigInt::from(right_expected[index])).abs();

        if error > BigInt::from(0) {
            right_nonzero_errors += 1;
        }

        if error > right_max_error {
            right_max_error = error;
        }
    }

    println!(
        "M1_5G_RIGHT_ROTATION_METRICS level={} base={}          keygen_ms={} rotation_ms={} max_abs_coeff_error={}          nonzero_error_coeffs={}",
        level,
        base_log,
        right_keygen_ms,
        right_rotation_ms,
        right_max_error,
        right_nonzero_errors,
    );

    assert!(
        right_max_error <= tolerance,
        "right rotation exceeded coefficient tolerance"
    );

    println!("M1_5G_BIDIRECTIONAL_ROTATION_STATUS=PASS");

    // Complete packed CKKS butterfly at the active modulus-chain level.
    // Reuses the two Gaussian hybrid keys already generated above.
    {
        use crate::application_support::ckks::encode_rns;
        use crate::ckks::{RnsCkksEvaluationKeys, RnsCkksEvaluator};

        let embedding = CkksCanonicalEmbedding::new(N);
        let shape = Fft2Shape::new(2, 64);
        let tile_size = shape.elements();
        let tiles_per_ciphertext = embedding.slot_count() / tile_size;

        let input_slots: Vec<Complex64> = (0..embedding.slot_count())
            .map(|index| {
                let i = index % tile_size;
                Complex64::new((i + 1) as f64 / 256.0, (i % 7) as f64 / 512.0)
            })
            .collect();

        let plaintext = encode_rns(
            &input_slots,
            &embedding,
            chain.level(level),
            profile.initial_scale(),
        );

        let mut fft_rng = ChaCha20Rng::seed_from_u64(0x5150_3635_3533_3620);

        let encrypted_fft = encrypt_rns_raw_with_distribution_ntt_rng(
            &plaintext,
            2,
            ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
            &secret,
            &q_plan,
            &mut fft_rng,
        );

        let fft_input = RnsCkksCiphertext::new(
            encrypted_fft,
            CkksChainState::new(&chain, level, profile.initial_scale()),
            &chain,
        );

        let diagonals = PackedFft2DifStageDiagonals::new(
            shape,
            PackedFft2Axis::Rows,
            stage_span,
            FftDirection::Forward,
        );

        assert_eq!(diagonals.rotation(), rotation_steps);

        let prepared_fft = PreparedRepeatedPackedFft2DifStage::new(
            &fft_input,
            &diagonals,
            tiles_per_ciphertext,
            &embedding,
            &chain,
            &q_plan,
        );

        let prepared_q = PreparedRnsNttPlan::new(&q_plan);

        let evaluation_keys = RnsCkksEvaluationKeys::new();
        let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

        let butterfly_start = std::time::Instant::now();

        let outputs = execute_packed_local_row_stage_cp_hybrid_batch(
            &[fft_input],
            &prepared_fft,
            &evaluator,
            &rotation_context,
            &chain,
            &prepared_q,
            1,
        );

        let butterfly_ms = butterfly_start.elapsed().as_millis();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].level(), level + 1);

        let reference = execute_repeated_packed_fft2_dif_stage_pp(
            &input_slots,
            &diagonals,
            tiles_per_ciphertext,
        );

        let output = &outputs[0];
        let output_plan = RnsNttPlan::new(output.basis().moduli().to_vec(), N);

        let decrypted = decrypt_rns_raw_with_ntt(output.rlwe(), &secret, &output_plan);

        let modulus = composite_modulus_big(decrypted.basis());

        let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
            .iter()
            .map(|residue| {
                centered_representative_big(residue, &modulus)
                    .to_f64()
                    .expect("CKKS coefficient conversion failed")
                    / output.scale()
            })
            .collect();

        let decoded = embedding.coefficients_to_slots(&coefficients);

        assert_eq!(decoded.len(), reference.len());

        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        let mut max_abs_error = 0.0_f64;

        for (actual, expected) in decoded.iter().zip(reference.iter()) {
            let error = (*actual - *expected).norm();

            squared_error += error * error;
            squared_reference += expected.norm_sqr();
            max_abs_error = max_abs_error.max(error);
        }

        let relative_l2 = (squared_error / squared_reference).sqrt();

        println!(
            "M1_5H_PACKED_BUTTERFLY_METRICS              level_in={} level_out={} degree={}              tiles={} butterfly_ms={}              rel_l2={:.12e} max_abs={:.12e}",
            level,
            output.level(),
            N,
            tiles_per_ciphertext,
            butterfly_ms,
            relative_l2,
            max_abs_error,
        );

        assert!(relative_l2.is_finite());
        assert!(max_abs_error.is_finite());
        assert!(
            relative_l2 <= 5e-3,
            "degree-65536 packed hybrid butterfly accuracy failed"
        );

        println!(
            "M1_5I_STAGE_METRICS level={} span={} rotation={}",
            level, stage_span, rotation_steps,
        );

        println!("M1_5H_PACKED_BUTTERFLY_STATUS=PASS");
    }

    println!(
        "M1_5C_SINGLE_KEY_STATUS=PASS          degree={} level={} base={} digits={}          p_limbs={} keygen_ms={}",
        N,
        level,
        base_log,
        key.digit_count(),
        p.len(),
        elapsed_ms,
    );
}

#[test]
#[ignore = "large degree-65536 six-stage Gaussian hybrid integration"]
fn degree_65536_six_stage_hybrid_row_fft() {
    use crate::application_support::ckks::encode_rns;
    use crate::ckks::{
        CkksChainState, PreparedHybridRnsGaloisKey, PreparedHybridRotationContext,
        RnsCkksEvaluationKeys,
    };
    use crate::grafting::{
        decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
    };
    use crate::ring::{
        centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, Modulus,
        ModulusBasis, RnsNttPlan,
    };
    use crate::rlwe::ErrorDistribution;
    use num_traits::ToPrimitive;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    assert_eq!(
        std::env::var("FHE_M1_6_ALLOW_FULL_KEYS").as_deref(),
        Ok("1"),
        "explicit full-key experiment opt-in required"
    );

    let ciphertext_count: usize = std::env::var("FHE_M1_7_CIPHERTEXTS")
        .ok()
        .map(|value| value.parse().expect("invalid ciphertext count"))
        .unwrap_or(1);

    assert!(
        [1, 2, 4, 8].contains(&ciphertext_count),
        "unsupported bounded ciphertext count"
    );

    let parallelism: usize = std::env::var("FHE_M1_7_PARALLELISM")
        .ok()
        .map(|value| value.parse().expect("invalid parallelism"))
        .unwrap_or(1);

    assert!(
        (1..=4).contains(&parallelism),
        "parallelism must be between 1 and 4"
    );

    const N: usize = 65_536;
    const START_LEVEL: usize = 4;
    const BASE_LOG: u32 = 55;
    const SPANS: [usize; 6] = [64, 32, 16, 8, 4, 2];

    let profile = crate::ckks::research_profile_65536_for_levels(20, 2);
    let chain = profile.modulus_chain();
    let embedding = CkksCanonicalEmbedding::new(N);
    let scale = profile.initial_scale();

    let shape = Fft2Shape::new(2, 64);
    let tiles_per_ciphertext = embedding.slot_count() / shape.elements();

    assert_eq!(tiles_per_ciphertext, 256);

    let auxiliary = ModulusBasis::new(vec![
        Modulus::new(0x7ffffffbee0001),
        Modulus::new(0x7ffffffbe40001),
    ]);

    let distribution = ErrorDistribution::DiscreteGaussian { sigma: 3.19 };

    let secret: Vec<i8> = (0..N)
        .map(|index| match index % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();

    let input: Vec<Complex64> = (0..embedding.slot_count())
        .map(|index| {
            let i = index % shape.elements();
            Complex64::new((i + 1) as f64 / 256.0, (i % 7) as f64 / 512.0)
        })
        .collect();

    let initial_basis = chain.level(START_LEVEL);
    let initial_plan = RnsNttPlan::new(initial_basis.moduli().to_vec(), N);

    let plaintext = encode_rns(&input, &embedding, initial_basis, scale);

    let ciphertexts: Vec<RnsCkksCiphertext> = (0..ciphertext_count)
        .map(|index| {
            let mut rng = ChaCha20Rng::seed_from_u64(0x5160_0000 + index as u64);

            let encrypted = encrypt_rns_raw_with_distribution_ntt_rng(
                &plaintext,
                2,
                distribution,
                &secret,
                &initial_plan,
                &mut rng,
            );

            RnsCkksCiphertext::new(
                encrypted,
                CkksChainState::new(&chain, START_LEVEL, scale),
                &chain,
            )
        })
        .collect();

    assert_eq!(ciphertexts.len(), ciphertext_count);

    // Generate one left/right key pair per CKKS level.
    // All key material remains externally owned.
    let mut key_material = Vec::with_capacity(SPANS.len());

    let keygen_start = std::time::Instant::now();

    for (stage, span) in SPANS.into_iter().enumerate() {
        let level = START_LEVEL + stage;
        let q = chain.level(level).clone();

        let mut qp_moduli = q.moduli().to_vec();
        qp_moduli.extend_from_slice(auxiliary.moduli());

        let qp_plan = RnsNttPlan::new(qp_moduli, N);
        let layout = BoundedGadgetLayout::new(q, BASE_LOG);

        let rotation = span / 2;
        let left_exponent = rotation_exponent_left(N, rotation);
        let right_exponent = rotation_exponent_right(N, rotation);

        let mut left_rng = ChaCha20Rng::seed_from_u64(0x5160_1000 + (stage as u64) * 2);

        let left = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
            layout.clone(),
            auxiliary.clone(),
            &secret,
            left_exponent,
            distribution,
            2,
            &qp_plan,
            &mut left_rng,
        );

        let mut right_rng = ChaCha20Rng::seed_from_u64(0x5160_1001 + (stage as u64) * 2);

        let right = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
            layout,
            auxiliary.clone(),
            &secret,
            right_exponent,
            distribution,
            2,
            &qp_plan,
            &mut right_rng,
        );

        println!(
            "M1_6_KEY_STAGE=PASS stage={} level={} \
             span={} rotation={} digits={}",
            stage + 1,
            level,
            span,
            rotation,
            left.digit_count(),
        );

        key_material.push((left, right, qp_plan));
    }

    let keygen_ms = keygen_start.elapsed().as_millis();

    let mut rotations = PreparedHybridRotationContext::new(&chain);

    for (stage, (left, right, plan)) in key_material.iter().enumerate() {
        let level = START_LEVEL + stage;
        rotations.insert(level, left, plan);
        rotations.insert(level, right, plan);
    }

    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let mut cache = PackedRowPreparationContext::new();

    let execute_start = std::time::Instant::now();

    let result = execute_packed_local_rows_cp_hybrid_staged_cached(
        ciphertexts,
        (shape, tiles_per_ciphertext),
        &evaluator,
        &rotations,
        &embedding,
        &chain,
        (parallelism, &mut cache),
    );

    let execute_ms = execute_start.elapsed().as_millis();

    assert_eq!(result.len(), ciphertext_count);
    assert_eq!(result[0].level(), START_LEVEL + 6);
    assert_eq!(cache.hybrid_preparations, 6);
    assert_eq!(cache.hybrid_hits, 0);

    let mut expected = input;

    for span in SPANS {
        let diagonals = PackedFft2DifStageDiagonals::new(
            shape,
            PackedFft2Axis::Rows,
            span,
            FftDirection::Forward,
        );

        expected =
            execute_repeated_packed_fft2_dif_stage_pp(&expected, &diagonals, tiles_per_ciphertext);
    }

    let mut worst_relative_l2 = 0.0_f64;
    let mut worst_max_abs = 0.0_f64;

    for (index, output) in result.iter().enumerate() {
        assert_eq!(output.level(), START_LEVEL + 6);

        let output_plan = RnsNttPlan::new(output.basis().moduli().to_vec(), N);

        let decrypted = decrypt_rns_raw_with_ntt(output.rlwe(), &secret, &output_plan);

        let modulus = composite_modulus_big(decrypted.basis());

        let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
            .iter()
            .map(|residue| {
                centered_representative_big(residue, &modulus)
                    .to_f64()
                    .expect("CKKS coefficient conversion failed")
                    / output.scale()
            })
            .collect();

        let decoded = embedding.coefficients_to_slots(&coefficients);

        assert_eq!(decoded.len(), expected.len());

        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        let mut max_abs_error = 0.0_f64;

        for (actual, reference) in decoded.iter().zip(expected.iter()) {
            let error = (*actual - *reference).norm();

            squared_error += error * error;
            squared_reference += reference.norm_sqr();
            max_abs_error = max_abs_error.max(error);
        }

        assert!(squared_reference > 0.0);

        let relative_l2 = (squared_error / squared_reference).sqrt();

        println!(
            "M1_7_CIPHERTEXT_METRICS index={}              rel_l2={:.12e} max_abs={:.12e}",
            index, relative_l2, max_abs_error,
        );

        assert!(relative_l2.is_finite());
        assert!(max_abs_error.is_finite());

        assert!(
            relative_l2 <= 5e-3,
            "six-stage hybrid FFT ciphertext accuracy failed"
        );

        worst_relative_l2 = worst_relative_l2.max(relative_l2);

        worst_max_abs = worst_max_abs.max(max_abs_error);
    }

    println!(
        "M1_7_MULTICT_METRICS degree={}          ciphertexts={} parallelism={} keys={}          keygen_ms={} execution_ms={}          worst_rel_l2={:.12e} worst_max_abs={:.12e}          cache_preparations={} cache_hits={}",
        N,
        ciphertext_count,
        parallelism,
        key_material.len() * 2,
        keygen_ms,
        execute_ms,
        worst_relative_l2,
        worst_max_abs,
        cache.hybrid_preparations,
        cache.hybrid_hits,
    );

    assert_eq!(cache.hybrid_preparations, 6);
    assert_eq!(cache.hybrid_hits, 0);

    println!("M1_7_MULTICT_STATUS=PASS");

    println!("M1_6_SIX_STAGE_STATUS=PASS");
}

#[test]
#[ignore = "full-degree Gaussian hybrid tile-block integration"]
fn degree_65536_gaussian_hybrid_tile_block() {
    use crate::application_support::ckks::encode_rns;
    use crate::ckks::{
        CkksChainState, PreparedHybridRnsGaloisKey, PreparedHybridRotationContext,
        RnsCkksEvaluationKeys,
    };
    use crate::grafting::{
        decrypt_rns_raw_with_ntt, encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout,
    };
    use crate::ring::{
        centered_representative_big, composite_modulus_big, reconstruct_coefficients_big, Modulus,
        ModulusBasis, RnsNttPlan,
    };
    use crate::rlwe::ErrorDistribution;
    use num_traits::ToPrimitive;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    assert_eq!(
        std::env::var("FHE_M1_9_ALLOW_FULL_KEYS").as_deref(),
        Ok("1"),
        "explicit full-degree experiment opt-in required"
    );

    const N: usize = 65_536;
    const LEVEL: usize = 11;
    const BASE_LOG: u32 = 55;

    let profile = crate::ckks::research_profile_65536_for_levels(20, 2);

    let chain = profile.modulus_chain();
    let embedding = CkksCanonicalEmbedding::new(N);
    let scale = profile.initial_scale();

    let diagonals = PackedTileDifStageDiagonals::new_column(64, 64, 8, 8, FftDirection::Forward);

    assert_eq!(diagonals.rotation(), 16_384);
    assert_eq!(
        diagonals.packed_tiles() * diagonals.tile_elements(),
        embedding.slot_count()
    );

    let input: Vec<Complex64> = (0..embedding.slot_count())
        .map(|index| {
            let tile = index / (64 * 64);
            let local = index % (64 * 64);

            Complex64::new(
                (tile + 1) as f64 / 16.0 + (local % 31) as f64 / 512.0,
                (local % 17) as f64 / 1024.0,
            )
        })
        .collect();

    let expected = execute_packed_tile_dif_stage_pp(&input, &diagonals);

    let secret: Vec<i8> = (0..N)
        .map(|index| match index % 4 {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 0,
        })
        .collect();

    let distribution = ErrorDistribution::DiscreteGaussian { sigma: 3.19 };

    let q = chain.level(LEVEL).clone();
    let q_plan = RnsNttPlan::new(q.moduli().to_vec(), N);

    let plaintext = encode_rns(&input, &embedding, &q, scale);

    let mut encryption_rng = ChaCha20Rng::seed_from_u64(0x519D_0000);

    let encrypted = encrypt_rns_raw_with_distribution_ntt_rng(
        &plaintext,
        2,
        distribution,
        &secret,
        &q_plan,
        &mut encryption_rng,
    );

    let ciphertext =
        RnsCkksCiphertext::new(encrypted, CkksChainState::new(&chain, LEVEL, scale), &chain);

    let auxiliary = ModulusBasis::new(vec![
        Modulus::new(0x7ffffffbee0001),
        Modulus::new(0x7ffffffbe40001),
    ]);

    let mut qp_moduli = q.moduli().to_vec();
    qp_moduli.extend_from_slice(auxiliary.moduli());

    let qp_plan = RnsNttPlan::new(qp_moduli, N);

    let rotation = diagonals.rotation();
    let left_exponent = rotation_exponent_left(N, rotation);
    let right_exponent = rotation_exponent_right(N, rotation);

    assert_eq!(
        left_exponent, right_exponent,
        "half-slot rotation must share one Galois exponent"
    );

    let layout = BoundedGadgetLayout::new(q, BASE_LOG);

    let keygen_start = std::time::Instant::now();

    let mut left_rng = ChaCha20Rng::seed_from_u64(0x519D_1000);

    let left = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
        layout.clone(),
        auxiliary.clone(),
        &secret,
        left_exponent,
        distribution,
        2,
        &qp_plan,
        &mut left_rng,
    );

    let mut right_rng = ChaCha20Rng::seed_from_u64(0x519D_1001);

    let right = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
        layout,
        auxiliary,
        &secret,
        right_exponent,
        distribution,
        2,
        &qp_plan,
        &mut right_rng,
    );

    let keygen_ms = keygen_start.elapsed().as_millis();

    let mut rotations = PreparedHybridRotationContext::new(&chain);

    rotations.insert(LEVEL, &left, &qp_plan);

    if right.exponent() != left.exponent() {
        rotations.insert(LEVEL, &right, &qp_plan);
    }

    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let execute_start = std::time::Instant::now();

    let output = execute_packed_tile_dif_stage_cp_hybrid_context(
        &ciphertext,
        &diagonals,
        &evaluator,
        &rotations,
        &embedding,
        &chain,
        &q_plan,
    );

    let execution_ms = execute_start.elapsed().as_millis();

    assert_eq!(output.level(), LEVEL + 1);

    let output_plan = RnsNttPlan::new(output.basis().moduli().to_vec(), N);

    let decrypted = decrypt_rns_raw_with_ntt(output.rlwe(), &secret, &output_plan);

    let modulus = composite_modulus_big(decrypted.basis());

    let coefficients: Vec<f64> = reconstruct_coefficients_big(&decrypted)
        .iter()
        .map(|residue| {
            centered_representative_big(residue, &modulus)
                .to_f64()
                .expect("CKKS coefficient conversion failed")
                / output.scale()
        })
        .collect();

    let actual = embedding.coefficients_to_slots(&coefficients);

    assert_eq!(actual.len(), expected.len());

    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for (actual, reference) in actual.iter().zip(expected.iter()) {
        let error = (*actual - *reference).norm();

        squared_error += error * error;
        squared_reference += reference.norm_sqr();
        max_abs = max_abs.max(error);
    }

    assert!(squared_reference > 0.0);

    let relative_l2 = (squared_error / squared_reference).sqrt();

    println!(
        "M1_9D_TILE_METRICS degree={} \
         level_in={} level_out={} rotation={} \
         digits={} keygen_ms={} execution_ms={} \
         rel_l2={:.12e} max_abs={:.12e}",
        N,
        LEVEL,
        output.level(),
        rotation,
        left.digit_count(),
        keygen_ms,
        execution_ms,
        relative_l2,
        max_abs,
    );

    assert!(relative_l2.is_finite());
    assert!(max_abs.is_finite());

    assert!(
        relative_l2 <= 5e-3,
        "Gaussian hybrid tile-block numerical validation failed"
    );

    println!("M1_9D_TILE_STATUS=PASS");
}
