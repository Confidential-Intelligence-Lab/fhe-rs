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
