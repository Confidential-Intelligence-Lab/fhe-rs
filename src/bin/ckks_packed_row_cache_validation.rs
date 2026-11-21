use ccmm_rs::application_support::ckks::encode_rns;
use ccmm_rs::ckks::{
    research_profile_65536_for_levels, CkksCanonicalEmbedding, CkksChainState, RnsCkksCiphertext,
    RnsCkksEvaluationKeys, RnsCkksEvaluator,
};
use ccmm_rs::eblas::fft::PackedRowPreparationContext;
use ccmm_rs::grafting::encrypt_rns_raw_with_distribution_ntt_rng;
use ccmm_rs::ring::RnsNttPlan;
use ccmm_rs::rlwe::ErrorDistribution;
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::time::Instant;

const START_LEVEL: usize = 2;
const EXPECTED_FINAL_LEVEL: usize = 12;
const CT_PER_MACRO: usize = 32;
const SIGMA: f64 = 3.19;

fn main() {
    let profile = research_profile_65536_for_levels(24, 2);
    let degree = profile.degree();
    let scale = profile.initial_scale();
    let chain = profile.modulus_chain();
    let embedding = CkksCanonicalEmbedding::new(degree);

    let basis = chain.level(START_LEVEL).clone();
    let plan = RnsNttPlan::new(basis.moduli().to_vec(), degree);

    assert_eq!(degree, 65_536);
    assert_eq!(basis.len(), 24);

    println!("ROW_CACHE_VALIDATION_BEGIN");
    println!("ROW_CACHE_START_LEVEL={START_LEVEL}");
    println!("ROW_CACHE_START_LIMBS={}", basis.len());
    println!("ROW_CACHE_SECURITY_BEARING=false");

    let mut secret_rng = ChaCha20Rng::seed_from_u64(0x524f_5743_4143_4845);
    let mut secret: Vec<i8> = (0..degree)
        .map(|_| secret_rng.gen_range(-1_i8..=1_i8))
        .collect();

    if secret.iter().all(|&coefficient| coefficient == 0) {
        secret[0] = 1;
    }

    let keygen_start = Instant::now();
    // Generate Gaussian hybrid rotation keys once for both cache runs.
    // Deterministic seeds are validation fixtures only.
    let auxiliary = ccmm_rs::ring::ModulusBasis::new(vec![
        ccmm_rs::ring::Modulus::new(0x7ffffffbee0001),
        ccmm_rs::ring::Modulus::new(0x7ffffffbe40001),
    ]);

    let schedule = ccmm_rs::eblas::fft::packed_1k_hybrid_rotation_schedule(
        START_LEVEL,
        ccmm_rs::eblas::fft::PackedFft2Axis::Rows,
    );

    assert_eq!(schedule.len(), 6);

    let mut key_material = Vec::new();

    for (level, rotation) in schedule {
        let mut left_rng = ChaCha20Rng::seed_from_u64(0x518B_1000 + level as u64 * 2);
        let mut right_rng = ChaCha20Rng::seed_from_u64(0x518B_1001 + level as u64 * 2);

        let (left, right, plan) = ccmm_rs::ckks::generate_hybrid_rotation_stage_keys_with_rng(
            &chain,
            degree,
            level,
            rotation,
            &secret,
            &auxiliary,
            55,
            ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
            2,
            &mut left_rng,
            &mut right_rng,
        );

        let right = right.expect("row rotation requires two keys");
        key_material.push((level, left, right, plan));
        println!("ROW_CACHE_KEYGEN_LEVEL={level} rotation={rotation}");
    }

    let mut rotations = ccmm_rs::ckks::PreparedHybridRotationContext::new(&chain);

    for (level, left, right, plan) in &key_material {
        rotations.insert(*level, left, plan);
        rotations.insert(*level, right, plan);
    }

    let key_count = rotations.key_count();
    assert_eq!(key_count, 12);

    println!("ROW_CACHE_KEY_COUNT={key_count}");
    println!("ROW_CACHE_KEYGEN_MS={}", keygen_start.elapsed().as_millis());

    let evaluation_keys = RnsCkksEvaluationKeys::new();
    let evaluator = RnsCkksEvaluator::new(&chain, &evaluation_keys);

    let make_macrotile = |macro_index: usize| -> Vec<RnsCkksCiphertext> {
        (0..CT_PER_MACRO)
            .map(|ct_index| {
                let slots = (0..embedding.slot_count())
                    .map(|slot| {
                        let value = (slot % 257 + ct_index * 7 + macro_index * 13) as f64 / 1024.0;
                        Complex64::new(value, 0.0)
                    })
                    .collect::<Vec<_>>();

                let plaintext = encode_rns(&slots, &embedding, &basis, scale);

                let mut rng = ChaCha20Rng::seed_from_u64(
                    0x524f_5745_4e43_0000 ^ ((macro_index as u64) << 16) ^ ct_index as u64,
                );

                let rlwe = encrypt_rns_raw_with_distribution_ntt_rng(
                    &plaintext,
                    2,
                    ErrorDistribution::DiscreteGaussian { sigma: SIGMA },
                    &secret,
                    &plan,
                    &mut rng,
                );

                RnsCkksCiphertext::new(
                    rlwe,
                    CkksChainState::new(&chain, START_LEVEL, scale),
                    &chain,
                )
            })
            .collect()
    };

    let mut cache = PackedRowPreparationContext::new();

    let first_input = make_macrotile(0);
    let first_start = Instant::now();
    let first_output = ccmm_rs::eblas::fft::execute_packed_1k_fft_rows_cp_hybrid_cached(
        first_input,
        &evaluator,
        &rotations,
        &embedding,
        &chain,
        (2, 4),
        &mut cache,
    );

    assert_eq!(first_output.len(), CT_PER_MACRO);
    assert!(first_output
        .iter()
        .all(|ct| ct.level() == EXPECTED_FINAL_LEVEL));

    println!(
        "ROW_CACHE_FIRST_MACRO_MS={}",
        first_start.elapsed().as_millis()
    );

    drop(first_output);

    let second_input = make_macrotile(1);
    let reference_input = second_input.clone();

    let cached_start = Instant::now();
    let cached = ccmm_rs::eblas::fft::execute_packed_1k_fft_rows_cp_hybrid_cached(
        second_input,
        &evaluator,
        &rotations,
        &embedding,
        &chain,
        (2, 4),
        &mut cache,
    );
    let cached_ms = cached_start.elapsed().as_millis();

    let mut fresh_cache = PackedRowPreparationContext::new();
    let uncached_start = Instant::now();
    let uncached = ccmm_rs::eblas::fft::execute_packed_1k_fft_rows_cp_hybrid_cached(
        reference_input,
        &evaluator,
        &rotations,
        &embedding,
        &chain,
        (2, 4),
        &mut fresh_cache,
    );
    let uncached_ms = uncached_start.elapsed().as_millis();

    assert_eq!(cached.len(), CT_PER_MACRO);
    assert_eq!(uncached.len(), CT_PER_MACRO);

    for (index, (lhs, rhs)) in cached.iter().zip(uncached.iter()).enumerate() {
        assert_eq!(
            lhs.rlwe(),
            rhs.rlwe(),
            "cached/uncached RLWE mismatch at ciphertext {index}"
        );
        assert_eq!(
            lhs.state(),
            rhs.state(),
            "cached/uncached state mismatch at ciphertext {index}"
        );
        assert_eq!(lhs.level(), EXPECTED_FINAL_LEVEL);
        assert_eq!(lhs.basis().len(), 14);
    }

    println!("ROW_CACHE_CIPHERTEXTS_VERIFIED={CT_PER_MACRO}");
    println!("ROW_CACHE_CACHED_MS={cached_ms}");
    println!("ROW_CACHE_UNCACHED_MS={uncached_ms}");

    drop(cached);
    drop(uncached);
    drop(cache);

    println!("ROW_CACHE_RELEASED=1");
    println!("ROW_CACHE_EXACT_RLWE_STATUS=PASS");
}
