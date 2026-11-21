//! Level-aware lookup for prepared hybrid CKKS rotation keys.
//!
//! This context borrows existing key material and QP plans.
//! It does not generate keys, own secrets, or permit fallback
//! to an alternative key-switching implementation.

use std::collections::BTreeMap;

use crate::ring::{ModulusChain, RnsNttPlan};

use super::{
    rotate_left_hybrid_rns_ckks_with_prepared_ntt, rotate_right_hybrid_rns_ckks_with_prepared_ntt,
    rotation_exponent_left, rotation_exponent_right, PreparedHybridRnsGaloisKey, RnsCkksCiphertext,
};

type KeyEntry<'a> = (&'a PreparedHybridRnsGaloisKey, &'a RnsNttPlan);

/// Borrowed, level-indexed hybrid rotation resources.
///
/// Key identity is determined by the active CKKS level and
/// normalized Galois exponent. Each key retains its associated
/// QP NTT plan.
pub struct PreparedHybridRotationContext<'a> {
    chain: &'a ModulusChain,
    keys: BTreeMap<(usize, usize), KeyEntry<'a>>,
}

impl<'a> PreparedHybridRotationContext<'a> {
    pub fn new(chain: &'a ModulusChain) -> Self {
        Self {
            chain,
            keys: BTreeMap::new(),
        }
    }

    /// Registers an existing prepared key without copying it.
    pub fn insert(
        &mut self,
        level: usize,
        key: &'a PreparedHybridRnsGaloisKey,
        qp_plan: &'a RnsNttPlan,
    ) {
        let basis = self.chain.level(level);
        let degree = qp_plan.degree();

        assert!(degree > 0, "invalid hybrid rotation degree");

        assert_eq!(key.degree(), degree, "hybrid rotation key degree mismatch");

        assert!(
            key.exponent() < 2 * degree && key.exponent() % 2 == 1,
            "invalid hybrid rotation exponent"
        );

        let mut expected = basis.moduli().to_vec();
        expected.extend_from_slice(key.auxiliary_basis().moduli());

        assert_eq!(
            qp_plan.moduli(),
            expected.as_slice(),
            "hybrid rotation QP basis mismatch"
        );

        assert!(
            self.keys
                .insert((level, key.exponent()), (key, qp_plan))
                .is_none(),
            "duplicate hybrid rotation key"
        );
    }

    pub fn key_for_level(&self, level: usize, exponent: usize) -> KeyEntry<'a> {
        *self
            .keys
            .get(&(level, exponent))
            .expect("missing prepared hybrid rotation key")
    }

    pub fn rotate_left(&self, ciphertext: &RnsCkksCiphertext, steps: usize) -> RnsCkksCiphertext {
        ciphertext.assert_matches_chain(self.chain);

        let degree = ciphertext.rlwe().degree();
        let exponent = rotation_exponent_left(degree, steps);
        let (key, plan) = self.key_for_level(ciphertext.level(), exponent);

        rotate_left_hybrid_rns_ckks_with_prepared_ntt(ciphertext, steps, key, self.chain, plan)
    }

    pub fn rotate_right(&self, ciphertext: &RnsCkksCiphertext, steps: usize) -> RnsCkksCiphertext {
        ciphertext.assert_matches_chain(self.chain);

        let degree = ciphertext.rlwe().degree();
        let exponent = rotation_exponent_right(degree, steps);
        let (key, plan) = self.key_for_level(ciphertext.level(), exponent);

        rotate_right_hybrid_rns_ckks_with_prepared_ntt(ciphertext, steps, key, self.chain, plan)
    }

    pub fn key_count(&self) -> usize {
        self.keys.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "missing prepared hybrid rotation key")]
    fn missing_key_is_rejected() {
        let profile = crate::ckks::correctness_profile_8();
        let chain = profile.modulus_chain();
        let context = PreparedHybridRotationContext::new(&chain);

        let _ = context.key_for_level(0, 3);
    }

    #[test]
    fn empty_context_owns_no_keys() {
        let profile = crate::ckks::correctness_profile_8();
        let chain = profile.modulus_chain();
        let context = PreparedHybridRotationContext::new(&chain);

        assert_eq!(context.key_count(), 0);
    }

    fn gaussian_fixture(chain: &ModulusChain) -> (PreparedHybridRnsGaloisKey, RnsNttPlan) {
        use crate::grafting::BoundedGadgetLayout;
        use crate::ring::{Modulus, ModulusBasis};
        use crate::rlwe::ErrorDistribution;
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        let q = chain.level(0).clone();
        let p = ModulusBasis::new(vec![Modulus::new(114_689), Modulus::new(786_433)]);

        let mut moduli = q.moduli().to_vec();
        moduli.extend_from_slice(p.moduli());

        let plan = RnsNttPlan::new(moduli, 8);
        let secret = [1_i8, 0, -1, 1, 0, -1, 1, 0];

        let mut rng = ChaCha20Rng::seed_from_u64(0xA012_B001);

        let key = PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
            BoundedGadgetLayout::new(q, 8),
            p,
            &secret,
            3,
            ErrorDistribution::DiscreteGaussian { sigma: 3.19 },
            2,
            &plan,
            &mut rng,
        );

        (key, plan)
    }

    #[test]
    fn gaussian_key_registration_and_lookup() {
        let profile = crate::ckks::correctness_profile_8();
        let chain = profile.modulus_chain();
        let (key, plan) = gaussian_fixture(&chain);

        let mut context = PreparedHybridRotationContext::new(&chain);
        context.insert(0, &key, &plan);

        assert_eq!(context.key_count(), 1);

        let (found_key, found_plan) = context.key_for_level(0, 3);

        assert!(std::ptr::eq(found_key, &key));
        assert!(std::ptr::eq(found_plan, &plan));
        assert_eq!(found_key.degree(), 8);
        assert_eq!(found_key.exponent(), 3);
    }

    #[test]
    #[should_panic(expected = "duplicate hybrid rotation key")]
    fn duplicate_registration_is_rejected() {
        let profile = crate::ckks::correctness_profile_8();
        let chain = profile.modulus_chain();
        let (key, plan) = gaussian_fixture(&chain);

        let mut context = PreparedHybridRotationContext::new(&chain);
        context.insert(0, &key, &plan);
        context.insert(0, &key, &plan);
    }

    #[test]
    #[should_panic(expected = "hybrid rotation QP basis mismatch")]
    fn incompatible_qp_basis_is_rejected() {
        let profile = crate::ckks::correctness_profile_8();
        let chain = profile.modulus_chain();
        let (key, _) = gaussian_fixture(&chain);

        let wrong_plan = RnsNttPlan::new(chain.level(0).moduli().to_vec(), 8);

        let mut context = PreparedHybridRotationContext::new(&chain);
        context.insert(0, &key, &wrong_plan);
    }

    #[test]
    fn gaussian_encrypted_rotations_match_direct_hybrid() {
        use crate::ckks::CkksChainState;
        use crate::grafting::{encrypt_rns_raw_with_distribution_ntt_rng, BoundedGadgetLayout};
        use crate::ring::{Modulus, ModulusBasis, Polynomial, RnsPolynomial};
        use crate::rlwe::ErrorDistribution;
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;

        const N: usize = 8;
        const STEPS: usize = 1;

        let profile = crate::ckks::correctness_profile_8();
        let chain = profile.modulus_chain();
        let q = chain.level(0).clone();

        let p = ModulusBasis::new(vec![Modulus::new(114_689), Modulus::new(786_433)]);

        let mut qp_moduli = q.moduli().to_vec();
        qp_moduli.extend_from_slice(p.moduli());

        let q_plan = RnsNttPlan::new(q.moduli().to_vec(), N);
        let qp_plan = RnsNttPlan::new(qp_moduli, N);

        let secret = [1_i8, 0, -1, 1, 0, -1, 1, 0];
        let distribution = ErrorDistribution::DiscreteGaussian { sigma: 3.19 };

        let message = RnsPolynomial::from_residues(
            q.moduli()
                .iter()
                .map(|&modulus| Polynomial::new(modulus, (0..N).map(|i| (i as u64) + 1).collect()))
                .collect(),
        );

        let mut encryption_rng = ChaCha20Rng::seed_from_u64(0xA012_C001);

        let encrypted = encrypt_rns_raw_with_distribution_ntt_rng(
            &message,
            2,
            distribution,
            &secret,
            &q_plan,
            &mut encryption_rng,
        );

        let state = CkksChainState::top(&chain, 65_537.0);
        let ciphertext = RnsCkksCiphertext::new(encrypted, state.clone(), &chain);

        let left_exponent = rotation_exponent_left(N, STEPS);
        let right_exponent = rotation_exponent_right(N, STEPS);

        let make_key = |exponent: usize, seed: u64| {
            let mut rng = ChaCha20Rng::seed_from_u64(seed);

            PreparedHybridRnsGaloisKey::generate_with_distribution_ntt_rng(
                BoundedGadgetLayout::new(q.clone(), 8),
                p.clone(),
                &secret,
                exponent,
                distribution,
                2,
                &qp_plan,
                &mut rng,
            )
        };

        let left_key = make_key(left_exponent, 0xA012_C002);
        let right_key = make_key(right_exponent, 0xA012_C003);

        let mut context = PreparedHybridRotationContext::new(&chain);
        context.insert(0, &left_key, &qp_plan);

        if right_exponent != left_exponent {
            context.insert(0, &right_key, &qp_plan);
        }

        let left = context.rotate_left(&ciphertext, STEPS);
        let right = context.rotate_right(&ciphertext, STEPS);

        let direct_left = rotate_left_hybrid_rns_ckks_with_prepared_ntt(
            &ciphertext,
            STEPS,
            &left_key,
            &chain,
            &qp_plan,
        );

        let direct_right = rotate_right_hybrid_rns_ckks_with_prepared_ntt(
            &ciphertext,
            STEPS,
            if right_exponent == left_exponent {
                &left_key
            } else {
                &right_key
            },
            &chain,
            &qp_plan,
        );

        assert_eq!(left, direct_left);
        assert_eq!(right, direct_right);

        assert_eq!(left.state(), &state);
        assert_eq!(right.state(), &state);

        assert_eq!(left.level(), ciphertext.level());
        assert_eq!(right.level(), ciphertext.level());

        assert_eq!(left.scale(), ciphertext.scale());
        assert_eq!(right.scale(), ciphertext.scale());

        assert_eq!(left.basis(), ciphertext.basis());
        assert_eq!(right.basis(), ciphertext.basis());

        println!("M1_2C_GAUSSIAN_CONTEXT_ROTATION_STATUS=PASS");
    }
}
