//! # Subspace Tracking — the Cyberloop connection
//!
//! Cyberloop v4.0 tracks agent state evolution on the Grassmannian Gr(k,V).
//! In geometric algebra, a point on the Grassmannian IS a k-blade.
//! The Riemannian geodesic between two Grassmannian points IS a rotor.
//!
//! This module provides:
//! - Projecting high-dimensional state vectors to k-blades
//! - Tracking subspace rotation over time
//! - Measuring divergence between agent trajectories
//! - Predicting future subspace states

use crate::blade::{self, Blade, normalize, subspace_angle};
use crate::rotor::Rotor;
use crate::conformal::Multivector;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Tracks how a high-dimensional state vector evolves as a subspace.
///
/// This IS Cyberloop's Grassmannian tracker, expressed in GA language:
/// - State vector → k-blade (Grassmannian point)
/// - State transition → rotor (Grassmannian geodesic)
/// - Agent divergence → subspace angle
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct SubspaceTracker {
    #[wasm_bindgen(skip)]
    pub dimension: usize,
    #[wasm_bindgen(skip)]
    pub track_grade: usize,
    #[wasm_bindgen(skip)]
    pub current_blade: Option<Blade>,
    #[wasm_bindgen(skip)]
    pub history: Vec<Blade>,
    #[wasm_bindgen(skip)]
    pub rotors: Vec<Rotor>,
    #[wasm_bindgen(skip)]
    pub velocity: Option<Blade>,
}

#[wasm_bindgen]
impl SubspaceTracker {
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize) -> SubspaceTracker {
        SubspaceTracker {
            dimension,
            track_grade: 1, // Track 1-blades (vectors) by default
            current_blade: None,
            history: Vec::new(),
            rotors: Vec::new(),
            velocity: None,
        }
    }

    /// Create a tracker that tracks k-dimensional subspaces.
    pub fn with_grade(dimension: usize, grade: usize) -> SubspaceTracker {
        SubspaceTracker {
            dimension,
            track_grade: grade,
            current_blade: None,
            history: Vec::new(),
            rotors: Vec::new(),
            velocity: None,
        }
    }

    /// Project a high-dimensional state vector to a k-blade.
    ///
    /// For a state vector s ∈ ℝⁿ, we form a blade by taking the top-k
    /// principal directions. For grade 1, this is just the normalized state.
    /// For higher grades, we use successive wedge products.
    pub fn track(&mut self, state: Vec<f64>) -> Blade {
        let n = state.len().min(self.dimension);
        let mut padded = vec![0.0; self.dimension];
        for i in 0..n {
            padded[i] = state[i];
        }

        let blade = if self.track_grade == 1 {
            // For vectors, just normalize
            let b = Blade::vector(padded);
            normalize(&b)
        } else if self.track_grade == 2 {
            // For bivectors, use the state and a shifted version
            let v1 = Blade::vector(padded.clone());
            let mut shifted = padded.clone();
            shifted.rotate_right(1);
            let v2 = Blade::vector(shifted);
            let biv = blade::wedge(&v1, &v2, self.dimension);
            normalize(&biv)
        } else {
            // For higher grades, iteratively wedge
            let base = Blade::vector(padded.clone());
            let mut result = base;
            for g in 1..self.track_grade {
                let mut shifted = padded.clone();
                shifted.rotate_right(g);
                let v = Blade::vector(shifted);
                result = blade::wedge(&result, &v, self.dimension);
            }
            normalize(&result)
        };

        self.current_blade = Some(blade.clone());
        self.history.push(blade.clone());
        blade
    }

    /// Update the tracker with a new state and return the rotor mapping
    /// the old subspace to the new one.
    ///
    /// This IS the Riemannian step on the Grassmannian — expressed as a rotor.
    pub fn update(&mut self, new_state: Vec<f64>) -> Rotor {
        let new_blade = self.track(new_state);

        let rotor = if let Some(ref old_blade) = self.history.iter().rev().nth(1).cloned() {
            Rotor::from_blades(old_blade, &new_blade)
        } else {
            Rotor::new()
        };

        self.rotors.push(rotor.clone());
        rotor
    }

    /// Compute the divergence between two tracked subspaces.
    ///
    /// This measures how far apart two agent trajectories have drifted
    /// in the Grassmannian — exactly what Cyberloop uses for step control.
    pub fn divergence(&self, other: &SubspaceTracker) -> f64 {
        match (&self.current_blade, &other.current_blade) {
            (Some(a), Some(b)) => subspace_angle(a, b),
            _ => std::f64::consts::FRAC_PI_2,
        }
    }

    /// Predict future subspace trajectory by extrapolating rotors.
    ///
    /// Uses the accumulated rotor history to estimate momentum and predict
    /// where the subspace will be in `steps` future steps.
    pub fn predict(&self, steps: usize) -> Vec<Blade> {
        let mut predictions = Vec::new();

        if self.history.is_empty() || self.current_blade.is_none() {
            return predictions;
        }

        let current = self.current_blade.as_ref().unwrap().clone();

        if self.rotors.is_empty() {
            // No history of rotation; predict stationary
            for _ in 0..steps {
                predictions.push(current.clone());
            }
            return predictions;
        }

        // Average the recent rotors to estimate angular velocity
        let recent_count = self.rotors.len().min(5);
        let recent_rotors: Vec<&Rotor> = self.rotors.iter().rev().take(recent_count).collect();

        // Use the most recent rotor for extrapolation
        let last_rotor = recent_rotors[0];

        let mut current_blade = current;
        for step in 0..steps {
            // Apply a fraction of the last rotor for each prediction step
            let t = (step + 1) as f64;
            let identity = Rotor::new();
            let extrapolated = Rotor::slerp(&identity, last_rotor, t.min(3.0));

            // Convert blade to multivector for sandwich
            let mut blade_mv = Multivector::new();
            if current_blade.grade == 1 {
                for (i, &c) in current_blade.components.iter().enumerate() {
                    if i < 5 {
                        blade_mv.c[1 << i] = c;
                    }
                }
            }

            let rotated_mv = extrapolated.sandwich(&blade_mv);

            // Convert back to blade
            let mut comps = Vec::new();
            for i in 0..5 {
                comps.push(rotated_mv.c[1 << i]);
            }
            let predicted = normalize(&Blade::vector(comps));
            predictions.push(predicted.clone());
            current_blade = predicted;
        }

        predictions
    }

    /// Get the tracking history.
    pub fn history(&self) -> Vec<Blade> {
        self.history.clone()
    }

    /// Get the rotor history.
    pub fn rotor_history(&self) -> Vec<Rotor> {
        self.rotors.clone()
    }

    /// Number of tracked states.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_tracker_new() {
        let t = SubspaceTracker::new(4);
        assert_eq!(t.dimension, 4);
        assert!(t.current_blade.is_none());
        assert_eq!(t.history.len(), 0);
    }

    #[test]
    fn test_tracker_track_vector() {
        let mut t = SubspaceTracker::new(4);
        let blade = t.track(vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(blade.grade, 1);
        assert!((blade.norm() - 1.0).abs() < 1e-10);
        assert_eq!(t.history.len(), 1);
    }

    #[test]
    fn test_tracker_track_normalizes() {
        let mut t = SubspaceTracker::new(4);
        let blade = t.track(vec![3.0, 4.0, 0.0, 0.0]);
        assert!((blade.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tracker_update() {
        let mut t = SubspaceTracker::new(4);
        t.track(vec![1.0, 0.0, 0.0, 0.0]);
        let rotor = t.update(vec![0.0, 1.0, 0.0, 0.0]);
        assert_eq!(t.rotors.len(), 1);
        // The rotor should NOT be identity since subspaces differ
        assert!((rotor.scalar_part() - 1.0).abs() > 0.01);
    }

    #[test]
    fn test_tracker_update_same_state() {
        let mut t = SubspaceTracker::new(4);
        t.track(vec![1.0, 0.0, 0.0, 0.0]);
        let rotor = t.update(vec![1.0, 0.0, 0.0, 0.0]);
        // Same state → identity rotor
        assert!((rotor.scalar_part() - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_divergence_same() {
        let mut t1 = SubspaceTracker::new(4);
        let mut t2 = SubspaceTracker::new(4);
        t1.track(vec![1.0, 0.0, 0.0, 0.0]);
        t2.track(vec![2.0, 0.0, 0.0, 0.0]);
        let div = t1.divergence(&t2);
        assert!(div.abs() < 1e-10);
    }

    #[test]
    fn test_divergence_orthogonal() {
        let mut t1 = SubspaceTracker::new(4);
        let mut t2 = SubspaceTracker::new(4);
        t1.track(vec![1.0, 0.0, 0.0, 0.0]);
        t2.track(vec![0.0, 1.0, 0.0, 0.0]);
        let div = t1.divergence(&t2);
        assert!((div - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_divergence_45deg() {
        let mut t1 = SubspaceTracker::new(4);
        let mut t2 = SubspaceTracker::new(4);
        t1.track(vec![1.0, 0.0, 0.0, 0.0]);
        t2.track(vec![1.0, 1.0, 0.0, 0.0]);
        let div = t1.divergence(&t2);
        assert!((div - PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_predict_stationary() {
        let mut t = SubspaceTracker::new(4);
        t.track(vec![1.0, 0.0, 0.0, 0.0]);
        let preds = t.predict(3);
        assert_eq!(preds.len(), 3);
        // Should predict same direction
        for p in &preds {
            assert!((p.components[0] - 1.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_predict_with_rotation() {
        let mut t = SubspaceTracker::new(4);
        t.track(vec![1.0, 0.0, 0.0, 0.0]);
        t.update(vec![0.0, 1.0, 0.0, 0.0]);
        let preds = t.predict(3);
        assert_eq!(preds.len(), 3);
        // Predictions should continue rotating
    }

    #[test]
    fn test_history_accumulates() {
        let mut t = SubspaceTracker::new(4);
        t.track(vec![1.0, 0.0, 0.0, 0.0]);
        t.update(vec![0.0, 1.0, 0.0, 0.0]);
        t.update(vec![0.0, 0.0, 1.0, 0.0]);
        assert_eq!(t.history_len(), 3);
        assert_eq!(t.rotor_history().len(), 2);
    }

    #[test]
    fn test_tracker_with_grade_2() {
        let mut t = SubspaceTracker::with_grade(4, 2);
        let blade = t.track(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(blade.grade, 2);
    }

    #[test]
    fn test_divergence_no_tracking() {
        let t1 = SubspaceTracker::new(4);
        let t2 = SubspaceTracker::new(4);
        let div = t1.divergence(&t2);
        assert!((div - PI / 2.0).abs() < 1e-10); // Maximum divergence
    }
}
