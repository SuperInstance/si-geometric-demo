//! # si-geometric-demo
//!
//! Proof of concept: geometric algebra multivectors in WASM — proving
//! Cyberloop's Grassmannian tracking IS ga-core.
//!
//! ## The Key Insight
//!
//! Cyberloop v4.0 uses "Grassmannian subspace tracking" for agent step control.
//! The Grassmannian Gr(k,V) = space of k-dimensional subspaces of V.
//! In geometric algebra, a k-blade (wedge product of k vectors) represents exactly this.
//! So Cyberloop IS doing geometric algebra — just without calling it that.
//!
//! ## Modules
//!
//! - [`blade`] — k-blade operations (exterior algebra)
//! - [`conformal`] — conformal geometric algebra Cl(3,1)
//! - [`rotor`] — rotation and interpolation
//! - [`tracking`] — subspace tracking (the Cyberloop connection)
//! - [`agent_demo`] — agent trajectory as geometric object

pub mod blade;
pub mod conformal;
pub mod rotor;
pub mod tracking;
pub mod agent_demo;

use wasm_bindgen::prelude::*;

/// Run the agent demo and return JSON results.
#[wasm_bindgen]
pub fn demo(steps: usize) -> String {
    agent_demo::run_agent_demo(steps)
}

/// Compute the wedge product of two vectors and return components.
#[wasm_bindgen]
pub fn wedge_product(a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    let ba = blade::Blade::vector(a);
    let bb = blade::Blade::vector(b);
    let result = blade::wedge(&ba, &bb, 4);
    result.components
}

/// Compute subspace angle between two state vectors.
#[wasm_bindgen]
pub fn compute_subspace_angle(a: Vec<f64>, b: Vec<f64>) -> f64 {
    let ba = blade::Blade::vector(a);
    let bb = blade::Blade::vector(b);
    blade::subspace_angle(&ba, &bb)
}

/// Create a rotor from axis-angle.
#[wasm_bindgen]
pub fn make_rotor(ax: f64, ay: f64, az: f64, angle: f64) -> Vec<f64> {
    let r = rotor::Rotor::from_axis_angle(ax, ay, az, angle);
    r.mv.c.to_vec()
}

/// Apply rotor to a vector via sandwich product.
#[wasm_bindgen]
pub fn apply_rotor(rotor_components: Vec<f64>, vx: f64, vy: f64, vz: f64) -> Vec<f64> {
    let mut r = rotor::Rotor::new();
    for (i, &c) in rotor_components.iter().enumerate() {
        if i < 32 {
            r.mv.c[i] = c;
        }
    }
    let mut v = conformal::Multivector::new();
    v.c[1] = vx;
    v.c[2] = vy;
    v.c[4] = vz;
    let rotated = r.sandwich(&v);
    vec![rotated.c[1], rotated.c[2], rotated.c[4]]
}

/// Version info.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wedge_product_js() {
        let result = wedge_product(vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]);
        assert_eq!(result.len(), 6); // 4D bivector has 6 components
        assert!(result[0].abs() > 0.5); // e12 component
    }

    #[test]
    fn test_compute_subspace_angle() {
        let angle = compute_subspace_angle(
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        );
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_demo_runs() {
        let result = demo(3);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed["agents"].is_array());
    }

    #[test]
    fn test_make_and_apply_rotor() {
        let r = make_rotor(0.0, 0.0, 1.0, std::f64::consts::FRAC_PI_2);
        assert_eq!(r.len(), 32);
        let rotated = apply_rotor(r, 1.0, 0.0, 0.0);
        // e1 rotated 90° around z → mostly e2
        assert!(rotated[1].abs() > 0.5);
    }

    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }

    #[test]
    fn test_full_pipeline() {
        // Full pipeline: state → blade → rotor → prediction
        use crate::blade::{Blade, normalize, subspace_angle};
        use crate::rotor::Rotor;
        use crate::tracking::SubspaceTracker;

        // Track agent moving in a circle
        let mut tracker = SubspaceTracker::new(4);
        for i in 0..10 {
            let angle = (i as f64) * 0.3;
            let state = vec![angle.cos(), angle.sin(), 0.0, 0.0];
            tracker.update(state);
        }

        // Should have accumulated history
        assert!(tracker.history_len() > 1);

        // Predictions should continue the trajectory
        let preds = tracker.predict(3);
        assert_eq!(preds.len(), 3);

        // The rotor from initial to final should encode the total rotation
        if tracker.history_len() >= 2 {
            let first = &tracker.history[0];
            let last = &tracker.history[tracker.history_len() - 1];
            let rotor = Rotor::from_blades(first, last);
            // Should be a non-identity rotor
            assert!((rotor.scalar_part() - 1.0).abs() > 0.01);
        }
    }

    #[test]
    fn test_cyberloop_correspondence() {
        // Demonstrate the key correspondence:
        // Cyberloop's Grassmannian tracking = GA's blade tracking
        // Cyberloop's Riemannian step = GA's rotor
        use crate::blade::Blade;
        use crate::rotor::Rotor;
        use crate::tracking::SubspaceTracker;

        let mut tracker = SubspaceTracker::new(4);

        // Agent moves through states (simulating Cyberloop step control)
        let states = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.7, 0.3, 0.0, 0.0],
            vec![0.5, 0.5, 0.0, 0.0],
        ];

        for state in &states {
            let rotor = tracker.update(state.clone());
            // Each Cyberloop step produces a rotor = Grassmannian geodesic
        }

        assert_eq!(tracker.history_len(), states.len());
        assert_eq!(tracker.rotor_history().len(), states.len()); // one rotor per update call

        // The divergence between first and last state
        let first = &tracker.history[0];
        let last = &tracker.history[tracker.history_len() - 1];
        let angle = crate::blade::subspace_angle(first, last);

        // Should be about 45° (from e1 to normalized (0.5, 0.5))
        assert!(angle > 0.3 && angle < 1.2);
    }
}
