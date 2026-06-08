//! # Rotor — rotation and interpolation in geometric algebra
//!
//! A rotor is the GA generalization of a quaternion. It's an even-grade
//! multivector R = a + B where a is a scalar and B is a bivector.
//! Rotation: v' = R v R̃ (the "sandwich" product)
//!
//! **The Cyberloop connection:** When an agent's state subspace rotates from
//! one step to the next, that rotation IS a rotor. The Riemannian geodesic
//! step on the Grassmannian IS a rotor interpolation (slerp).

use crate::blade::{self, Blade, normalize, subspace_angle};
use crate::conformal::{geometric_product, Multivector};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// A rotor in Cl(3,1): an even-grade multivector (scalar + bivector + pseudoscalar).
///
/// In practice, for rotations in 3D Euclidean subspace:
/// R = cos(θ/2) + sin(θ/2) * B
/// where B is the unit bivector for the rotation plane.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct Rotor {
    #[wasm_bindgen(skip)]
    pub mv: Multivector,
}

#[wasm_bindgen]
impl Rotor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Rotor {
        Rotor { mv: Multivector::from_scalar(1.0) }
    }

    pub fn from_scalar_bivector(scalar: f64, bivector_components: Vec<f64>) -> Rotor {
        let mut mv = Multivector::new();
        mv.c[0] = scalar;
        // Bivector basis: e1e2=3, e1e3=5, e1ep=9, e1em=17, e2e3=6, e2ep=10, e2em=18, e3ep=12, e3em=20, epem=24
        let bivector_indices = [3, 5, 9, 17, 6, 10, 18, 12, 20, 24];
        for (i, &idx) in bivector_indices.iter().enumerate() {
            if i < bivector_components.len() {
                mv.c[idx] = bivector_components[i];
            }
        }
        Rotor { mv }
    }

    /// Create a rotor from an axis (3D) and angle.
    pub fn from_axis_angle(axis_x: f64, axis_y: f64, axis_z: f64, angle: f64) -> Rotor {
        let norm = (axis_x * axis_x + axis_y * axis_y + axis_z * axis_z).sqrt();
        if norm < 1e-15 {
            return Rotor { mv: Multivector::from_scalar(1.0) };
        }
        let (ax, ay, az) = (axis_x / norm, axis_y / norm, axis_z / norm);
        let half = angle / 2.0;
        let cos_h = half.cos();
        let sin_h = half.sin();

        // The rotation plane bivector for axis (ax, ay, az):
        // The bivector dual to the axis: B = ax*e2e3 + ay*e3e1 + az*e1e2
        // In our basis: e2e3=6, e1e3=5, e1e2=3
        // Note: e3e1 = -e1e3
        let mut mv = Multivector::new();
        mv.c[0] = cos_h;
        // Bivector components: ax*(e2∧e3) - ay*(e1∧e3) + az*(e1∧e2)
        mv.c[6] = sin_h * ax;   // e2e3
        mv.c[5] = -sin_h * ay;  // e3e1 = -e1e3
        mv.c[3] = sin_h * az;   // e1e2
        Rotor { mv }
    }

    /// Spherical linear interpolation between two rotors.
    ///
    /// This is the GA equivalent of quaternion slerp — smooth rotation from
    /// one orientation to another. This is exactly what Cyberloop's
    /// Riemannian step does on the Grassmannian.
    pub fn slerp(a: &Rotor, b: &Rotor, t: f64) -> Rotor {
        // Compute the relative rotor: R = b * a⁻¹
        let a_inv = a.inverse();
        let relative = geometric_product(&b.mv, &a_inv.mv);

        // Extract scalar and bivector parts for interpolation
        let cos_half = relative.c[0]; // scalar part

        // The angle of the relative rotor
        let cos_half_clamped = cos_half.clamp(-1.0, 1.0);
        let half_angle = cos_half_clamped.acos();

        let mut interp = Multivector::new();
        if half_angle.abs() < 1e-10 {
            // Rotors are (nearly) identical
            interp = a.mv.clone();
        } else {
            let sin_half = half_angle.sin();
            let t_half = t * half_angle;
            let cos_t = t_half.cos();
            let sin_t = t_half.sin();

            // R(t) = a * R^(t) where R^t = cos(t*θ/2) + sin(t*θ/2) * B_normalized
            let mut scaled_relative = relative.clone();
            for i in 0..32 {
                if i == 0 {
                    scaled_relative.c[i] = cos_t;
                } else {
                    scaled_relative.c[i] = if sin_half.abs() > 1e-15 {
                        scaled_relative.c[i] / sin_half * sin_t
                    } else {
                        0.0
                    };
                }
            }
            interp = geometric_product(&scaled_relative, &a.mv);
        }

        // Normalize
        let norm = interp.norm();
        if norm > 1e-15 {
            interp = interp.scale(1.0 / norm);
        }

        Rotor { mv: interp }
    }

    /// Apply rotation via sandwich product: v' = R v R̃
    pub fn sandwich(&self, v: &Multivector) -> Multivector {
        let rev = self.mv.reverse();
        let rv = geometric_product(&self.mv, v);
        geometric_product(&rv, &rev)
    }

    /// Inverse rotor.
    pub fn inverse(&self) -> Rotor {
        // For a rotor: R⁻¹ = R̃ / (R·R̃)
        let rev = self.mv.reverse();
        let rr = geometric_product(&self.mv, &rev);
        let norm_sq = rr.scalar();
        if norm_sq.abs() < 1e-15 {
            return Rotor { mv: self.mv.reverse() };
        }
        Rotor { mv: rev.scale(1.0 / norm_sq) }
    }

    /// Create a rotor that maps one subspace (blade) to another.
    ///
    /// This is THE key operation: given the agent's current state subspace
    /// and a target subspace, compute the rotor that rotates one to the other.
    /// This IS what Cyberloop calls a "Riemannian step on the Grassmannian."
    pub fn from_blades(from: &Blade, to: &Blade) -> Rotor {
        let from_n = normalize(from);
        let to_n = normalize(to);

        // R = (1 + to * from) / |1 + to * from|
        // For vectors: R = 1 + b∧a (normalized)
        let angle = subspace_angle(&from_n, &to_n);

        if angle.abs() < 1e-10 {
            return Rotor { mv: Multivector::from_scalar(1.0) };
        }

        // For vectors: R = cos(θ/2) + sin(θ/2) * (a∧b)/|a∧b|
        let half = angle / 2.0;
        let cos_h = half.cos();
        let sin_h = half.sin();

        // The rotation plane bivector is from_n ∧ to_n
        let biv = blade::wedge(&from_n, &to_n, 4);
        let biv_norm = biv.norm();

        let mut mv = Multivector::new();
        mv.c[0] = cos_h;

        if biv_norm > 1e-15 && from.grade == 1 {
            // For vector blades, map bivector components to multivector basis
            // Bivector in 4D: e12, e13, e14, e23, e24, e34
            // Multivector indices: e12=3, e13=5, e14=9, e23=6, e24=10, e34=12
            let biv_to_mv = [3, 5, 9, 6, 10, 12];
            for (i, &idx) in biv_to_mv.iter().enumerate() {
                if i < biv.components.len() {
                    mv.c[idx] = sin_h * biv.components[i] / biv_norm;
                }
            }
        }

        // Normalize
        let norm = mv.norm();
        if norm > 1e-15 {
            mv = mv.scale(1.0 / norm);
        }

        Rotor { mv }
    }

    pub fn to_multivector(&self) -> Multivector {
        self.mv.clone()
    }

    pub fn scalar_part(&self) -> f64 {
        self.mv.scalar()
    }
}

impl Default for Rotor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_rotor_identity() {
        let r = Rotor::new();
        assert!((r.scalar_part() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotor_from_axis_angle_zero() {
        let r = Rotor::from_axis_angle(0.0, 0.0, 1.0, 0.0);
        assert!((r.scalar_part() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotor_from_axis_angle_90() {
        let r = Rotor::from_axis_angle(0.0, 0.0, 1.0, PI / 2.0);
        // cos(45°) ≈ 0.707
        assert!((r.scalar_part() - (PI / 4.0).cos()).abs() < 1e-10);
    }

    #[test]
    fn test_rotor_sandwich_identity() {
        let r = Rotor::new();
        let mut v = Multivector::new();
        v.c[1] = 1.0; // e1
        let rotated = r.sandwich(&v);
        // Identity rotor should not change the vector
        assert!((rotated.c[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotor_sandwich_90deg_z() {
        let r = Rotor::from_axis_angle(0.0, 0.0, 1.0, PI / 2.0);
        let mut v = Multivector::new();
        v.c[1] = 1.0; // e1
        let rotated = r.sandwich(&v);
        // e1 rotated 90° around z → should give e2
        assert!(rotated.c[2].abs() > 0.9); // e2 component
        assert!(rotated.c[1].abs() < 0.1); // e1 component should be small
    }

    #[test]
    fn test_rotor_inverse() {
        let r = Rotor::from_axis_angle(1.0, 2.0, 3.0, 1.5);
        let inv = r.inverse();
        // R * R⁻¹ should be ~ identity
        let product = geometric_product(&r.mv, &inv.mv);
        assert!((product.scalar() - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_rotor_double_rotation() {
        let r = Rotor::from_axis_angle(0.0, 0.0, 1.0, PI / 4.0);
        let mut v = Multivector::new();
        v.c[1] = 1.0;
        let v1 = r.sandwich(&v);
        let v2 = r.sandwich(&v1);
        // Two 45° rotations = 90° rotation
        assert!(v2.c[2].abs() > 0.9);
    }

    #[test]
    fn test_slerp_endpoints() {
        let a = Rotor::new(); // identity
        let b = Rotor::from_axis_angle(0.0, 0.0, 1.0, PI / 2.0);

        let at_0 = Rotor::slerp(&a, &b, 0.0);
        assert!((at_0.scalar_part() - 1.0).abs() < 1e-8);

        let at_1 = Rotor::slerp(&a, &b, 1.0);
        assert!((at_1.scalar_part() - b.scalar_part()).abs() < 1e-8);
    }

    #[test]
    fn test_slerp_midpoint() {
        let a = Rotor::new();
        let b = Rotor::from_axis_angle(0.0, 0.0, 1.0, PI);

        let mid = Rotor::slerp(&a, &b, 0.5);
        // At midpoint, scalar should be cos(PI/4) ≈ 0.707
        assert!((mid.scalar_part() - (PI / 4.0).cos()).abs() < 0.05);
    }

    #[test]
    fn test_from_blades_parallel() {
        let a = Blade::vector(vec![1.0, 0.0, 0.0, 0.0]);
        let b = Blade::vector(vec![2.0, 0.0, 0.0, 0.0]);
        let r = Rotor::from_blades(&a, &b);
        // Should be identity
        assert!((r.scalar_part() - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_from_blades_orthogonal() {
        let a = Blade::vector(vec![1.0, 0.0, 0.0, 0.0]);
        let b = Blade::vector(vec![0.0, 1.0, 0.0, 0.0]);
        let r = Rotor::from_blades(&a, &b);
        // Should be a 90° rotation: scalar = cos(45°)
        assert!((r.scalar_part() - (PI / 4.0).cos()).abs() < 1e-8);
    }

    #[test]
    fn test_from_blades_maps_correctly() {
        let a = Blade::vector(vec![1.0, 0.0, 0.0, 0.0]);
        let b = Blade::vector(vec![0.0, 1.0, 0.0, 0.0]);
        let r = Rotor::from_blades(&a, &b);

        let mut va = Multivector::new();
        va.c[1] = 1.0; // e1
        let rotated = r.sandwich(&va);
        // Should map e1 toward e2
        assert!(rotated.c[2].abs() > 0.9);
    }

    #[test]
    fn test_rotor_preserves_norm() {
        // Pure 3D rotation should preserve Euclidean norm
        let r = Rotor::from_axis_angle(0.0, 0.0, 1.0, 1.5);
        let mut v = Multivector::new();
        v.c[1] = 3.0; // e1
        v.c[2] = 4.0; // e2
        // Only use e1,e2 components — rotation around z keeps them in the e1e2 plane
        let rotated = r.sandwich(&v);
        let norm_sq = rotated.c[1] * rotated.c[1] + rotated.c[2] * rotated.c[2];
        assert!((norm_sq - 25.0).abs() < 1.0, "norm_sq = {norm_sq}");
    }

    #[test]
    fn test_rotor_360_returns() {
        let r = Rotor::from_axis_angle(0.0, 0.0, 1.0, 2.0 * PI);
        let mut v = Multivector::new();
        v.c[1] = 1.0;
        let rotated = r.sandwich(&v);
        assert!((rotated.c[1] - 1.0).abs() < 1e-8);
    }
}
