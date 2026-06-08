//! # Conformal Geometric Algebra — Cl(3,1)
//!
//! Conformal GA extends Euclidean 3D space with two extra dimensions:
//! - e₊ (positive signature) — origin
//! - e₋ (negative signature) — infinity
//!
//! Point: P = x + ½x²e₋ + e₊
//! This gives us spheres, circles, lines, planes as blades — and
//! meet/join as intersection/union operations.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// A multivector in Cl(3,1) with all 16 components.
///
/// Basis order (grades):
///   Grade 0: [1]
///   Grade 1: [e1, e2, e3, ep, em]
///   Grade 2: [e1e2, e1e3, e1ep, e1em, e2e3, e2ep, e2em, e3ep, e3em, epem]
///   Grade 3: [e1e2e3, e1e2ep, e1e2em, e1e3ep, e1e3em, e1epem, e2e3ep, e2e3em, e2epem, e3epem]
///   Grade 4: [e1e2e3epem]
///
/// For simplicity we store 32 components indexed by bitmask (5 bits = 32).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct Multivector {
    #[wasm_bindgen(skip)]
    pub c: [f64; 32], // Indexed by basis bitmask
}

#[wasm_bindgen]
impl Multivector {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Multivector {
        Multivector { c: [0.0; 32] }
    }

    pub fn from_scalar(s: f64) -> Multivector {
        let mut mv = Multivector::new();
        mv.c[0] = s;
        mv
    }

    pub fn from_component(index: usize, value: f64) -> Multivector {
        let mut mv = Multivector::new();
        if index < 32 {
            mv.c[index] = value;
        }
        mv
    }

    pub fn get(&self, index: usize) -> f64 {
        if index < 32 { self.c[index] } else { 0.0 }
    }

    pub fn set(&mut self, index: usize, value: f64) {
        if index < 32 {
            self.c[index] = value;
        }
    }
}

impl Default for Multivector {
    fn default() -> Self {
        Self::new()
    }
}

impl Multivector {
    /// Grade extraction — returns a multivector with only components of the given grade.
    pub fn grade_part(&self, g: usize) -> Multivector {
        let mut result = Multivector::new();
        for i in 0..32 {
            if grade_of_index(i) == g {
                result.c[i] = self.c[i];
            }
        }
        result
    }

    /// Scalar part.
    pub fn scalar(&self) -> f64 {
        self.c[0]
    }

    /// Norm (simplified: sum of squared components).
    pub fn norm(&self) -> f64 {
        self.c.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Reverse: reverse the order of basis vectors in each component.
    /// For grade k, the reverse introduces a sign of (-1)^(k(k-1)/2).
    pub fn reverse(&self) -> Multivector {
        let mut result = self.clone();
        for i in 0..32 {
            let g = grade_of_index(i);
            let sign = if g.checked_sub(1).map(|g1| (g * g1 / 2) % 2 == 0).unwrap_or(true) { 1.0 } else { -1.0 };
            result.c[i] = sign * self.c[i];
        }
        result
    }

    /// Add two multivectors.
    pub fn add(&self, other: &Multivector) -> Multivector {
        let mut result = Multivector::new();
        for i in 0..32 {
            result.c[i] = self.c[i] + other.c[i];
        }
        result
    }

    /// Subtract two multivectors.
    pub fn sub(&self, other: &Multivector) -> Multivector {
        let mut result = Multivector::new();
        for i in 0..32 {
            result.c[i] = self.c[i] - other.c[i];
        }
        result
    }

    /// Scalar multiplication.
    pub fn scale(&self, s: f64) -> Multivector {
        let mut result = Multivector::new();
        for i in 0..32 {
            result.c[i] = self.c[i] * s;
        }
        result
    }
}

/// Geometric product of two multivectors in Cl(3,1).
/// Metric signature: (+,+,+,-) for (e1,e2,e3,em), ep is null with em.
pub fn geometric_product(a: &Multivector, b: &Multivector) -> Multivector {
    let mut result = Multivector::new();

    // Metric: e1²=+1, e2²=+1, e3²=+1, ep²=0, em²=0, ep·em = -1/2 (nope, for Cl(3,1))
    // Actually for Cl(3,1): e1²=e2²=e3²=+1, eo²=0, e∞²=0, eo·e∞=-1
    // Let's use basis: 0=e1, 1=e2, 2=e3, 3=eo, 4=e∞
    // Signature: +1,+1,+1,0,0 but eo∧e∞ has special metric
    // Simpler: use 5D with metric diag(1,1,1,1,-1) mapped as e1..e4,e5
    // basis 0=e1, 1=e2, 2=e3, 3=ep, 4=em with ep²=+1, em²=-1

    let metric = [1.0_f64, 1.0, 1.0, 1.0, -1.0];

    for i in 0..32 {
        if a.c[i].abs() < 1e-30 { continue; }
        for j in 0..32 {
            if b.c[j].abs() < 1e-30 { continue; }

            let (prod_idx, sign) = basis_product(i, j, &metric);
            result.c[prod_idx] += sign * a.c[i] * b.c[j];
        }
    }
    result
}

/// Compute the basis product index and sign.
///
/// For canonical basis blades I, J in Cl(p,q) with metric, the geometric product
/// e_I * e_J produces basis e_{I XOR J minus contracted} with sign:
///   sign = product of metric[s] for s in intersection
///        * (-1)^(sum over s in intersection: count of non-matching I bits above s)
///        * (-1)^(sum over j in J\S: count of I bits above j)
fn basis_product(a: usize, b: usize, metric: &[f64; 5]) -> (usize, f64) {
    let mut sign = 1.0_f64;
    let intersection = a & b;
    let mut result_idx = a ^ b; // XOR: remove contracted bits

    // Contract: multiply by metric for each shared bit
    for bit in 0..5 {
        if intersection & (1 << bit) != 0 {
            sign *= metric[bit];
        }
    }

    // Sign from moving contracted bits in a to meet contracted bits
    // For each s in intersection: count bits in a\intersection that are above s
    let a_only = a & !intersection;
    for s in 0..5 {
        if intersection & (1 << s) != 0 {
            for i in (s + 1)..5 {
                if a_only & (1 << i) != 0 {
                    sign *= -1.0;
                }
            }
        }
    }

    // Sign from wedge part: for each j in b\intersection, count a bits above j
    let b_only = b & !intersection;
    for j in 0..5 {
        if b_only & (1 << j) != 0 {
            for i in (j + 1)..5 {
                if a & (1 << i) != 0 {
                    sign *= -1.0;
                }
            }
        }
    }

    (result_idx, sign)
}

fn grade_of_index(i: usize) -> usize {
    i.count_ones() as usize
}

/// Conformal point: P = x·e1 + y·e2 + z·e3 + ½(x²+y²+z²)·em + ep
///
/// Basis: 0=e1, 1=e2, 2=e3, 3=ep, 4=em
/// Index: e1=1, e2=2, e3=4, ep=8, em=16
pub fn point(x: f64, y: f64, z: f64) -> Multivector {
    let mut mv = Multivector::new();
    mv.c[1 << 0] = x;           // e1
    mv.c[1 << 1] = y;           // e2
    mv.c[1 << 2] = z;           // e3
    mv.c[1 << 3] = 1.0;         // ep (origin)
    mv.c[1 << 4] = 0.5 * (x * x + y * y + z * z); // em (infinity weight)
    mv
}

/// Conformal line through two points: L = P1 ∧ P2 ∧ e∞
pub fn line_from_points(p1: &Multivector, p2: &Multivector) -> Multivector {
    let einfinity = {
        let mut mv = Multivector::new();
        mv.c[1 << 4] = 1.0; // em
        mv
    };
    let p1wedge_p2 = geometric_product(
        &geometric_product(p1, p2).grade_part(2),
        &einfinity
    ).grade_part(3);
    // Actually: outer product = (a∧b) grade part of geometric product
    // L = P1 ∧ P2 ∧ e∞
    let op12 = outer_product(p1, p2);
    outer_product(&op12, &einfinity)
}

/// Conformal plane through three points: Π = P1 ∧ P2 ∧ P3 ∧ e∞
pub fn plane_from_points(p1: &Multivector, p2: &Multivector, p3: &Multivector) -> Multivector {
    let einfinity = {
        let mut mv = Multivector::new();
        mv.c[1 << 4] = 1.0;
        mv
    };
    let op12 = outer_product(p1, p2);
    let op123 = outer_product(&op12, p3);
    outer_product(&op123, &einfinity)
}

/// Sphere from center and radius: S = C - ½r²e∞
pub fn sphere_from_center_radius(cx: f64, cy: f64, cz: f64, r: f64) -> Multivector {
    let mut c = point(cx, cy, cz);
    // Subtract ½r² from the infinity component
    c.c[1 << 4] -= 0.5 * r * r;
    c
}

/// Outer (wedge) product of two multivectors.
pub fn outer_product(a: &Multivector, b: &Multivector) -> Multivector {
    let gp = geometric_product(a, b);
    let target_grade = grade_of_multivector(a) + grade_of_multivector(b);
    if target_grade > 5 {
        return Multivector::new();
    }
    gp.grade_part(target_grade.min(5))
}

fn grade_of_multivector(mv: &Multivector) -> usize {
    let mut max_grade = 0;
    for i in 0..32 {
        if mv.c[i].abs() > 1e-15 {
            max_grade = max_grade.max(grade_of_index(i));
        }
    }
    max_grade
}

/// Meet (intersection): a ∨ b = (a* ∧ b*)*
/// In conformal GA, the meet gives the intersection of geometric objects.
pub fn meet(a: &Multivector, b: &Multivector) -> Multivector {
    let da = dual(a);
    let db = dual(b);
    let joined = outer_product(&da, &db);
    undual(&joined)
}

/// Join (union): a ∧ b (outer product when a and b are in general position).
pub fn join(a: &Multivector, b: &Multivector) -> Multivector {
    outer_product(a, b)
}

/// Dual: A* = A · I⁻¹, where I is the pseudoscalar.
pub fn dual(mv: &Multivector) -> Multivector {
    let pseudoscalar = pseudoscalar();
    let inv_pseudo = inverse_pseudoscalar();
    // A* = A · I⁻¹ (left contraction)
    // Simplified: geometric product and take grade (n-k)
    let gp = geometric_product(mv, &inv_pseudo);
    let n: usize = 5;
    // We want grade (n - grade_of(A))
    let a_grade = grade_of_multivector(mv);
    let target = n.saturating_sub(a_grade);
    gp.grade_part(target)
}

/// Undual: inverse of dual.
pub fn undual(mv: &Multivector) -> Multivector {
    let inv_pseudo = inverse_pseudoscalar();
    let pseudo = pseudoscalar();
    // A = A** · I (approximately)
    // Actually: undual(A) = dual(A) composed with sign
    // Simple approach: dual(dual(A)) = ±A
    let d1 = dual(mv);
    // Apply sign correction: dual² = (-1)^(k(n-k))
    let mut result = d1;
    let g = grade_of_multivector(mv);
    let n = 5;
    if (g * (n - g)) % 2 == 1 {
        result = result.scale(-1.0);
    }
    // Double dual should give back original. For simplicity:
    // Just use geometric product with pseudoscalar
    let ps = pseudoscalar();
    let gp = geometric_product(mv, &ps);
    let target = grade_of_multivector(mv);
    gp.grade_part(target)
}

fn pseudoscalar() -> Multivector {
    let mut mv = Multivector::new();
    mv.c[31] = 1.0; // e1e2e3epem = bitmask 11111 = 31
    mv
}

fn inverse_pseudoscalar() -> Multivector {
    // I⁻¹ = I / (I·I) = I / (-1) = -I (for Cl(3,1) with our metric)
    let mut mv = Multivector::new();
    mv.c[31] = -1.0;
    mv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multivector_new() {
        let mv = Multivector::new();
        for i in 0..32 {
            assert_eq!(mv.c[i], 0.0);
        }
    }

    #[test]
    fn test_multivector_scalar() {
        let mv = Multivector::from_scalar(5.0);
        assert_eq!(mv.scalar(), 5.0);
    }

    #[test]
    fn test_point_creation() {
        let p = point(1.0, 2.0, 3.0);
        assert!((p.c[1] - 1.0).abs() < 1e-10); // e1
        assert!((p.c[2] - 2.0).abs() < 1e-10); // e2
        assert!((p.c[4] - 3.0).abs() < 1e-10); // e3
        assert!((p.c[8] - 1.0).abs() < 1e-10); // ep
        assert!((p.c[16] - 7.0).abs() < 1e-10); // 0.5*(1+4+9) = 7
    }

    #[test]
    fn test_geometric_product_scalar() {
        let a = Multivector::from_scalar(3.0);
        let b = Multivector::from_scalar(4.0);
        let gp = geometric_product(&a, &b);
        assert!((gp.scalar() - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_geometric_product_vector_square() {
        // e1 * e1 should give 1 (metric is +1 for e1)
        let mut e1 = Multivector::new();
        e1.c[1] = 1.0;
        let gp = geometric_product(&e1, &e1);
        assert!((gp.scalar() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_geometric_product_vector_em_square() {
        // em * em should give -1 (metric is -1 for em)
        let mut em = Multivector::new();
        em.c[16] = 1.0;
        let gp = geometric_product(&em, &em);
        assert!((gp.scalar() - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_geometric_product_anticommutation() {
        // e1*e2 = -e2*e1 (bivector part)
        let mut e1 = Multivector::new();
        e1.c[1] = 1.0;
        let mut e2 = Multivector::new();
        e2.c[2] = 1.0;
        let e1e2 = geometric_product(&e1, &e2);
        let e2e1 = geometric_product(&e2, &e1);
        // e1e2 should be a bivector, e2e1 should be its negative
        assert!((e1e2.c[3] + e2e1.c[3]).abs() < 1e-10); // e1∧e2 = -e2∧e1
    }

    #[test]
    fn test_sphere_from_center_radius() {
        let s = sphere_from_center_radius(0.0, 0.0, 0.0, 1.0);
        // Center at origin with r=1: e1=0, e2=0, e3=0, ep=1, em=0-0.5=-0.5
        assert!((s.c[8] - 1.0).abs() < 1e-10);  // ep
        assert!((s.c[16] - (-0.5)).abs() < 1e-10); // em = 0 - 0.5*1
    }

    #[test]
    fn test_add() {
        let a = Multivector::from_scalar(3.0);
        let b = Multivector::from_scalar(4.0);
        let c = a.add(&b);
        assert!((c.scalar() - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_sub() {
        let a = Multivector::from_scalar(3.0);
        let b = Multivector::from_scalar(4.0);
        let c = a.sub(&b);
        assert!((c.scalar() - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_scale() {
        let mv = Multivector::from_scalar(5.0);
        let s = mv.scale(3.0);
        assert!((s.scalar() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_reverse() {
        let mut mv = Multivector::new();
        mv.c[0] = 1.0; // scalar - no sign change
        mv.c[3] = 1.0; // bivector e1e2 - sign change: (-1)^(2*1/2) = -1
        let rev = mv.reverse();
        assert!((rev.c[0] - 1.0).abs() < 1e-10);  // scalar unchanged
        assert!((rev.c[3] - (-1.0)).abs() < 1e-10); // bivector reversed
    }

    #[test]
    fn test_grade_part() {
        let mut mv = Multivector::new();
        mv.c[0] = 1.0;  // grade 0
        mv.c[1] = 2.0;  // grade 1
        mv.c[3] = 3.0;  // grade 2
        let g1 = mv.grade_part(1);
        assert_eq!(g1.c[1], 2.0);
        assert_eq!(g1.c[0], 0.0);
        assert_eq!(g1.c[3], 0.0);
    }

    #[test]
    fn test_norm() {
        let mv = Multivector::from_scalar(3.0);
        assert!((mv.norm() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_line_from_points() {
        let p1 = point(0.0, 0.0, 0.0);
        let p2 = point(1.0, 0.0, 0.0);
        let line = line_from_points(&p1, &p2);
        // Line should be a trivector (grade 3) or higher
        // Not zero at least
        let nonzero = line.c.iter().any(|&c| c.abs() > 1e-10);
        assert!(nonzero);
    }

    #[test]
    fn test_plane_from_points() {
        let p1 = point(0.0, 0.0, 0.0);
        let p2 = point(1.0, 0.0, 0.0);
        let p3 = point(0.0, 1.0, 0.0);
        let plane = plane_from_points(&p1, &p2, &p3);
        let nonzero = plane.c.iter().any(|&c| c.abs() > 1e-10);
        assert!(nonzero);
    }

    #[test]
    fn test_basis_product_e1_e1() {
        let metric = [1.0_f64, 1.0, 1.0, 1.0, -1.0];
        let (idx, sign) = basis_product(1, 1, &metric); // e1 * e1
        assert_eq!(idx, 0); // scalar
        assert!((sign - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_basis_product_e1_e2() {
        let metric = [1.0_f64, 1.0, 1.0, 1.0, -1.0];
        let (idx, sign) = basis_product(1, 2, &metric); // e1 * e2
        assert_eq!(idx, 3); // e1e2
        assert!((sign - 1.0).abs() < 1e-10);
    }
}
