//! # Blade — k-blade operations for geometric algebra
//!
//! A k-blade is the wedge (exterior) product of k vectors, representing a
//! k-dimensional subspace. This is exactly what Cyberloop calls "Grassmannian
//! subspace tracking" — the Grassmannian Gr(k,V) is the space of all k-blades
//! in V, up to scalar multiplication.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// A k-blade: a homogeneous multivector of a single grade.
///
/// Grade 0 = scalar, 1 = vector, 2 = bivector, 3 = trivector, 4 = pseudoscalar.
/// Components are stored in lexicographic basis order.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct Blade {
    #[wasm_bindgen(skip)]
    pub grade: usize,
    #[wasm_bindgen(skip)]
    pub components: Vec<f64>,
}

#[wasm_bindgen]
impl Blade {
    #[wasm_bindgen(constructor)]
    pub fn new(grade: usize, components: Vec<f64>) -> Blade {
        let expected = binomial(4, grade); // default to 4D
        let mut comps = components;
        if comps.len() < expected {
            comps.resize(expected, 0.0);
        }
        Blade { grade, components: comps }
    }

    pub fn scalar(s: f64) -> Blade {
        Blade { grade: 0, components: vec![s] }
    }

    pub fn vector(components: Vec<f64>) -> Blade {
        Blade { grade: 1, components }
    }

    pub fn bivector(components: Vec<f64>) -> Blade {
        Blade { grade: 2, components }
    }

    pub fn trivector(components: Vec<f64>) -> Blade {
        Blade { grade: 3, components }
    }

    pub fn pseudoscalar(s: f64) -> Blade {
        Blade { grade: 4, components: vec![s] }
    }

    pub fn grade(&self) -> usize {
        self.grade
    }

    pub fn components(&self) -> Vec<f64> {
        self.components.clone()
    }

    pub fn norm(&self) -> f64 {
        self.components.iter().map(|c| c * c).sum::<f64>().sqrt()
    }
}

// Rust-only impl block
impl Blade {
    pub fn zero(grade: usize, dimension: usize) -> Blade {
        Blade {
            grade,
            components: vec![0.0; binomial(dimension, grade)],
        }
    }

    /// Returns true if this blade is approximately zero.
    pub fn is_zero(&self, eps: f64) -> bool {
        self.components.iter().all(|c| c.abs() < eps)
    }
}

/// Compute the wedge (exterior) product of two blades.
///
/// The wedge product a ∧ b produces a (grade_a + grade_b)-blade representing
/// the union of the two subspaces. If the result would exceed the dimension,
/// it returns a zero blade (linearly dependent subspaces).
pub fn wedge(a: &Blade, b: &Blade, dimension: usize) -> Blade {
    let result_grade = a.grade + b.grade;
    if result_grade > dimension {
        return Blade::zero(result_grade.min(dimension), dimension);
    }

    let result_dim = binomial(dimension, result_grade);
    let mut result = vec![0.0; result_dim];

    // For the general case, we expand into basis blades and collect
    // Simplified implementation for dimensions <= 4
    match (a.grade, b.grade) {
        (0, _) | (_, 0) => {
            // Scalar wedge = scalar multiplication
            let scalar_blade = if a.grade == 0 { a } else { b };
            let other = if a.grade == 0 { b } else { a };
            let s = scalar_blade.components[0];
            result = other.components.iter().map(|c| s * c).collect();
        }
        (1, 1) => {
            // Vector ∧ Vector = Bivector
            // Bivector basis: e12, e13, e14, e23, e24, e34 (for 4D)
            let n = dimension;
            let mut idx = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    if i < a.components.len() && j < b.components.len() {
                        result[idx] = a.components[i] * b.components[j]
                            - a.components[j] * b.components[i];
                    }
                    idx += 1;
                }
            }
        }
        (1, 2) => {
            // Vector ∧ Bivector = Trivector
            let n = dimension;
            let mut idx = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    for k in (j + 1)..n {
                        // Get bivector component for (j,k)
                        let biv_jk = get_bivector_component(&b.components, j, k, n);
                        // Get vector component for i
                        let vec_i = if i < a.components.len() { a.components[i] } else { 0.0 };
                        result[idx] += vec_i * biv_jk;

                        // Get bivector component for (i,k) and vector for j
                        let biv_ik = get_bivector_component(&b.components, i, k, n);
                        let vec_j = if j < a.components.len() { a.components[j] } else { 0.0 };
                        result[idx] -= vec_j * biv_ik;

                        // Get bivector component for (i,j) and vector for k
                        let biv_ij = get_bivector_component(&b.components, i, j, n);
                        let vec_k = if k < a.components.len() { a.components[k] } else { 0.0 };
                        result[idx] += vec_k * biv_ij;

                        idx += 1;
                    }
                }
            }
        }
        (2, 1) => {
            // Bivector ∧ Vector = -Vector ∧ Bivector
            let neg = wedge(b, a, dimension);
            result = neg.components.iter().map(|c| -c).collect();
        }
        (1, 3) => {
            // Vector ∧ Trivector = Pseudoscalar
            let n = dimension;
            // Trivector component for (j,k,l), vector for i
            // Pseudoscalar = sum of det contributions
            let mut sum = 0.0;
            let mut tri_idx = 0;
            for j in 0..n {
                for k in (j + 1)..n {
                    for l in (k + 1)..n {
                        // We need to compute the signed determinant contribution
                        // For vector ∧ trivector = pseudoscalar component
                        // This is equivalent to contracting the vector with the trivector basis
                        if tri_idx < b.components.len() {
                            // Find which index is missing from (j,k,l)
                            let indices = [j, k, l];
                            for m in 0..n {
                                if !indices.contains(&m) {
                                    let sign = permutation_sign_4(j, k, l, m);
                                    let vi = if m < a.components.len() { a.components[m] } else { 0.0 };
                                    sum += sign * vi * b.components[tri_idx];
                                }
                            }
                        }
                        tri_idx += 1;
                    }
                }
            }
            result[0] = sum;
        }
        (3, 1) => {
            let neg = wedge(b, a, dimension);
            result = neg.components.iter().map(|c| -c).collect();
        }
        (2, 2) => {
            // Bivector ∧ Bivector = 0 if they share a common subspace in 4D,
            // or a pseudoscalar in 4D
            // In 4D: eij ∧ ekl where {i,j} ∩ {k,l} = ∅
            if dimension == 4 {
                // The only pairs that produce non-zero results:
                // e12∧e34, e13∧e24, e14∧e23
                let basis_pairs: [(usize, usize); 3] = [(0, 1), (0, 2), (0, 3)];
                for (bi, (i1, j1)) in basis_pairs.iter().enumerate() {
                    for (bj, (i2, j2)) in basis_pairs.iter().enumerate() {
                        if bi != bj {
                            // Check if they're disjoint
                            if *i1 != *i2 && *i1 != *j2 && *j1 != *i2 && *j1 != *j2 {
                                let biv_a = if bi < a.components.len() { a.components[bi] } else { 0.0 };
                                let biv_b = if bj < b.components.len() { b.components[bj] } else { 0.0 };
                                result[0] += biv_a * biv_b;
                            }
                        }
                    }
                }
                // Handle the full set of basis pairs for 4D bivector
                // Bivector basis in 4D: e01, e02, e03, e12, e13, e23
                // Wedge of non-overlapping pairs gives pseudoscalar
                let pairs: [(usize, usize); 6] = [
                    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
                ];
                result[0] = 0.0;
                for i in 0..6 {
                    for j in 0..6 {
                        let (i1, j1) = pairs[i];
                        let (i2, j2) = pairs[j];
                        if i1 != i2 && i1 != j2 && j1 != i2 && j1 != j2 {
                            let ai = if i < a.components.len() { a.components[i] } else { 0.0 };
                            let bj = if j < b.components.len() { b.components[j] } else { 0.0 };
                            let sign = wedge_bivector_sign(i1, j1, i2, j2);
                            result[0] += sign * ai * bj;
                        }
                    }
                }
            }
            // In 3D, bivector ∧ bivector = 0 (would be grade 4 > dim 3)
        }
        _ => {
            // For other grade combinations, return zero
        }
    }

    Blade { grade: result_grade, components: result }
}

/// Compute the Hodge dual of a blade in n dimensions.
///
/// The dual maps a k-blade to an (n-k)-blade, representing the orthogonal complement.
/// In GA terms: *B = B · I⁻¹ where I is the pseudoscalar.
pub fn dual(b: &Blade, n: usize) -> Blade {
    let dual_grade = n - b.grade;
    if b.grade == 0 {
        // Dual of scalar = pseudoscalar
        return Blade { grade: n, components: vec![b.components[0]] };
    }
    if b.grade == n {
        // Dual of pseudoscalar = scalar
        return Blade { grade: 0, components: b.components.clone() };
    }

    // General dual computation using the Levi-Civita symbol
    let dual_dim = binomial(n, dual_grade);
    let mut result = vec![0.0; dual_dim];

    // For each dual basis element, compute the contraction with the pseudoscalar
    // Simplified: use sign from permutation
    let k = b.grade;
    let biv_indices = basis_indices(n, k);
    let dual_indices = basis_indices(n, dual_grade);

    for (di, d_idx) in dual_indices.iter().enumerate() {
        for (bi, b_idx) in biv_indices.iter().enumerate() {
            // Check if b_idx ∪ d_idx = {0,1,...,n-1}
            let mut combined = b_idx.clone();
            combined.extend_from_slice(d_idx);
            combined.sort();
            let all_indices: Vec<usize> = (0..n).collect();
            if combined == all_indices {
                let sign = permutation_sign(&b_idx, d_idx, n);
                let bc = if bi < b.components.len() { b.components[bi] } else { 0.0 };
                result[di] += sign * bc;
            }
        }
    }

    Blade { grade: dual_grade, components: result }
}

/// Normalize a blade to unit magnitude.
pub fn normalize(b: &Blade) -> Blade {
    let n = b.norm();
    if n < 1e-15 {
        return b.clone();
    }
    Blade {
        grade: b.grade,
        components: b.components.iter().map(|c| c / n).collect(),
    }
}

/// Compute the angle between two subspaces represented by blades.
///
/// Uses the formula: θ = arccos(|⟨a,b⟩| / (|a| |b|))
/// where ⟨,⟩ is the scalar product.
pub fn subspace_angle(a: &Blade, b: &Blade) -> f64 {
    if a.grade != b.grade {
        // Different grades: compute the minimal principal angle
        // via the scalar part of the geometric product
        let dot: f64 = a.components
            .iter()
            .zip(b.components.iter())
            .map(|(x, y)| x * y)
            .sum();
        let na = a.norm();
        let nb = b.norm();
        if na < 1e-15 || nb < 1e-15 {
            return std::f64::consts::FRAC_PI_2;
        }
        let cos_val = (dot / (na * nb)).clamp(-1.0, 1.0);
        return cos_val.acos();
    }

    let dot: f64 = a.components
        .iter()
        .zip(b.components.iter())
        .map(|(x, y)| x * y)
        .sum();
    let na = a.norm();
    let nb = b.norm();

    if na < 1e-15 || nb < 1e-15 {
        return std::f64::consts::FRAC_PI_2;
    }

    let cos_val = (dot / (na * nb)).clamp(-1.0, 1.0);
    cos_val.acos()
}

/// Scalar product of two blades (same grade).
pub fn scalar_product(a: &Blade, b: &Blade) -> f64 {
    a.components
        .iter()
        .zip(b.components.iter())
        .map(|(x, y)| x * y)
        .sum()
}

/// Inner product of two blades.
pub fn inner_product(a: &Blade, b: &Blade, dimension: usize) -> Blade {
    if a.grade == 0 || b.grade == 0 {
        return Blade::zero(0, dimension);
    }

    let result_grade = if a.grade > b.grade { a.grade - b.grade } else { b.grade - a.grade };
    // Simplified: contraction
    if a.grade <= b.grade {
        // Left contraction: a ⌋ b
        contraction(a, b, dimension)
    } else {
        contraction(b, a, dimension)
    }
}

fn contraction(a: &Blade, b: &Blade, dimension: usize) -> Blade {
    let result_grade = b.grade.saturating_sub(a.grade);
    if result_grade == 0 {
        // Scalar result
        let dot: f64 = a.components.iter().zip(b.components.iter()).map(|(x, y)| x * y).sum();
        return Blade::scalar(dot);
    }
    Blade::zero(result_grade, dimension)
}

// ---- Helper functions ----

/// Binomial coefficient C(n, k).
pub fn binomial(n: usize, k: usize) -> usize {
    if k > n { return 0; }
    if k == 0 || k == n { return 1; }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Get bivector component for basis (i,j) where i < j, in n dimensions.
fn get_bivector_component(components: &[f64], i: usize, j: usize, n: usize) -> f64 {
    let mut idx = 0;
    for a in 0..n {
        for b in (a + 1)..n {
            if a == i && b == j {
                return if idx < components.len() { components[idx] } else { 0.0 };
            }
            idx += 1;
        }
    }
    0.0
}

/// Permutation sign for 4 indices.
fn permutation_sign_4(i: usize, j: usize, k: usize, l: usize) -> f64 {
    let mut arr = [i, j, k, l];
    let mut swaps = 0isize;
    for a in 0..4 {
        for b in (a + 1)..4 {
            if arr[a] > arr[b] {
                arr.swap(a, b);
                swaps += 1;
            }
        }
    }
    if swaps % 2 == 0 { 1.0 } else { -1.0 }
}

/// Sign for wedge of two bivector basis elements.
fn wedge_bivector_sign(i1: usize, j1: usize, i2: usize, j2: usize) -> f64 {
    let mut arr = [i1, j1, i2, j2];
    let mut swaps = 0isize;
    for a in 0..4 {
        for b in (a + 1)..4 {
            if arr[a] > arr[b] {
                arr.swap(a, b);
                swaps += 1;
            }
        }
    }
    if swaps % 2 == 0 { 1.0 } else { -1.0 }
}

/// Get all k-element subsets of {0, ..., n-1}, in lexicographic order.
fn basis_indices(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > n {
        return vec![];
    }

    fn choose(from: &[usize], k: usize) -> Vec<Vec<usize>> {
        if k == 0 {
            return vec![vec![]];
        }
        if from.len() < k {
            return vec![];
        }
        let mut result = Vec::new();
        for (i, &first) in from.iter().enumerate() {
            let rest = &from[i + 1..];
            for mut combo in choose(rest, k - 1) {
                let mut full = vec![first];
                full.append(&mut combo);
                result.push(full);
            }
        }
        result
    }

    let indices: Vec<usize> = (0..n).collect();
    choose(&indices, k)
}

/// Permutation sign when concatenating two sets of indices and sorting to {0,...,n-1}.
fn permutation_sign(a_idx: &[usize], b_idx: &[usize], n: usize) -> f64 {
    let mut combined: Vec<usize> = a_idx.to_vec();
    combined.extend_from_slice(b_idx);
    let mut swaps = 0isize;
    for i in 0..combined.len() {
        for j in (i + 1)..combined.len() {
            if combined[i] > combined[j] {
                combined.swap(i, j);
                swaps += 1;
            }
        }
    }
    // Verify it's actually {0,...,n-1}
    let expected: Vec<usize> = (0..n).collect();
    if combined != expected {
        return 0.0;
    }
    if swaps % 2 == 0 { 1.0 } else { -1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_blade_scalar() {
        let s = Blade::scalar(3.0);
        assert_eq!(s.grade, 0);
        assert_eq!(s.components, vec![3.0]);
    }

    #[test]
    fn test_blade_vector() {
        let v = Blade::vector(vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(v.grade, 1);
        assert_eq!(v.components, vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_blade_norm() {
        let v = Blade::vector(vec![3.0, 4.0]);
        assert!((v.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let v = Blade::vector(vec![3.0, 4.0, 0.0, 0.0]);
        let n = normalize(&v);
        assert!((n.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_wedge_scalar_vector() {
        let s = Blade::scalar(2.0);
        let v = Blade::vector(vec![3.0, 4.0, 0.0, 0.0]);
        let result = wedge(&s, &v, 4);
        assert_eq!(result.grade, 1);
        assert!((result.components[0] - 6.0).abs() < 1e-10);
        assert!((result.components[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_wedge_vector_vector() {
        let a = Blade::vector(vec![1.0, 0.0, 0.0, 0.0]);
        let b = Blade::vector(vec![0.0, 1.0, 0.0, 0.0]);
        let result = wedge(&a, &b, 4);
        assert_eq!(result.grade, 2);
        // Bivector e12 component should be 1
        assert!((result.components[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_wedge_anticommutative() {
        let a = Blade::vector(vec![1.0, 2.0, 0.0, 0.0]);
        let b = Blade::vector(vec![3.0, 4.0, 0.0, 0.0]);
        let ab = wedge(&a, &b, 4);
        let ba = wedge(&b, &a, 4);
        // a∧b = -b∧a
        for i in 0..ab.components.len() {
            assert!((ab.components[i] + ba.components[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dual_vector_3d() {
        // In 3D, dual of e1 = e23 (up to sign)
        let v = Blade::vector(vec![1.0, 0.0, 0.0]);
        let d = dual(&v, 3);
        assert_eq!(d.grade, 2); // 3-1 = 2
        // The dual of e1 in 3D should give e23 with appropriate sign
        // e1* = e1·I⁻¹ = e1·(e123)⁻¹ = e23 (with positive sign for Euclidean)
        assert!(d.components.iter().any(|&c| c.abs() > 0.5));
    }

    #[test]
    fn test_dual_pseudoscalar() {
        let ps = Blade::pseudoscalar(5.0);
        let d = dual(&ps, 4);
        assert_eq!(d.grade, 0);
        assert!((d.components[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_dual_of_dual() {
        // **B should give ±B
        let v = Blade::vector(vec![1.0, 0.0, 0.0]);
        let d1 = dual(&v, 3);
        let d2 = dual(&d1, 3);
        assert_eq!(d2.grade, v.grade);
        // Up to sign
        let sign = if d2.components[0] * v.components[0] >= 0.0 { 1.0 } else { -1.0 };
        for i in 0..v.components.len() {
            assert!((sign * d2.components[i] - v.components[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_subspace_angle_parallel() {
        let a = Blade::vector(vec![1.0, 0.0, 0.0]);
        let b = Blade::vector(vec![2.0, 0.0, 0.0]);
        let angle = subspace_angle(&a, &b);
        assert!(angle.abs() < 1e-10);
    }

    #[test]
    fn test_subspace_angle_orthogonal() {
        let a = Blade::vector(vec![1.0, 0.0, 0.0]);
        let b = Blade::vector(vec![0.0, 1.0, 0.0]);
        let angle = subspace_angle(&a, &b);
        assert!((angle - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_subspace_angle_45deg() {
        let a = Blade::vector(vec![1.0, 0.0, 0.0]);
        let b = Blade::vector(vec![1.0, 1.0, 0.0]);
        let angle = subspace_angle(&a, &b);
        assert!((angle - PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(4, 0), 1);
        assert_eq!(binomial(4, 1), 4);
        assert_eq!(binomial(4, 2), 6);
        assert_eq!(binomial(4, 3), 4);
        assert_eq!(binomial(4, 4), 1);
        assert_eq!(binomial(5, 2), 10);
    }

    #[test]
    fn test_is_zero() {
        let v = Blade::vector(vec![0.0, 0.0, 0.0]);
        assert!(v.is_zero(1e-10));
        let v2 = Blade::vector(vec![1e-16, 1e-16]);
        assert!(v2.is_zero(1e-10));
        let v3 = Blade::vector(vec![1.0, 0.0]);
        assert!(!v3.is_zero(1e-10));
    }

    #[test]
    fn test_wedge_vector_bivector() {
        let v = Blade::vector(vec![0.0, 0.0, 1.0, 0.0]); // e3
        let bv = Blade::bivector(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]); // e12
        let result = wedge(&v, &bv, 4);
        assert_eq!(result.grade, 3);
        // e3 ∧ e12 = e312, which in lex order is e123
        assert!(result.components.iter().any(|&c| c.abs() > 0.5));
    }

    #[test]
    fn test_blade_new_truncates() {
        let b = Blade::new(2, vec![1.0, 2.0]); // Only 2 components for 4D bivector (needs 6)
        assert_eq!(b.components.len(), 6);
        assert_eq!(b.components[0], 1.0);
        assert_eq!(b.components[1], 2.0);
        assert_eq!(b.components[2], 0.0);
    }

    #[test]
    fn test_scalar_product() {
        let a = Blade::vector(vec![1.0, 2.0, 3.0]);
        let b = Blade::vector(vec![4.0, 5.0, 6.0]);
        let sp = scalar_product(&a, &b);
        assert!((sp - 32.0).abs() < 1e-10); // 1*4 + 2*5 + 3*6 = 32
    }
}
