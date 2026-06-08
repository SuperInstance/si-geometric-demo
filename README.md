# si-geometric-demo

**Proof of concept: geometric algebra multivectors in WASM — proving Cyberloop's Grassmannian tracking IS ga-core**

[![Built with Rust](https://img.shields.io/badge/built%20with-Rust-orange)](https://www.rust-lang.org/)
[![WASM Ready](https://img.shields.io/badge/WASM-ready-blueviolet)](https://webassembly.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Table of Contents

1. [The Convergence Thesis](#the-convergence-thesis)
2. [The Key Insight](#the-key-insight)
3. [Mathematical Foundations](#mathematical-foundations)
4. [How Grassmannian Tracking Maps to k-Blades](#how-grassmannian-tracking-maps-to-k-blades)
5. [How Riemannian Steps Map to Rotors](#how-riemannian-steps-map-to-rotors)
6. [Why This Matters for Agent Engineering](#why-this-matters-for-agent-engineering)
7. [Architecture](#architecture)
8. [Module Reference](#module-reference)
9. [API Documentation](#api-documentation)
10. [The Demo](#the-demo)
11. [Building & Running](#building--running)
12. [Correspondence Table](#correspondence-table)
13. [Examples](#examples)
14. [Performance](#performance)
15. [Roadmap](#roadmap)
16. [Contributing](#contributing)
17. [License](#license)

---

## The Convergence Thesis

Two independent lines of research have converged on the same mathematical structure:

1. **Cyberloop v4.0** uses Grassmannian subspace tracking for agent step control — treating agent states as points on the Grassmannian manifold Gr(k,V) and using Riemannian geometry to control transitions.

2. **Geometric Algebra** represents subspaces as k-blades (exterior products of vectors) and rotations between subspaces as rotors (even-grade multivectors).

**These are the same thing.** This crate proves it by implementing both perspectives in the same codebase, showing that every Cyberloop operation has a direct, one-to-one mapping to a GA operation.

This isn't just mathematical aesthetics. Recognizing this equivalence gives us:
- Access to GA's rich toolbox (conformal geometry, duality, meet/join)
- Cleaner implementations (rotors compose naturally, no matrix exponentials)
- Deeper understanding (subspace angle = Grassmannian distance = GA inner product)

---

## The Key Insight

The Grassmannian **Gr(k, V)** is the space of all k-dimensional subspaces of a vector space V. In geometric algebra:

- A **k-blade** (the wedge product of k vectors) represents exactly a k-dimensional subspace
- The space of k-blades (up to scalar multiplication) **is** the Grassmannian
- The **Riemannian metric** on Gr(k,V) comes from the inner product on blades
- **Geodesics** on Gr(k,V) correspond to **rotor interpolation** in GA

So when Cyberloop says it "tracks agent state on the Grassmannian," it is — whether it knows it or not — doing geometric algebra.

```
Cyberloop's Language          Geometric Algebra Language
─────────────────────         ──────────────────────────
Grassmannian point            k-blade
Subspace tracking             Blade tracking
Riemannian geodesic           Rotor interpolation
Divergence metric             Subspace angle
Step control                  Rotor magnitude limiting
Trajectory prediction         Rotor extrapolation
```

---

## Mathematical Foundations

### The Grassmannian as a Manifold

The Grassmannian Gr(k, V) where V is an n-dimensional vector space is the set of all k-dimensional linear subspaces of V. It is a compact Riemannian manifold of dimension k(n-k).

Key properties:
- Gr(1, V) = RP^(n-1) (real projective space)
- Gr(k, V) ≅ Gr(n-k, V) via orthogonal complement (Hodge dual)
- The canonical metric comes from the embedding in the exterior algebra ⋀^k(V)

### k-Blades and the Exterior Algebra

The exterior (Grassmann) algebra ⋀(V) is the direct sum of all exterior powers:

```
⋀(V) = ⋀⁰(V) ⊕ ⋀¹(V) ⊕ ⋀²(V) ⊕ ... ⊕ ⋀ⁿ(V)
      = ℝ    ⊕ V    ⊕ ⋀²V  ⊕ ... ⊕ ⋀ⁿV
```

A **k-blade** is a decomposable element of ⋀^k(V), i.e., one that can be written as:

```
B = v₁ ∧ v₂ ∧ ... ∧ vₖ
```

The key insight: **a k-blade determines a k-dimensional subspace** (its span), and every k-dimensional subspace has a unique (up to scalar) representing k-blade.

### Geometric Algebra: The Full Picture

Geometric algebra extends the exterior algebra with a metric-dependent product — the **geometric product**. For vectors a, b:

```
ab = a · b + a ∧ b
```

This combines the inner product (scalar) and exterior product (bivector) into a single operation. The geometric algebra Cl(V, Q) over a vector space V with quadratic form Q is:

```
Cl(V, Q) = ⋀(V)  (as a vector space)
```

but with a richer product structure that respects the metric.

### Rotors: Rotation as Algebra

A **rotor** R in Cl(V, Q) is an even-grade multivector satisfying:

```
RR̃ = 1    (R̃ is the reverse of R)
```

Rotors act on the algebra via the **sandwich product**:

```
v' = RvR̃
```

For a rotor R = exp(B/2) where B is a bivector, this rotates vectors in the plane of B by an angle |B|.

**The crucial connection:** On the Grassmannian, a geodesic from point A to point B corresponds to a rotor R such that:

```
B = RAR̃
```

The Riemannian distance between A and B equals the angle of the rotor R.

---

## How Grassmannian Tracking Maps to k-Blades

### State → Blade Projection

When Cyberloop projects an agent's high-dimensional state vector s ∈ ℝⁿ to a Grassmannian point in Gr(k, ℝⁿ), it's computing a k-blade:

**Cyberloop's process:**
1. Collect state vectors s₁, s₂, ..., sₖ (via SVD or PCA)
2. Form the subspace S = span{s₁, ..., sₖ}
3. This is a point on Gr(k, ℝⁿ)

**GA equivalent:**
1. Take the same vectors
2. Compute the blade B = s₁ ∧ s₂ ∧ ... ∧ sₖ
3. This is the SAME subspace, represented as a blade

```rust
// GA version (this crate)
let v1 = Blade::vector(vec![1.0, 0.0, 0.0, 0.0]);
let v2 = Blade::vector(vec![0.0, 1.0, 0.0, 0.0]);
let bivector = wedge(&v1, &v2, 4);  // e₁₂ = plane containing v1 and v2
```

### Tracking Through Time

As the agent evolves, its state subspace changes. Cyberloop tracks this as a trajectory on Gr(k,V). In GA, this is a trajectory in the space of k-blades:

```rust
let mut tracker = SubspaceTracker::new(4);
tracker.track(vec![1.0, 0.0, 0.0, 0.0]);  // First state → blade
tracker.update(vec![0.9, 0.1, 0.0, 0.0]);  // Second state → new blade + rotor
```

Each call to `update` produces both a new blade (current subspace) and a rotor (the geodesic from old to new).

### Divergence Measurement

Cyberloop measures divergence between agents as the Riemannian distance on Gr(k,V). In GA, this is the **subspace angle**:

```rust
let angle = subspace_angle(&blade_a, &blade_b);
// This IS the Riemannian distance on the Grassmannian
```

For 1-blades (vectors), this reduces to the familiar angle between vectors. For higher-grade blades, it generalizes naturally via the inner product on the exterior algebra.

---

## How Riemannian Steps Map to Rotors

### The Step Function

In Cyberloop, a "Riemannian step" on Gr(k,V) moves from one subspace to a nearby one along the geodesic:

**Cyberloop:**
```
S(t) = Exp_{S₀}(t · V)    where V ∈ T_{S₀}Gr(k,V)
```

**GA equivalent:**
```
B(t) = R(t) · B₀ · R̃(t)    where R(t) = exp(t·B/2) is a rotor
```

These are identical operations expressed in different languages. The tangent vector V on the Grassmannian corresponds to the bivector B in the rotor.

### Rotor Interpolation = Geodesic Interpolation

```rust
let rotor_a = Rotor::from_blades(&from_blade, &to_blade);
let rotor_mid = Rotor::slerp(&Rotor::identity(), &rotor_a, 0.5);
```

This is `slerp` — spherical linear interpolation — which traces out exactly the geodesic on the Grassmannian. The parameter t ∈ [0, 1] interpolates smoothly between subspaces.

### Step Control via Rotor Magnitude

Cyberloop controls step size by limiting the Riemannian distance of each step. In GA, this is limiting the angle of the rotor:

```rust
let rotor = Rotor::from_blades(&current, &target);
let angle = rotor.scalar_part().acos() * 2.0;  // Rotor angle
if angle > max_step {
    rotor = Rotor::slerp(&identity, &rotor, max_step / angle);
}
```

This is cleaner than the matrix-based approach because rotors compose multiplicatively and never degenerate (unlike rotation matrices near singularities).

### Prediction via Rotor Extrapolation

Cyberloop predicts future agent states by extrapolating the Grassmannian trajectory. In GA, this is rotor extrapolation:

```rust
let predictions = tracker.predict(10);  // Predict 10 steps ahead
```

The prediction works by:
1. Estimating the angular velocity from recent rotors
2. Extrapolating the rotor forward in time
3. Applying the extrapolated rotor to the current blade

---

## Why This Matters for Agent Engineering

### 1. Cleaner Code

A rotor is just a scalar + bivector. No need for:
- Matrix exponentials (rotors use half-angle trig)
- Singular value decompositions (blades are explicit)
- Special handling for edge cases (rotors handle all dimensions uniformly)

### 2. Composable Operations

In GA, operations compose naturally:
- Rotation + Rotation = Rotation (rotor multiplication)
- Subspace + Complement = Full space (wedge product)
- Intersection = Meet operation
- Union = Join operation

Cyberloop re-implements these from scratch. GA gives them for free.

### 3. Richer Geometry

Conformal geometric algebra (Cl(3,1)) adds:
- **Spheres** as first-class objects (not just point clouds)
- **Tangent planes** as natural constructs
- **Intersections** via the meet operation
- **Distances** via the inner product of conformal points

This could enable Cyberloop to reason about agent states in terms of geometric relationships (near, far, intersecting, contained) rather than just numerical distances.

### 4. WASM-Native

Geometric algebra operations are:
- **SIMD-friendly** (regular array operations)
- **Branch-free** (no special cases)
- **Cache-friendly** (small fixed-size arrays)

This makes them ideal for WASM deployment — exactly where Cyberloop needs to run.

### 5. Theoretical Clarity

When you recognize that Grassmannian tracking IS geometric algebra, you gain:
- Access to 150+ years of mathematical literature
- Proven algorithms for every operation
- Clear relationships between seemingly different concepts
- A unified language for discussing agent geometry

---

## Architecture

```
si-geometric-demo/
├── src/
│   ├── lib.rs          # WASM entry point + top-level API
│   ├── blade.rs        # k-blade operations (exterior algebra)
│   ├── conformal.rs    # Conformal GA Cl(3,1) — points, lines, spheres
│   ├── rotor.rs        # Rotors — rotation, interpolation, sandwich product
│   ├── tracking.rs     # Subspace tracking (Cyberloop connection)
│   └── agent_demo.rs   # Agent trajectory demo
├── demo.html           # Interactive browser visualization
├── Cargo.toml          # Build configuration
└── README.md           # This file
```

The architecture mirrors the Cyberloop→GA mapping:

```
┌─────────────────────────────────────────────────────────┐
│  Cyberloop Concept          │  GA Module               │
│  ─────────────────          │  ──────────               │
│  Subspace state             │  blade::Blade             │
│  State projection           │  tracking::track()        │
│  Geodesic step              │  rotor::Rotor             │
│  Step interpolation         │  rotor::slerp()           │
│  Divergence metric          │  blade::subspace_angle()  │
│  State transition           │  tracking::update()       │
│  Prediction                 │  tracking::predict()      │
│  Spatial reasoning          │  conformal::*             │
└─────────────────────────────────────────────────────────┘
```

---

## Module Reference

### `blade` — k-Blade Operations

The foundation module. k-blades are the mathematical objects that represent subspaces.

| Function | Description | Cyberloop Equivalent |
|----------|-------------|---------------------|
| `Blade::new(grade, comps)` | Create a k-blade | State → subspace |
| `wedge(a, b, dim)` | Exterior product | Subspace union |
| `dual(b, n)` | Hodge dual | Normal space |
| `normalize(b)` | Unit blade | Canonical subspace |
| `subspace_angle(a, b)` | Angle between subspaces | Riemannian distance |
| `scalar_product(a, b)` | Inner product | Subspace overlap |

### `conformal` — Conformal Geometric Algebra (Cl(3,1))

Extends 3D Euclidean space with origin and infinity basis vectors for unified treatment of geometric objects.

| Function | Description |
|----------|-------------|
| `point(x, y, z)` | Conformal point representation |
| `line_from_points(p1, p2)` | Line through two points |
| `plane_from_points(p1, p2, p3)` | Plane through three points |
| `sphere_from_center_radius(c, r)` | Sphere from center and radius |
| `meet(a, b)` | Intersection of geometric objects |
| `join(a, b)` | Union of geometric objects |
| `geometric_product(a, b)` | Full GA product |

### `rotor` — Rotation and Interpolation

Rotors are the GA generalization of quaternions. They encode rotations as even-grade multivectors.

| Function | Description | Cyberloop Equivalent |
|----------|-------------|---------------------|
| `Rotor::from_axis_angle(ax, ay, az, θ)` | Rotation from axis-angle | Step specification |
| `Rotor::slerp(a, b, t)` | Spherical interpolation | Geodesic interpolation |
| `sandwich(r, v)` | Apply rotation | Apply step |
| `Rotor::from_blades(from, to)` | Rotor between subspaces | Riemannian geodesic |

### `tracking` — Subspace Tracking

The module that directly mirrors Cyberloop's Grassmannian tracking.

| Function | Description | Cyberloop Equivalent |
|----------|-------------|---------------------|
| `SubspaceTracker::new(dim)` | Initialize tracker | Initialize Gr(k,V) tracker |
| `track(state)` | Project state → blade | Project to Grassmannian |
| `update(state)` | Track transition → rotor | Riemannian step |
| `divergence(other)` | Distance between trackers | Agent divergence |
| `predict(steps)` | Extrapolate trajectory | State prediction |

### `agent_demo` — Agent Trajectory Demo

Demonstrates the full pipeline with simulated agents.

| Function | Description |
|----------|-------------|
| `AgentState::new(pos, mom)` | Create agent state |
| `step_hamiltonian(state, dt, V)` | Hamiltonian evolution |
| `track_trajectory(traj)` | Track full trajectory |
| `run_agent_demo(steps)` | Run demo, return JSON |

---

## API Documentation

### Creating and Working with Blades

```rust
use si_geometric_demo::blade::*;

// Create a vector (1-blade)
let v = Blade::vector(vec![1.0, 2.0, 3.0, 0.0]);
assert_eq!(v.grade(), 1);

// Create a scalar (0-blade)
let s = Blade::scalar(3.14);
assert_eq!(s.grade(), 0);

// Wedge product: vector ∧ vector = bivector
let a = Blade::vector(vec![1.0, 0.0, 0.0, 0.0]);  // e₁
let b = Blade::vector(vec![0.0, 1.0, 0.0, 0.0]);  // e₂
let e12 = wedge(&a, &b, 4);  // e₁∧e₂ (bivector)
assert_eq!(e12.grade(), 2);

// Normalize to unit blade
let unit = normalize(&v);

// Subspace angle
let angle = subspace_angle(&a, &b);
assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
```

### Conformal Geometry

```rust
use si_geometric_demo::conformal::*;

// Create points in conformal GA
let p1 = point(1.0, 0.0, 0.0);
let p2 = point(0.0, 1.0, 0.0);
let p3 = point(0.0, 0.0, 1.0);

// Line through two points
let line = line_from_points(&p1, &p2);

// Plane through three points
let plane = plane_from_points(&p1, &p2, &p3);

// Sphere from center and radius
let sphere = sphere_from_center_radius(0.0, 0.0, 0.0, 1.0);

// Intersection (meet)
let intersection = meet(&line, &plane);
```

### Rotors

```rust
use si_geometric_demo::rotor::*;
use si_geometric_demo::conformal::Multivector;
use std::f64::consts::PI;

// Rotation from axis-angle
let r = Rotor::from_axis_angle(0.0, 0.0, 1.0, PI / 2.0);

// Apply rotation (sandwich product)
let mut v = Multivector::new();
v.c[1] = 1.0;  // e₁
let rotated = r.sandwich(&v);
// e₁ rotated 90° around z → approximately e₂

// Interpolation
let a = Rotor::new();  // identity
let b = Rotor::from_axis_angle(0.0, 0.0, 1.0, PI);
let mid = Rotor::slerp(&a, &b, 0.5);  // 90° rotation

// Rotor between subspaces
let from = Blade::vector(vec![1.0, 0.0, 0.0, 0.0]);
let to = Blade::vector(vec![0.0, 1.0, 0.0, 0.0]);
let r = Rotor::from_blades(&from, &to);
```

### Subspace Tracking (Cyberloop Mode)

```rust
use si_geometric_demo::tracking::SubspaceTracker;
use std::f64::consts::PI;

let mut tracker = SubspaceTracker::new(4);

// Project states to blades
tracker.track(vec![1.0, 0.0, 0.0, 0.0]);
let rotor = tracker.update(vec![0.0, 1.0, 0.0, 0.0]);

// Measure divergence between agents
let mut tracker2 = SubspaceTracker::new(4);
tracker2.track(vec![1.0, 1.0, 0.0, 0.0]);
let div = tracker.divergence(&tracker2);

// Predict future trajectory
let predictions = tracker.predict(5);
```

### Agent Demo

```rust
use si_geometric_demo::agent_demo::run_agent_demo;

// Run simulation with 100 steps, returns JSON
let json = run_agent_demo(100);
println!("{}", json);
```

### WASM API

When compiled to WASM, these functions are available from JavaScript:

```javascript
import init, {
  demo,
  wedge_product,
  compute_subspace_angle,
  make_rotor,
  apply_rotor,
  version
} from './pkg/si_geometric_demo.js';

await init();

// Run the demo
const result = JSON.parse(demo(50));

// Compute wedge product
const bivector = wedge_product([1, 0, 0, 0], [0, 1, 0, 0]);

// Subspace angle
const angle = compute_subspace_angle([1, 0, 0], [0, 1, 0]);

// Create and apply rotor
const rotor = make_rotor(0, 0, 1, Math.PI / 2);
const rotated = apply_rotor(rotor, 1, 0, 0);
```

---

## The Demo

Open `demo.html` in any browser. No build step required — it includes a pure JavaScript implementation of the core GA operations.

### What You'll See

The demo shows two side-by-side views of the same simulation:

**Left (Cyberloop's View):**
- 3 agents moving through 2D "cognitive space"
- Each agent's subspace visualized as an ellipse showing its current direction
- Dashed lines between agents showing subspace divergence
- This is what Cyberloop "sees" — agents as points on the Grassmannian

**Right (GA View):**
- The same agents, but rendered as k-blades (wedge shapes)
- Rotor arcs showing the geometric rotation between steps
- The bivector plane indicator at each agent
- This is the GA interpretation — the same data, richer visualization

### Controls

- **Play/Pause**: Start or stop the simulation
- **Reset**: Reset agents to initial positions
- **Speed**: Control simulation speed (1-10)
- **Agents**: Number of agents (1-6)

### Info Panel

- **Step**: Current simulation step count
- **Avg Divergence**: Mean subspace angle between all agent pairs
- **Total Rotation**: Cumulative rotor angle across all agents
- **Prediction Error**: Difference between predicted and actual subspace states

---

## Building & Running

### Native Build

```bash
cargo build
cargo test
```

### WASM Build

```bash
wasm-pack build --target web --out-dir pkg
```

### Run Demo

```bash
# Just open demo.html in a browser
open demo.html
# or
python3 -m http.server 8080  # then visit http://localhost:8080/demo.html
```

### Run Tests

```bash
cargo test                    # All 81 tests
cargo test --lib blade        # Blade tests only
cargo test --lib rotor        # Rotor tests only
cargo test --lib tracking     # Tracking tests only
cargo test --lib conformal    # Conformal tests only
cargo test --lib agent_demo   # Agent demo tests only
```

---

## Correspondence Table

The complete mapping between Cyberloop concepts and geometric algebra:

| Cyberloop Concept | GA Concept | Implementation | Mathematical Form |
|---|---|---|---|
| Agent state | State vector in ℝⁿ | `Blade::vector()` | v ∈ V |
| State subspace | k-blade | `Blade { grade: k, ... }` | B = v₁∧...∧vₖ ∈ ⋀ᵏV |
| Grassmannian Gr(k,V) | Space of k-blades | `Blade` type | Gr(k,V) ≅ {k-blades}/ℝ* |
| Subspace projection | Wedge product | `wedge()` | B = v₁∧v₂∧...∧vₖ |
| Orthogonal complement | Hodge dual | `dual()` | B* = BI⁻¹ |
| Riemannian geodesic | Rotor | `Rotor` | R = exp(B/2) |
| Geodesic interpolation | Rotor slerp | `Rotor::slerp()` | R(t) = R₁(R₂R₁⁻¹)^t |
| Step application | Sandwich product | `sandwich()` | v' = RvR̃ |
| Riemannian distance | Subspace angle | `subspace_angle()` | θ = arccos(⟨A,B⟩/(|A||B|)) |
| Step control | Rotor angle limiting | `slerp` with clamp | ‖R‖ ≤ max_step |
| Trajectory | Rotor sequence | `SubspaceTracker::rotors` | R₁, R₂, ..., Rₙ |
| Prediction | Rotor extrapolation | `predict()` | R_pred = R_last^t |
| Agent divergence | Subspace divergence | `divergence()` | θ(A,B) = arccos(...) |
| State potential | Hamiltonian | `step_hamiltonian()` | Ḣ = T + V |
| Momentum direction | Heading blade | `AgentState::heading` | normalized momentum |

---

## Examples

### Example 1: Basic Blade Operations

```rust
use si_geometric_demo::blade::*;

// Two vectors in 4D
let a = Blade::vector(vec![1.0, 0.0, 0.0, 0.0]);
let b = Blade::vector(vec![0.0, 1.0, 0.0, 0.0]);

// Their wedge product is a bivector (2-blade)
let biv = wedge(&a, &b, 4);
println!("Grade: {}", biv.grade());      // 2
println!("Norm: {}", biv.norm());         // 1.0
println!("Components: {:?}", biv.components());  // [1, 0, 0, 0, 0, 0]

// Dual gives the orthogonal complement (also 2-blade in 4D)
let d = dual(&biv, 4);
println!("Dual grade: {}", d.grade());  // 2 (4-2=2)

// Angle between parallel vectors is 0
let c = Blade::vector(vec![2.0, 0.0, 0.0, 0.0]);
assert_eq!(subspace_angle(&a, &c), 0.0);

// Angle between orthogonal vectors is π/2
assert_eq!(subspace_angle(&a, &b), std::f64::consts::FRAC_PI_2);
```

### Example 2: Tracking an Agent's Cognitive Trajectory

```rust
use si_geometric_demo::tracking::SubspaceTracker;
use std::f64::consts::PI;

// Agent starts heading in x-direction
let mut tracker = SubspaceTracker::new(4);
tracker.track(vec![1.0, 0.0, 0.0, 0.0]);

// Simulate agent gradually turning
for i in 1..=10 {
    let angle = i as f64 * 0.1;
    let state = vec![angle.cos(), angle.sin(), 0.0, 0.0];
    let rotor = tracker.update(state);
    println!("Step {}: rotor scalar = {:.4}", i, rotor.scalar_part());
}

// Predict where the agent is heading
let predictions = tracker.predict(5);
for (i, pred) in predictions.iter().enumerate() {
    println!("Prediction {}: {:?}", i + 1, pred.components);
}

// Compare with another agent
let mut other = SubspaceTracker::new(4);
other.track(vec![0.0, 0.0, 1.0, 0.0]);
println!("Divergence: {:.2}°", tracker.divergence(&other) * 180.0 / PI);
```

### Example 3: Conformal Geometry

```rust
use si_geometric_demo::conformal::*;

// Three points defining a triangle
let p1 = point(1.0, 0.0, 0.0);
let p2 = point(0.0, 1.0, 0.0);
let p3 = point(0.0, 0.0, 1.0);

// Line through p1 and p2
let line = line_from_points(&p1, &p2);

// Plane through all three
let plane = plane_from_points(&p1, &p2, &p3);

// Unit sphere at origin
let sphere = sphere_from_center_radius(0.0, 0.0, 0.0, 1.0);
```

### Example 4: Full Agent Simulation

```rust
use si_geometric_demo::agent_demo::*;

// Run 100 steps of the multi-agent demo
let json_output = run_agent_demo(100);

// Parse and analyze
let data: serde_json::Value = serde_json::from_str(&json_output).unwrap();
for agent in data["agents"].as_array().unwrap() {
    println!("Agent {} positions: {}",
        agent["agent"],
        agent["positions"].as_array().unwrap().len()
    );
}
```

---

## Performance

All operations are O(2^n) where n is the dimension of the underlying vector space, which is constant for fixed n. For our default Cl(3,1) (32-component multivectors):

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Wedge product | O(1) | Fixed-size basis combinations |
| Geometric product | O(2^2n) | 32×32 = 1024 component pairs |
| Rotor sandwich | O(2^2n) | Two geometric products |
| Subspace angle | O(n) | Inner product + norm |
| Track | O(n) | Normalization |
| Predict | O(k × 2^2n) | k extrapolation steps |

For WASM targets, the geometric product can be further optimized with SIMD instructions, reducing the constant factor significantly.

---

## Roadmap

- [ ] SIMD optimization for geometric product
- [ ] Higher-dimensional support (Cl(4,1), Cl(5,1))
- [ ] WebGPU visualization
- [ ] Integration with Cyberloop's actual step control
- [ ] Benchmark suite comparing GA vs. matrix approaches
- [ ] Q-GA convergence analysis (quantum geometric algebra)
- [ ] Subspace clustering for multi-agent coordination

---

## Contributing

This is a proof-of-concept demonstrating the Cyberloop↔GA equivalence. Contributions welcome:

1. Fork the repo
2. Create a feature branch
3. Add tests for any new operations
4. Ensure `cargo test` passes (81+ tests)
5. Submit a PR

---

## License

MIT — see [LICENSE](LICENSE)

---

## Acknowledgments

- **William Kingdon Clifford** — invented geometric algebra (1878)
- **Hermann Grassmann** — invented exterior algebra (1844)
- **Marcel Riesz** — developed the Clifford analysis framework
- **Leo Dorst, Daniel Fontijne, Stephen Mann** — "Geometric Algebra for Computer Science"
- **Cyberloop v4.0** — for independently rediscovering GA and calling it "Grassmannian tracking"

---

> *"The Grassmannian Gr(k,V) is the space of k-blades. Cyberloop tracks on Gr(k,V).*
> *Therefore Cyberloop tracks k-blades. Q.E.D."*

— The si-geometric-demo thesis
