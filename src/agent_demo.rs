//! # Agent Demo — agent trajectory as a geometric object
//!
//! This module demonstrates the full pipeline:
//! 1. Agent moves through "cognitive space" (high-dimensional state)
//! 2. Each state is projected to a k-blade (Grassmannian point)
//! 3. Transitions between states are rotors (Grassmannian geodesics)
//! 4. The full trajectory is a curve on the Grassmannian
//!
//! This proves that Cyberloop's "Grassmannian subspace tracking for agent
//! step control" IS geometric algebra — every Cyberloop operation has a
//! direct GA equivalent.

use crate::blade::Blade;
use crate::rotor::Rotor;
use crate::tracking::SubspaceTracker;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// An agent's state in cognitive space.
///
/// position: location in the state space
/// momentum: velocity/gradient direction
/// heading: the current subspace orientation (1-blade for direction)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct AgentState {
    #[wasm_bindgen(skip)]
    pub position: Vec<f64>,
    #[wasm_bindgen(skip)]
    pub momentum: Vec<f64>,
    #[wasm_bindgen(skip)]
    pub heading: Blade,
}

#[wasm_bindgen]
impl AgentState {
    #[wasm_bindgen(constructor)]
    pub fn new(position: Vec<f64>, momentum: Vec<f64>) -> AgentState {
        let heading = {
            let norm: f64 = momentum.iter().map(|m| m * m).sum::<f64>().sqrt();
            if norm > 1e-15 {
                Blade::vector(momentum.iter().map(|m| m / norm).collect())
            } else {
                Blade::vector(vec![1.0, 0.0, 0.0, 0.0])
            }
        };
        AgentState { position, momentum, heading }
    }

    pub fn position(&self) -> Vec<f64> {
        self.position.clone()
    }

    pub fn momentum(&self) -> Vec<f64> {
        self.momentum.clone()
    }

    pub fn heading(&self) -> Blade {
        self.heading.clone()
    }
}

/// A trajectory of agent states with the rotors connecting them.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct AgentTrajectory {
    #[wasm_bindgen(skip)]
    pub states: Vec<AgentState>,
    #[wasm_bindgen(skip)]
    pub rotors: Vec<Rotor>,
}

#[wasm_bindgen]
impl AgentTrajectory {
    #[wasm_bindgen(constructor)]
    pub fn new() -> AgentTrajectory {
        AgentTrajectory {
            states: Vec::new(),
            rotors: Vec::new(),
        }
    }

    pub fn add_state(&mut self, state: AgentState) {
        if !self.states.is_empty() {
            let last = &self.states.last().unwrap().heading;
            let rotor = Rotor::from_blades(last, &state.heading);
            self.rotors.push(rotor);
        }
        self.states.push(state);
    }

    pub fn states(&self) -> Vec<AgentState> {
        self.states.clone()
    }

    pub fn rotors(&self) -> Vec<Rotor> {
        self.rotors.clone()
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }
}

impl Default for AgentTrajectory {
    fn default() -> Self {
        Self::new()
    }
}

/// Hamiltonian step: evolve agent state using a potential function.
///
/// This simulates how an agent moves through cognitive space:
/// - Position updates via momentum
/// - Momentum updates via the negative gradient of the potential
/// - Heading updates to match momentum direction
///
/// In Cyberloop terms: this is the agent step function, and the
/// heading changes ARE Grassmannian geodesics (rotors).
pub fn step_hamiltonian(
    state: &AgentState,
    dt: f64,
    potential: &dyn Fn(&Vec<f64>) -> f64,
) -> AgentState {
    let dim = state.position.len();

    // Compute gradient of potential via finite differences
    let eps = 1e-6;
    let mut gradient = vec![0.0; dim];
    for i in 0..dim {
        let mut pos_plus = state.position.clone();
        let mut pos_minus = state.position.clone();
        pos_plus[i] += eps;
        pos_minus[i] -= eps;
        gradient[i] = (potential(&pos_plus) - potential(&pos_minus)) / (2.0 * eps);
    }

    // Symplectic Euler: update momentum first, then position
    let mut new_momentum = state.momentum.clone();
    for i in 0..dim {
        new_momentum[i] -= dt * gradient[i];
    }

    let mut new_position = state.position.clone();
    for i in 0..dim {
        new_position[i] += dt * new_momentum[i];
    }

    // Normalize heading from momentum
    let mom_norm: f64 = new_momentum.iter().map(|m| m * m).sum::<f64>().sqrt();
    let new_heading = if mom_norm > 1e-15 {
        Blade::vector(new_momentum.iter().map(|m| m / mom_norm).collect())
    } else {
        state.heading.clone()
    };

    AgentState {
        position: new_position,
        momentum: new_momentum,
        heading: new_heading,
    }
}

/// Track a full agent trajectory using subspace tracking.
///
/// This creates a SubspaceTracker and feeds it the heading blades from
/// the trajectory, demonstrating that agent state evolution IS a curve
/// on the Grassmannian.
pub fn track_trajectory(trajectory: &AgentTrajectory) -> SubspaceTracker {
    let dim = trajectory.states.first().map(|s| s.position.len()).unwrap_or(4);
    let mut tracker = SubspaceTracker::new(dim);

    for state in &trajectory.states {
        let heading_comps: Vec<f64> = state.heading.components.clone();
        tracker.track(heading_comps);
    }

    tracker
}

/// Run a demo with agents moving in cognitive space.
///
/// Creates 3 agents moving through different potentials and tracks
/// their subspace trajectories. Returns JSON-compatible data for visualization.
#[wasm_bindgen]
pub fn run_agent_demo(steps: usize) -> String {
    let dt = 0.05;
    let dim = 4;

    // Three different potential landscapes (cognitive landscapes)
    let potentials: Vec<Box<dyn Fn(&Vec<f64>) -> f64>> = vec![
        // Agent 0: harmonic oscillator (periodic exploration)
        Box::new(|pos: &Vec<f64>| -> f64 {
            pos.iter().enumerate().map(|(i, x)| 0.5 * (i as f64 + 1.0) * x * x).sum()
        }),
        // Agent 1: double well (bistable decisions)
        Box::new(|pos: &Vec<f64>| -> f64 {
            if pos.is_empty() { return 0.0; }
            let x = pos[0];
            (x * x - 1.0).powi(2) + pos.iter().skip(1).map(|y| 0.5 * y * y).sum::<f64>()
        }),
        // Agent 2: saddle (exploration along ridges)
        Box::new(|pos: &Vec<f64>| -> f64 {
            if pos.len() < 2 { return 0.0; }
            pos[0] * pos[0] - pos[1] * pos[1] + pos.iter().skip(2).map(|y| 0.5 * y * y).sum::<f64>()
        }),
    ];

    // Initial conditions for 3 agents
    let initial_states: Vec<AgentState> = vec![
        AgentState::new(
            vec![1.0, 0.5, 0.3, 0.1],
            vec![0.0, 1.0, 0.5, 0.2],
        ),
        AgentState::new(
            vec![0.5, 1.0, 0.2, 0.3],
            vec![1.0, 0.0, 0.3, 0.1],
        ),
        AgentState::new(
            vec![0.3, 0.2, 1.0, 0.5],
            vec![0.2, 0.3, 0.0, 1.0],
        ),
    ];

    let mut results = Vec::new();

    for (agent_idx, init_state) in initial_states.iter().enumerate() {
        let mut trajectory = AgentTrajectory::new();
        let mut current = init_state.clone();

        trajectory.add_state(current.clone());

        let potential = &*potentials[agent_idx];

        for _ in 0..steps {
            current = step_hamiltonian(&current, dt, potential);
            trajectory.add_state(current.clone());
        }

        let tracker = track_trajectory(&trajectory);

        // Collect results
        let positions: Vec<Vec<f64>> = trajectory.states.iter().map(|s| s.position.clone()).collect();
        let headings: Vec<Vec<f64>> = trajectory.states.iter().map(|s| s.heading.components.clone()).collect();
        let divergences: Vec<f64> = (0..tracker.history_len())
            .map(|i| {
                if i > 0 {
                    crate::blade::subspace_angle(
                        &tracker.history[i],
                        &tracker.history[i - 1]
                    )
                } else {
                    0.0
                }
            })
            .collect();

        results.push(serde_json::json!({
            "agent": agent_idx,
            "positions": positions,
            "headings": headings,
            "divergences": divergences,
            "steps": steps + 1,
        }));
    }

    // Compute inter-agent divergences
    let trackers: Vec<SubspaceTracker> = (0..3).map(|i| {
        let mut t = SubspaceTracker::new(dim);
        // Re-run tracking (simplified)
        for result in &results {
            if result["agent"] == i {
                for heading in result["headings"].as_array().unwrap() {
                    let comps: Vec<f64> = heading.as_array().unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap())
                        .collect();
                    t.track(comps);
                }
            }
        }
        t
    }).collect();

    let inter_divergences: Vec<Vec<f64>> = (0..3).map(|i| {
        (0..3).map(|j| trackers[i].divergence(&trackers[j])).collect()
    }).collect();

    let output = serde_json::json!({
        "agents": results,
        "inter_divergences": inter_divergences,
        "dimension": dim,
        "dt": dt,
        "total_steps": steps,
    });

    serde_json::to_string_pretty(&output).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn harmonic_potential(pos: &Vec<f64>) -> f64 {
        pos.iter().map(|x| 0.5 * x * x).sum()
    }

    #[test]
    fn test_agent_state_creation() {
        let state = AgentState::new(vec![1.0, 2.0], vec![3.0, 4.0]);
        assert_eq!(state.position, vec![1.0, 2.0]);
        assert_eq!(state.momentum, vec![3.0, 4.0]);
        assert_eq!(state.heading.grade, 1);
    }

    #[test]
    fn test_agent_state_heading_normalized() {
        let state = AgentState::new(vec![0.0, 0.0], vec![3.0, 4.0]);
        assert!((state.heading.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_trajectory_new() {
        let t = AgentTrajectory::new();
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn test_trajectory_add_states() {
        let mut t = AgentTrajectory::new();
        t.add_state(AgentState::new(vec![0.0], vec![1.0]));
        t.add_state(AgentState::new(vec![1.0], vec![0.0]));
        assert_eq!(t.len(), 2);
        assert_eq!(t.rotors().len(), 1);
    }

    #[test]
    fn test_hamiltonian_step_stationary() {
        let state = AgentState::new(vec![0.0, 0.0], vec![0.0, 0.0]);
        let next = step_hamiltonian(&state, 0.1, &harmonic_potential);
        // At minimum of harmonic, gradient is zero, should stay put
        assert!(next.position[0].abs() < 1e-10);
    }

    #[test]
    fn test_hamiltonian_step_moves() {
        let state = AgentState::new(vec![1.0, 0.0], vec![0.0, 0.0]);
        let next = step_hamiltonian(&state, 0.1, &harmonic_potential);
        // At x=1 in harmonic, gradient pulls toward origin
        assert!(next.position[0] < 1.0);
    }

    #[test]
    fn test_hamiltonian_conserves_energy() {
        let state = AgentState::new(vec![1.0, 0.0], vec![0.0, 1.0]);
        let initial_energy = 0.5 * state.momentum.iter().map(|m| m * m).sum::<f64>()
            + harmonic_potential(&state.position);
        let next = step_hamiltonian(&state, 0.01, &harmonic_potential);
        let final_energy = 0.5 * next.momentum.iter().map(|m| m * m).sum::<f64>()
            + harmonic_potential(&next.position);
        // Energy should be approximately conserved (small dt)
        assert!((initial_energy - final_energy).abs() < 0.01);
    }

    #[test]
    fn test_track_trajectory() {
        let mut traj = AgentTrajectory::new();
        traj.add_state(AgentState::new(vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]));
        traj.add_state(AgentState::new(vec![0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 0.0]));
        let tracker = track_trajectory(&traj);
        assert_eq!(tracker.history_len(), 2);
    }

    #[test]
    fn test_run_agent_demo() {
        let result = run_agent_demo(5);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["agents"].as_array().unwrap().len(), 3);
        assert!(parsed["inter_divergences"].is_array());
    }

    #[test]
    fn test_hamiltonian_orbit() {
        // In 2D harmonic, should produce elliptical orbits
        let state = AgentState::new(vec![1.0, 0.0], vec![0.0, 1.0]);
        let mut current = state;
        for _ in 0..100 {
            current = step_hamiltonian(&current, 0.01, &harmonic_potential);
        }
        // After one partial orbit, should be somewhere near the initial position
        let dist = (current.position[0] - 1.0).powi(2) + (current.position[1] - 0.0).powi(2);
        assert!(dist < 2.0); // Rough check
    }

    #[test]
    fn test_heading_changes_with_momentum() {
        let state = AgentState::new(vec![1.0, 0.0], vec![1.0, 0.0]);
        let next = step_hamiltonian(&state, 0.1, &harmonic_potential);
        // After gradient step, momentum should shift, heading should follow
        // At x=1, gradient = 1, so momentum decreases in x
        assert!(next.momentum[0] < 1.0);
    }
}
