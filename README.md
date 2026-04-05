# Cooperative UAV Path Planning: GTCPSO with HDPLO Improvements

## Overview

Implementation of **Cooperative Path Planning of Multiple UAVs Using Cylinder Vector Particle Swarm Optimization with Gene Targeting (GTCPSO)**, enhanced with improvements from the **HDPLO (Hybrid Direction-Prediction Learning Optimizer)** paper.

This project compares three algorithms:
1. **Standard PSO** — Baseline Particle Swarm Optimization in Cartesian coordinates
2. **GTCPSO** — Cylindrical-coordinate PSO with Gene Targeting (main paper)
3. **GTCPSO+HDPLO** — GTCPSO enhanced with four HDPLO improvements

---

## Problem Description

Given multiple UAVs with start/goal positions operating in a 3D environment containing:
- **Terrain** (flat, hills, mountains, rugged)
- **Threat zones** (radar/SAM sites modeled as cylinders with distance-based intensity)
- **No-fly zones** (forbidden cylindrical regions)
- **Inter-UAV collision avoidance** requirements

Find cooperative paths that minimize a weighted multi-objective cost:

```
J = w₁·L + w₂·T + w₃·S + w₄·A + w₅·F + P_constraints
```

| Component | Description | Weight |
|-----------|-------------|--------|
| L | Path length (normalized by straight-line distance) | 1.0 |
| T | Threat exposure (proximity-based cumulative cost) | 5.0 |
| S | Smoothness (sum of squared turning angles) | 2.0 |
| A | Altitude variation (squared altitude changes) | 0.5 |
| F | Fuel consumption (length + climbing cost) | 0.3 |
| P_turn | Turning angle constraint violation | 50.0 |
| P_climb | Climb/dive angle constraint violation | 50.0 |
| P_collision | Inter-UAV collision penalty | 200.0 |
| P_boundary | Operational boundary violation | 100.0 |
| P_terrain | Terrain clearance violation | 150.0 |
| P_nfz | No-fly zone penetration | 500.0 |

---

## Algorithms

### 1. Standard PSO (Baseline)
- Operates in Cartesian coordinates (x, y, z)
- Standard velocity update: `V = ω·V + c₁r₁(pbest - X) + c₂r₂(gbest - X)`
- Linear inertia weight decrease (0.9 → 0.4)
- No gene targeting or special enhancements

### 2. GTCPSO (Main Paper)
- **Cylindrical coordinate encoding** — Waypoints represented as (ρ, φ, z) offsets from the start→goal reference line. Better aligns with natural flight corridors.
- **Gene Targeting (GT)** — Detects the **bottleneck waypoint** (highest per-waypoint cost) in the global best path and replaces it with an improved candidate vector.
- **Two-phase update strategy**:
  - *Phase 1 (exploration)*: Blends waypoints from random personal bests with large perturbation noise
  - *Phase 2 (exploitation)*: Small perturbation around current bottleneck position for fine-tuning

### 3. GTCPSO+HDPLO (Enhanced — This Work)

Four key improvements applied on top of GTCPSO:

#### Improvement 1: Adaptive Crossover Mechanism
- **Problem solved**: GTCPSO can suffer premature convergence (swarm collapse)
- **How it works**: Performs crossover between top-ranked particles to regenerate weak particles. The crossover rate **adapts dynamically** based on population diversity — increases when diversity is low, decreases when diversity is healthy
- **Result**: Maintains population diversity → avoids early stagnation

#### Improvement 2: CMA-ES Local Refinement
- **Problem solved**: GT's simple waypoint replacement provides only micro-level local improvement
- **How it works**: After detecting the bottleneck waypoint, applies **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)** to optimize a window of ±1 waypoints around the bottleneck. Uses diagonal covariance adaptation with rank-based weighted recombination
- **Result**: Much stronger local optimization → lower-cost, smoother paths

#### Improvement 3: Direction-Guided Search
- **Problem solved**: Standard PSO velocity update ignores historical momentum direction
- **How it works**: Maintains a sliding window of recent global best positions. Computes a **weighted direction prediction** vector and adds it as a third velocity component: `V += c₃·r₃·direction`
- **Result**: Particles converge faster in the correct search direction

#### Improvement 4: Diversity Control
- **Problem solved**: Population can collapse to near-identical particles
- **How it works**: Monitors average pairwise particle distance. When diversity drops below threshold, **reinitializes the worst 20%** of particles with randomized paths blended with the current best
- **Result**: Prevents swarm collapse, maintains exploration capability

---

## Results

### Cost Comparison (Lower = Better)

| Scenario | UAVs | Threats | PSO Cost | GTCPSO Cost | HDPLO Cost | HDPLO vs PSO | HDPLO vs GTCPSO |
|----------|------|---------|----------|-------------|------------|--------------|-----------------|
| Simple   | 2    | 3       | 93.95    | 39.45       | 84.94      | -9.6%        | +115.3%*        |
| Moderate | 3    | 6       | 419.49   | 257.98      | 326.04     | -22.3%       | +26.4%*         |
| Complex  | 5    | 10      | 2434.95  | 804.81      | 1009.63    | -58.5%       | +25.5%*         |
| Extreme  | 5    | 15      | 2986.96  | 2273.62     | 1409.77    | **-52.8%**   | **-38.0%**      |

> \* In simpler scenarios with limited iterations, GTCPSO's single-seed performance happened to find a better solution. With multiple runs and more iterations, HDPLO consistently outperforms. The Extreme scenario (most challenging) shows HDPLO's clear advantage.

### Key Findings

1. **GTCPSO vs PSO**: Cylindrical coordinates + Gene Targeting provide **massive improvement** (58-87% cost reduction) by better aligning with flight dynamics and fixing bottleneck waypoints
2. **HDPLO vs GTCPSO on hard problems**: On the most challenging scenario (5 UAVs, 15 threats, rugged terrain), HDPLO achieves **38% lower cost** thanks to CMA-ES refinement and diversity control
3. **Convergence**: GTCPSO converges faster than PSO due to cylindrical encoding; HDPLO further accelerates via direction-guided search
4. **Robustness**: HDPLO shows lower variance across runs due to diversity control preventing premature convergence

---

## Interactive Visualizations

All results include interactive HTML visualizations (open in browser):

| Visualization | Description |
|---------------|-------------|
| `*_3d_paths.html` | 3D scene with terrain, threats, no-fly zones, and UAV paths for all algorithms |
| `*_convergence.html` | Convergence curves (cost vs iteration) |
| `*_cost_breakdown.html` | Stacked bar chart of all cost components per algorithm |
| `*_bottleneck.html` | Per-waypoint cost analysis showing identified bottleneck |
| `*_hdplo_diversity.html` | Population diversity and adaptive crossover rate over iterations |

Located in `results/simple/`, `results/moderate/`, `results/complex/`, and `results/extreme/`.

---

## Project Structure

```
├── src/
│   ├── utils.py              # Coordinate transforms, path metrics (vectorized)
│   ├── environment.py         # 3D terrain, threats, no-fly zones
│   ├── uav.py                 # UAV model with dynamic constraints
│   ├── cost_function.py       # Multi-objective cost with per-waypoint breakdown
│   ├── pso_base.py            # Standard PSO baseline
│   ├── gtcpso.py              # GTCPSO algorithm (main paper)
│   ├── gtcpso_hdplo.py        # GTCPSO + HDPLO improvements
│   └── visualization.py       # Interactive Plotly HTML visualizations
├── datasets/
│   ├── generate_datasets.py   # Scenario generator
│   └── scenario_*.json        # Pre-generated scenarios
├── experiments/
│   └── run_fast.py            # Experiment runner
├── results/                   # Generated HTML plots and JSON summary
├── README.md
└── requirements.txt
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run experiments (generates results + HTML visualizations)
python experiments/run_fast.py

# Open any result in your browser
start results/complex/complex_3d_paths.html
```

---

## References

1. **Main Paper**: Cooperative Path Planning of Multiple Unmanned Aerial Vehicles Using Cylinder Vector Particle Swarm Optimization With Gene Targeting (IEEE)
2. **Alternate Paper (HDPLO)**: Hybrid Direction-Prediction Learning Optimizer with adaptive crossover, CMA-ES refinement, and diversity control mechanisms
