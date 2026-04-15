# Cooperative UAV Path Planning: GTCPSO with HDPLO & Dynamic RHC

## Overview

Implementation of **Cooperative Path Planning of Multiple UAVs** using **Cylinder Vector Particle Swarm Optimization with Gene Targeting (GTCPSO)**, enhanced with **Hybrid Direction-Prediction Learning (HDPLO)** and extended for real-time mission execution via **Receding Horizon Control (RHC)**.

This repository compares and integrates:
1. **Standard PSO** — Baseline in Cartesian coordinates.
2. **GTCPSO** — Cylindrical-coordinate PSO with Gene Targeting (IEEE Reference).
3. **GTCPSO+HDPLO** — Enhanced with Adaptive Crossover, CMA-ES Refinement, and Search Guidance.
4. **D-GTCPSO (Dynamic)** — Asynchronous online re-planning using Warm-Started RHC.

---

## Problem Description

The environment consists of multiple UAVs navigating a 3D space with:
- **Terrain** (Hills, Mountains, Rugged)
- **Cylindrical Threat Zones** (Radar/SAM sites with Piecewise Intensity)
- **No-Fly Zones** (Static forbidden volumes)
- **Multi-UAV Safety** (Collision avoidance and cooperative clearance)

### Multi-Objective Cost Function
The mission objective is to minimize a weighted total cost:
`J = w₁·L + w₂·T + w₃·S + w₄·A + w₅·F + P_constraints`

| Component | Description | Weight |
|-----------|-------------|--------|
| **L** | Path length (normalized) | 1.0 |
| **T** | **Threat Exposure (Piecewise Model)** | 5.0 |
| **S** | Smoothness (Turning angle penalty) | 2.0 |
| **A** | Altitude variation | 0.5 |
| **F** | Fuel consumption | 0.3 |
| **P_coll** | Inter-UAV collision penalty | 200.0 |
| **P_const** | Physical constraint violations (Climb/Turn) | 50.0 |

---

## Algorithms

### 1. Standard PSO
Baseline swarm optimizer operating in Cartesian space with linear inertia weight decay.

### 2. GTCPSO (Cylindrical with Gene Targeting)
Optimizes paths in cylindrical space (ρ, φ, z). Identifies "bottleneck" waypoints (gene targeting) and applies localized exploitation to fix the most costly mission segments.

### 3. GTCPSO+HDPLO (Advanced Static Optimization)
Adds four major improvements:
- **Adaptive Crossover**: Regenerates weak particles based on swarm diversity.
- **CMA-ES Refinement**: Applies evolution strategies to optimize bottleneck windows.
- **Direction-Guided Search**: Incorporates historical global-best momentum into velocity.
- **Diversity Control**: Prevents population collapse via selective re-initialization.

### 4. D-GTCPSO (Dynamic Receding Horizon Control)
Transforms static optimization into a continuous control loop:
- **Warm-Start Mechanism**: Every re-optimization cycle ($dt$) initializes the swarm with the previous best path (time-shifted), enabling near-instant convergence for dynamic events.
- **Asynchronous Handling**: Real-time integration of `UAV_ARRIVE` and `THREAT_ARRIVE` events mid-mission.

---

## Results and Performance

The current results reflect the **Piecewise Threat Model** (Equation 5) implemented in the core environment, which provides more aggressive threat avoidance compared to simple distance-decay models.

### Static Benchmarking (Cost Comparison)
*Results obtained using `experiments/run_fast.py` (Minimal Swarm Parameters).*

| Scenario | UAVs | PSO Cost | GTCPSO Cost | HDPLO Cost | Improvement (vs PSO) |
|----------|------|----------|-------------|------------|----------------------|
| **Simple** | 2 | 322.37 | 307.78 | 247.71 | **-23.2%** |
| **Moderate**| 3 | 1314.35| 605.70 | 847.82 | **-35.5%** |
| **Complex** | 5 | 3871.19| 3045.63| 2021.95| **-47.8%** |
| **Extreme** | 5 | 5510.27| 3357.07| 2570.47| **-53.4%** |

### Dynamic Verification
In dynamic missions (`demo_dynamic_run.py`), the system has been verified to:
- Detect and evade **popup threats** appearing directly in the trajectory.
- Safely integrate **late-arriving UAVs** by de-conflicting existing paths in real-time.
- Maintain **physical feasibility** (smoothness and climb constraints) during high-maneuver re-planning.

---

## Project Structure

```
├── src/
│   ├── rhc_manager.py        # NEW: Receding Horizon Control (Dynamic Logic)
│   ├── gtcpso_hdplo.py       # Enhanced GTCPSO with HDPLO improvements
│   ├── gtcpso.py             # Cylindrical PSO with Gene Targeting
│   ├── cost_function.py      # Piecewise Multi-objective cost
│   ├── environment.py        # 3D terrain and Piecewise Threat logic
│   ├── uav.py                # UAV dynamic constraints
│   ├── utils.py              # Coordinate transforms and RNG helpers
│   ├── pso_base.py           # Standard PSO baseline
│   └── visualization.py      # Interactive Plotly dashboards
├── datasets/                 # Mission scenarios and generators
├── experiments/
│   ├── run_fast.py           # Static algorithm benchmark
│   ├── demo_dynamic_run.py   # Dynamic RHC showcase (popup threats)
│   └── demo_deviation_comparison.py # Study on static vs dynamic deviations
├── results/                  # HTML Interactive Results
├── README.md
└── requirements.txt
```

---

## Visualizations

The project generates full interactive dashboards in `results/`:
- **3D Path Scene**: Holistic view of mission execution.
- **Convergence Curves**: Cost vs. Iteration progress.
- **Bottleneck Analysis**: Identifies critical waypoint stressors.
- **Dynamic Playback**: Visualizes the actual trajectory followed during RHC execution.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Static Benchmark
```bash
python experiments/run_fast.py
```

### 3. Run Dynamic Simulation
```bash
python experiments/demo_dynamic_run.py
```

---

## References
1. **IEEE Paper**: Cooperative Path Planning of Multiple Unmanned Aerial Vehicles Using Cylinder Vector Particle Swarm Optimization With Gene Targeting.
2. **HDPLO Work**: Hybrid Direction-Prediction Learning Optimizer with adaptive crossover, CMA-ES refinement, and diversity control mechanisms

