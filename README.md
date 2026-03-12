# Surrogate Model Predictive Control (MPC) for HVAC in District Energy Systems

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Method](https://img.shields.io/badge/Method-Model%20Predictive%20Control-green?style=flat-square)
![Domain](https://img.shields.io/badge/Domain-HVAC%20Control-orange?style=flat-square)
![Type](https://img.shields.io/badge/Model-Surrogate%20Grey--Box-lightgrey?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

*Physics-based benchmark controller for district HVAC energy optimization using receding horizon optimization*

</div>

---

## Overview

This repository implements a **Surrogate Model Predictive Control (MPC)** strategy for supervisory energy management in district HVAC systems.

Unlike conventional MPC that relies on high-fidelity physical simulations, this implementation uses a **grey-box surrogate thermal model** to approximate system dynamics — dramatically reducing computational complexity while preserving sufficient accuracy for real-time supervisory control.

The controller optimizes the coordinated operation of:

- 🌡️ **Thermal Energy Storage (TES)**
- ☀️ **Photovoltaic (PV) generation**
- 🔋 **Battery Energy Storage Systems (BESS)**
- ❄️ **Cooling production units (Heat Pumps)**

The surrogate MPC serves as a **physics-based benchmark** against which data-driven controllers (DRL, BC, IRL) are evaluated in the broader research project.

<img width="1002" height="455" alt="image" src="https://github.com/user-attachments/assets/af7f864f-6821-4f53-b287-8ef18d9e8637" />


> ⚠️ **Note:** This is a **surrogate version** of the MPC controller used in the research study. It reproduces the main optimization logic using simplified models to facilitate reproducibility and experimentation.

---

## Key Features

| Feature | Description |
|---|---|
| ⚡ **Receding Horizon** | Optimizes over a finite prediction horizon, applies first action only |
| 🔢 **Surrogate Model** | Grey-box energy balance replaces expensive physical simulation |
| 💶 **Cost Minimization** | Minimizes electricity cost using time-of-use price signals |
| 🌞 **PV-Aware** | Incorporates PV generation forecast into optimization |
| 🔋 **Multi-Asset** | Jointly optimizes TES, BESS, and heat pump operation |
| 📐 **Constraint Handling** | Enforces TES temperature limits and BESS SoC bounds |

---

## Surrogate Thermal Model

Instead of a full physical simulation, the TES dynamics are approximated by a **grey-box energy balance**:

```
C_TES · dT_TES/dt = Q_HP - Q_load - Q_loss
```

| Symbol | Description |
|---|---|
| `C_TES` | Effective thermal capacity of the storage tank |
| `Q_HP` | Cooling power delivered by the heat pump |
| `Q_load` | Building cooling demand |
| `Q_loss` | Thermal losses to the environment |

This formulation captures the dominant thermal behavior while enabling **fast optimization** suitable for real-time control.

---

## Optimization Problem

At each control step, the MPC solves a **finite-horizon optimization**:

```
minimize    Σ [ p(t) · P_grid(t) ]          (electricity cost)

subject to  T_min ≤ T_TES(t) ≤ T_max       (TES temperature bounds)
            SoC_min ≤ SoC(t) ≤ SoC_max     (battery state of charge)
            0 ≤ P_HP(t) ≤ P_HP_max          (heat pump power limits)
            dynamics constraints             (surrogate thermal model)
```

The **receding horizon** framework re-solves this problem at each timestep using updated measurements and forecasts — only the first optimal action is applied before the next solve.

```
t=0    t=1    t=2    t=3  ...  t=N
 │──────│──────│──────│────────│   ← Prediction horizon
 ▼
Apply only u*(t=0), then re-optimize at t=1
```

---

## Project Structure

```
HVAC-MPC-Python/
│
├── mpc_controller.py       # Main MPC implementation
│                           #   - Surrogate thermal model
│                           #   - Receding horizon optimizer
│                           #   - Constraint enforcement
│                           #   - PV & price forecast integration
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/shadmehr1993/HVAC-MPC-Python.git
cd HVAC-MPC-Python

# Create a virtual environment
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows

# Install dependencies
pip install numpy scipy cvxpy pandas matplotlib
```

---

## Usage

```python
from mpc_controller import MPCController

# Initialize the MPC controller
mpc = MPCController(
    prediction_horizon=24,    # hours
    control_step=1,           # hours
    tes_capacity=500,         # kWh
    bess_capacity=200,        # kWh
)

# At each timestep, call:
action = mpc.solve(
    current_state=state,
    price_forecast=prices,
    pv_forecast=pv_generation,
    demand_forecast=cooling_demand,
)
```

---

## MPC vs. Learning-Based Controllers

This repository is part of a broader research project comparing multiple control strategies:

| Controller | Type | Requires Training | Online Optimization | This Repo |
|---|---|---|---|---|
| **MPC** (this repo) | Model-based | ❌ | ✅ Receding horizon | ✅ |
| [Behavioral Cloning](https://github.com/shadmehr1993/Behavioral-Cloning) | Imitation Learning | ✅ Offline | ❌ | — |
| [IRL](https://github.com/shadmehr1993/inverse-reinforcement-learning-IRL-) | Inverse RL | ✅ Offline | ❌ | — |
| [Transfer Learning DRL](https://github.com/shadmehr1993/TL-hpc-cooling) | Deep RL | ✅ Offline | ❌ | — |

**MPC advantages:** no training data required, interpretable, constraint-guaranteed
**MPC limitations:** requires system model, computational cost scales with horizon length

---

## Relation to Other Work

This MPC controller serves as the **physics-based benchmark** in a comprehensive comparison of HVAC control strategies:

```
Rule-Based Control (RBC)         → simplest baseline
       ↓
Surrogate MPC (this repo)        → physics-based benchmark
       ↓
Behavioral Cloning / IRL         → imitation learning
       ↓
Transfer Learning DRL            → most advanced, best performance
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{zaregarizi2024mpc-hvac,
  author    = {Shadmehr Zaregarizi},
  title     = {Surrogate Model Predictive Control for HVAC in District Energy Systems},
  year      = {2024},
  publisher = {GitHub},
  url       = {https://github.com/shadmehr1993/HVAC-MPC-Python}
}
```

---

## Author

**Shadmehr Zaregarizi**
Politecnico di Torino
📧 shadmehr.zaregarizi@studenti.polito.it
🔗 [github.com/shadmehr1993](https://github.com/shadmehr1993)

---

<div align="center">
⭐ If you find this project useful, please consider giving it a star!
</div>
