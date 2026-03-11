## Surrogate Model Predictive Control (MPC)

This repository provides an implementation of a **Surrogate Model Predictive Control (MPC)** strategy for district HVAC energy management.
The controller optimizes the operation of key energy assets, including:
* thermal energy storage (TES)
* photovoltaic generation (PV)
* battery energy storage systems (BESS)
* cooling production units
Unlike conventional MPC approaches that rely on high-fidelity physical models, this implementation adopts a **surrogate thermal model** to approximate the system dynamics. The surrogate model captures the dominant thermal behavior of the district energy system while significantly reducing computational complexity.
The surrogate representation enables real-time optimization while maintaining sufficient accuracy for supervisory control.

### Surrogate Model
The simplified thermal dynamics of the storage system are modeled using an energy balance:

[
C_{TES}\frac{dT_{TES}}{dt} = Q_{HP} - Q_{load} - Q_{loss}
]

where:
* (C_{TES}) represents the effective thermal capacity of the storage tank
* (Q_{HP}) is the cooling power provided by the heat pump
* (Q_{load}) is the building cooling demand
* (Q_{loss}) represents thermal losses to the environment
This grey-box formulation approximates the behavior of the detailed physical model while enabling faster optimization.

### Optimization Objective

At each control step, the MPC solves a finite-horizon optimization problem that minimizes operational cost while maintaining system constraints:

* electricity cost minimization
* TES temperature safety limits
* battery state-of-charge constraints
* smooth control actions
The optimization is solved in a **receding horizon framework**, where only the first control action is implemented before the next optimization step.

### Surrogate MPC Purpose

The surrogate MPC is used in this project to:

* provide a **physics-based benchmark controller**
* enable **fast optimization within the co-simulation environment**
* allow fair comparison with learning-based approaches (DRL, OTL, BC, IRL)

### Note

The MPC implementation shared in this repository is a **surrogate version of the controller used in the research study**. It reproduces the main optimization logic while using simplified models to facilitate reproducibility and experimentation.

<img width="1002" height="455" alt="image" src="https://github.com/user-attachments/assets/2bd30370-53af-404c-a19a-5112ff989d4e" />
