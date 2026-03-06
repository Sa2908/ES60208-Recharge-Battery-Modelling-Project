# Real-Time State of Charge (SOC) Estimator for LG M50T 21700 Cells

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)
![Accuracy](https://img.shields.io/badge/SOC%20Error-%E2%89%A4%205%25-success)

## Project Overview
This repository contains a robust, discrete **Extended Kalman Filter (EKF)** designed to estimate the State of Charge (SOC) of an LG M50T 21700 Nickel Manganese Cobalt (NMC) lithium-ion cell. 

Standard Coulomb counting accumulates integration drift over time, especially under the highly erratic power demands typical of electric vehicle (EV) accumulators. This estimator solves that drift by utilizing a **1-Resistor-Capacitor (1RC) Equivalent Circuit Model (ECM)** to continuously balance theoretical cell physics against real-world sensor measurements. Validated against aggressive Hybrid Pulse Power Characterization (HPPC) load profiles, the algorithm successfully maintains an SOC tracking error of $\le 5.0\%$.

## 🔬 Methodology

### 1. OCV-SOC Characterization
The Open Circuit Voltage (OCV) vs. SOC relationship was extracted from a Begining-of-Life (BOL) $C/10$ pseudo-OCV discharge test at $25^\circ\text{C}$. This relationship is heavily non-linear and is dynamically linearized (via Jacobian matrices) at each time step during the EKF update phase.

### 2. Parameter Extraction (1RC ECM)
The transient thermodynamic parameters—Ohmic Resistance ($R_0$), Polarization Resistance ($R_1$), and Capacitance ($C_1$)—were rigorously identified using a bounded non-linear least-squares optimization (`scipy.optimize.curve_fit`). The model was fitted to the voltage relaxation phase of a $0.5\text{C}$ Hybrid CC-Pulse test.

### 3. State Estimation (Extended Kalman Filter)
The EKF predicts the internal polarization voltage and SOC at each time step $\Delta t$, and then corrects those predictions using the measured terminal voltage. 
* **State Vector:** $x = [\text{SOC}, V_1]^T$
* **Prediction:** Integrates current and decays $V_1$ based on the time constant $\tau = R_1 \times C_1$.
* **Update:** Calculates the Kalman Gain to minimize the residual between the measured voltage and the model's predicted voltage.

## Repository Structure
```text
ev_soc_estimator/
│
│── estimator.py                           # Run this file after installing requirements.txt for visualization of SOC Estimation
│── cell_D_RPT1_Hybrid_CC-Pulse_0.5C.csv   # Dynamic validation data
│── ocv_soc_lookup.csv                     # Extracted OCV-SOC mapping
│── soc_estimator.ipynb
├── README.md
└── requirements.txt
```

## Authors

* **[Sanket Agarwal - 23ME30051]** - *Parameter Extraction, OCV Modeling & System Integration*
* **[Gokul R - 23ME10029]** - *EKF Implementation & Statistical Validation* 

---
*Developed as part of the Battery Modeling Project for the course ES60208, Energy Science Department, IIT Kharagpur.*
