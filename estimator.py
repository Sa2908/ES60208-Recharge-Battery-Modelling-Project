"""
Real-Time State of Charge (SOC) Estimator 
Cell Type: LG M50T 21700 NMC Lithium-Ion
Methodology: 1RC Equivalent Circuit Model (ECM) with Discrete Extended Kalman Filter (EKF)
"""

# ==========================================
# PHASE 0: ENVIRONMENT SETUP & IMPORTS
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

print("Initializing EV SOC Estimator...")

# ==========================================
# PHASE 1: OCV-SOC MAPPING & PLOTTING
# ==========================================
print("\n--- Phase 1: OCV-SOC Characterization ---")
# 1. Load the C/10 discharge data (Reference Performance Test 0)
df_rpt0 = pd.read_csv('./cell_D_RPT0.csv')
Q_total = df_rpt0['Charge (mA.h)'].max() 
print(f"Total measured capacity: {Q_total:.2f} mAh")

# 2. Calculate State of Charge (SOC) via Coulomb Counting
df_rpt0['SOC'] = 1.0 - (df_rpt0['Charge (mA.h)'] / Q_total)

# 3. Create the Parametric OCV-SOC Lookup Model
soc_array = df_rpt0['SOC'].values[::-1]
ocv_array = df_rpt0['Voltage (V)'].values[::-1]

# Linear interpolation function (Callable mapping model)
ocv_model = interp1d(soc_array, ocv_array, kind='linear', fill_value="extrapolate")

# 4. Save the Reference Lookup Table
df_export = pd.DataFrame({'SOC': soc_array, 'OCV': ocv_array})
df_export.to_csv('ocv_soc_lookup.csv', index=False)

# 5. Generate Technical Report Graphic: OCV-SOC Curve
plt.figure(figsize=(8, 5))
plt.plot(df_export['SOC'] * 100, df_export['OCV'], color='darkblue', linewidth=2.5, label='Measured C/10 Pseudo-OCV')
plt.title('Open Circuit Voltage (OCV) vs. State of Charge (SOC)\nLG M50T 21700 Cell at BoL (25°C)', fontsize=12, fontweight='bold')
plt.xlabel('State of Charge (%)', fontsize=11)
plt.ylabel('Open Circuit Voltage (V)', fontsize=11)
plt.xlim(0, 100)
plt.ylim(2.4, 4.3)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right', fontsize=10)
plt.savefig('OCV_SOC_Curve.png', dpi=300, bbox_inches='tight')
print("Saved 'OCV_SOC_Curve.png'.")
# plt.show() # Commented out so the script runs continuously without pausing

# ==========================================
# PHASE 2: EXTRACTING R0, R1, and C1
# ==========================================
print("\n--- Phase 2: Parameter Extraction (Transient Dynamics) ---")
# 1. Load the 0.5C Pulse-Under-Load (PUL) Data
df_pul = pd.read_csv('cell D_RPT1_Hybrid CC-Pulse 0.5C discharge data.csv')
df_pul['Time (s)'] = df_pul['Time (s)'] - df_pul['Time (s)'].iloc[0] # Zero the time axis

# 2. Automatically Detect the Pulse Edge
df_pul['dI'] = df_pul['Current (mA)'].diff()
pulse_start_idx = df_pul[df_pul['dI'] < -1000].index[0]

# 3. Calculate Ohmic Resistance (R0)
I_before = df_pul['Current (mA)'].iloc[pulse_start_idx - 1] / 1000.0
I_instant = df_pul['Current (mA)'].iloc[pulse_start_idx + 1] / 1000.0
delta_V = df_pul['Voltage (V)'].iloc[pulse_start_idx - 1] - df_pul['Voltage (V)'].iloc[pulse_start_idx + 1]
delta_I = abs(I_instant - I_before)
R0 = delta_V / delta_I

# 4. Extract R1 and C1 via Bounded Curve Fitting
pulse_end_idx = df_pul[(df_pul.index > pulse_start_idx) & (df_pul['dI'] > 1000)].index[0]
relax_data = df_pul.iloc[pulse_end_idx : pulse_end_idx + 600] 
t_relax = relax_data['Time (s)'].values - relax_data['Time (s)'].values[0]
V_relax = relax_data['Voltage (V)'].values

def relaxation_curve(t, v_steady, v_drop, tau):
    return v_steady - v_drop * np.exp(-t / tau)

# Dynamic Guesses & Bounds
v_steady_guess = V_relax[-1]                 
v_drop_guess = V_relax[-1] - V_relax[0]      
tau_guess = 10.0                             
lower_bounds = [v_steady_guess - 0.05, 0.0, 1.0]
upper_bounds = [v_steady_guess + 0.05, 0.5, 100.0]

popt, _ = curve_fit(relaxation_curve, t_relax, V_relax, p0=[v_steady_guess, v_drop_guess, tau_guess], bounds=(lower_bounds, upper_bounds))
v_steady_fit, V_drop_R1_fit, tau_fit = popt

R1 = V_drop_R1_fit / delta_I
C1 = tau_fit / R1

print(f"R0 (Ohmic):       {R0:.5f} Ohms")
print(f"R1 (Polarization):{R1:.5f} Ohms")
print(f"C1 (Capacitance): {C1:.2f} Farads")
print(f"Tau (Time Const): {tau_fit:.2f} seconds")

# 5. Generate Technical Report Graphic: RC Curve Fit
plt.figure(figsize=(10, 4))
plt.plot(t_relax, V_relax, 'b-', label='Measured Relaxation')
plt.plot(t_relax, relaxation_curve(t_relax, *popt), 'r--', linewidth=2, label='1RC Model Fit')
plt.title('Voltage Relaxation Fit for R1 and C1')
plt.xlabel('Time since pulse ended (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True, alpha=0.5)
plt.savefig('Voltage_Relaxation_Fit.png', dpi=300, bbox_inches='tight')
print("Saved 'Voltage_Relaxation_Fit.png'.")

# ==========================================
# PHASE 3: EKF LOOP & RESULTS PLOTTING
# ==========================================
print("\n--- Phase 3: EKF Loop Execution ---")
# 1. Define Filter Inputs
time_seconds = df_pul['Time (s)'].values
current_amps = df_pul['Current (mA)'].values / 1000.0
voltage_measured = df_pul['Voltage (V)'].values
Q_total_As = 4.846 * 3600 # Nominal capacity in Amp-seconds
tau = R1 * C1

# 2. Initialize Matrices
x_est = np.array([[1.0], [0.0]])  # State: [SOC, V1]
P = np.array([[1e-4, 0], [0, 1e-4]]) # Initial Covariance
Q_noise = np.array([[1e-6, 0], [0, 1e-5]]) # Process Noise
R_noise = np.array([[1e-3]]) # Measurement Noise

soc_estimates = []

# 3. The EKF Loop
for k in range(1, len(time_seconds)):
    dt = time_seconds[k] - time_seconds[k-1]
    I_k = abs(current_amps[k-1]) 
    
    # --- PREDICTION STEP ---
    A = np.array([[1, 0], [0, np.exp(-dt / tau)]])
    B = np.array([[-dt / Q_total_As], [R1 * (1 - np.exp(-dt / tau))]])
    
    x_pred = A @ x_est + B * I_k
    P_pred = A @ P @ A.T + Q_noise
    
    soc_pred_clipped = np.clip(x_pred[0, 0], 0.001, 0.999)
    v1_pred = x_pred[1, 0]
    V_pred = ocv_model(soc_pred_clipped) - v1_pred - (I_k * R0)
    
    # --- UPDATE STEP ---
    delta_soc = 1e-4
    dOCV_dSOC = (ocv_model(soc_pred_clipped + delta_soc) - ocv_model(soc_pred_clipped - delta_soc)) / (2 * delta_soc)
    C_mat = np.array([[dOCV_dSOC, -1]])
    
    S = C_mat @ P_pred @ C_mat.T + R_noise
    K = P_pred @ C_mat.T @ np.linalg.inv(S) # Kalman Gain
    y_tilde = voltage_measured[k] - V_pred  # Residual
    
    x_est = x_pred + K * y_tilde
    P = (np.eye(2) - K @ C_mat) @ P_pred
    
    soc_estimates.append(x_est[0, 0])

# 4. Coulomb Counting Baseline
dt_array = np.diff(time_seconds, prepend=time_seconds[0])
charge_removed_As = np.cumsum(abs(current_amps) * dt_array)
true_soc = 1.0 - (charge_removed_As / Q_total_As)

ekf_soc_array = np.array(soc_estimates)
true_soc_aligned = true_soc[1:] 
time_aligned = time_seconds[1:]

# 5. Evaluate Metrics
soc_error_percent = (ekf_soc_array - true_soc_aligned) * 100
rmse_soc = np.sqrt(np.mean(soc_error_percent**2))
mae_soc = np.mean(np.abs(soc_error_percent))

print(f"\n--- EKF VALIDATION RESULTS ---")
print(f"Target Accuracy: <= 5.00% Error")
print(f"Achieved RMSE:   {rmse_soc:.2f}%")
print(f"Achieved MAE:    {mae_soc:.2f}%")

# 6. Generate Technical Report Graphic: EKF Tracking Performance
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

ax1.plot(time_aligned / 3600, true_soc_aligned * 100, 'k-', linewidth=2, label='True SOC (Coulomb Counting)')
ax1.plot(time_aligned / 3600, ekf_soc_array * 100, 'r--', linewidth=2, label='EKF Estimated SOC')
ax1.set_title('Real-Time SOC Estimation tracking 0.5C Pulse Data', fontsize=14, fontweight='bold')
ax1.set_ylabel('State of Charge (%)', fontsize=12)
ax1.set_ylim(0, 105)
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.plot(time_aligned / 3600, soc_error_percent, 'b-', linewidth=1.5, label='Estimation Error')
ax2.axhline(y=5, color='r', linestyle=':', linewidth=1.5, label='±5% Error Bound')
ax2.axhline(y=-5, color='r', linestyle=':', linewidth=1.5)
ax2.set_xlabel('Time (Hours)', fontsize=12)
ax2.set_ylabel('SOC Error (%)', fontsize=12)
ax2.set_ylim(-8, 8)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('SOC_Estimation_Accuracy.png', dpi=300, bbox_inches='tight')
print("Saved 'SOC_Estimation_Accuracy.png'.")

# Optional: display all figures at the very end
plt.show()