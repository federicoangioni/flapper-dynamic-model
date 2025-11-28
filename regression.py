import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.interpolate import UnivariateSpline

from utils import config_old

WORKING_DIR = Path.cwd()
DATA_DIR = WORKING_DIR / 'data'
save_fig = False
g0 = 9.80665


def compute_thrust(coeffs, f):
    """T = m*f + c"""
    return coeffs[0]*f + coeffs[1]

def compute_angle_attack(dihedral, yaw_servo_angle, wing="R"):

    alpha = 0
    lw = config_old.FLAPPER_DIMS["lw"]
    ly = config_old.FLAPPER_DIMS["ly"]
    lk = config_old.FLAPPER_DIMS["lk"]

    if wing == "R":
        alpha = np.arcsin((- lw * np.sin(dihedral) - ly * np.sin(yaw_servo_angle)) / lk)
    elif wing == "L":
        alpha = np.arcsin((- lw * np.sin(dihedral) + ly * np.sin(yaw_servo_angle)) / lk)
    else:
        raise ValueError('Only two wings, choose "R" or "L"')

    return alpha


def regression_pwm_frequency():
    columns_to_keep = ['optitrack.freq.left', 'onboard.motor.m2', 'optitrack.freq.right', 'onboard.motor.m4']

    print("======================================================================")
    print("                                                                      ")
    print("Regression for pwm to f relationship")
    print("Using a linear model for the thrust; f = c1 * pwm + c2")

    dfs = []

    for dir_path in DATA_DIR.iterdir():
        if dir_path.is_dir():
            for file_path in dir_path.glob("*_processed.csv"):
                print(f"Found dataset: {file_path.name}")
                # Find files ending with _processed.csv
                df = pd.read_csv(file_path, usecols=columns_to_keep)
                dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    mask = (combined_df['onboard.motor.m2'] > 1e4) & (combined_df['onboard.motor.m4'] > 1e4)

    masked_df = combined_df.loc[mask, :]

    motor_inputs = pd.concat([masked_df['onboard.motor.m2'], masked_df['onboard.motor.m4']], axis=0).to_numpy()

    frequency = pd.concat([masked_df['optitrack.freq.left'], masked_df['optitrack.freq.right']], axis=0)

    A = np.vstack([motor_inputs, np.ones(len(motor_inputs))]).T

    b = frequency

    coeffs, *_ = np.linalg.lstsq(A, b)

    b_pred = A @ coeffs
    r2 = r2_score(b, b_pred)

    print(f"The R^2 regression score for the regression from pwm to f is {r2:.4f}")    
    print(f"Linear regression parameters are: c1 = {coeffs[0]:.6f}, c2 = {coeffs[1]:.6f}")

    plt.scatter(masked_df['onboard.motor.m2'], masked_df['optitrack.freq.left'], label='freq_left', alpha = 0.5)
    plt.scatter(masked_df['onboard.motor.m4'], masked_df['optitrack.freq.right'], label='freq_right', alpha=0.5)
    plt.plot(masked_df['onboard.motor.m2'], coeffs[0]*masked_df['onboard.motor.m2'] + coeffs[1], label='fit line', color='lightgreen')
    plt.title(rf'PWM to $f$ relationship (R² = {r2:.4f})')
    plt.xlabel('pwm values')
    plt.ylabel(r'$f$ (Hz)')
    plt.legend()
    plt.tight_layout()

    if save_fig:
        plt.savefig("outputs/power_frequency_regression.png")

    print("                                                                      ")
    print("======================================================================")
    return coeffs, r2
    

def regression_vertical_forces():
    dataframes = {"hover1": slice(627, 3131), "hover2": slice(690, 2595), 
                  "climb1": slice(1002, 5070), "climb2": slice(613, 5104)}
    columns = ["optitrack.freq.left", "optitrack.freq.right", "optitrack.acc.z", "optitrack.vel.z"]
    print("======================================================================")
    print("                                                                      ")
    print("Regression for vertical ascend / descent")
    # print("Using a linear model for the thrust; T(f) = c1 * f + c2")
    
    dfs = []
    for data_name, row_slice in dataframes.items():
        file_path = DATA_DIR / data_name / f"{data_name}-processed.csv"
        df = pd.read_csv(file_path, usecols=columns)
        df = df.iloc[row_slice] 
        dfs.append(df)

    
    
    combined_df = pd.concat(dfs, ignore_index=True)
    w_dot = combined_df["optitrack.acc.z"]
    w = combined_df["optitrack.vel.z"]
    freq_L = combined_df["optitrack.freq.left"]
    freq_R = combined_df["optitrack.freq.right"]
    g0 = 9.80665
    b = config_old.MASS_WINGS * (w_dot + g0)
    A = np.vstack([- w * (freq_L + freq_R), -(freq_L + freq_R), -2 * np.ones_like(w)]).T
    
    coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    b_pred = A @ coeffs
    r2 = r2_score(b, b_pred)
    
    print(f"The R^2 regression score for the vertical regression is {r2:.4f}")
    print(f"Linear regression parameters are: k_zw = {coeffs[0]:.6f}, c1 = {coeffs[1]:.6f}, c2 = {coeffs[2]:.6f}")
    
    # Plot predicted vs actual
    # plt.figure()
    
    # plt.scatter(b, b_pred, alpha=0.5)
    # plt.plot([b.min(), b.max()], [b.min(), b.max()], 'r--', label='Perfect fit')
    # plt.xlabel('Actual Force (N)')
    # plt.ylabel('Predicted Force (N)')
    # plt.title(f'Model Fit (R² = {r2:.4f})')
    # plt.legend()
    # plt.tight_layout()

    if save_fig:
        plt.savefig("outputs/vertical_regression.png")

    print("                                                                      ")
    print("======================================================================")
    return coeffs, r2


def regression_longitudinal_forces(thrust_coeffs=0):
    dataframes = {
    "longitudinal1": slice(720, 5229),
    # "longitudinal2": slice(608, 5151)
    }

    columns = [
        "time", "optitrack.freq.left", "optitrack.freq.right", 
        "optitrack.acc.x", "optitrack.vel.x", "optitrack.vel.z", 
        "optitrack.dihedral.left", "optitrack.dihedral.right", 
        "optitrack.q", "onboard.q_dot", "optitrack.q_dot", "optitrack.pitch", "optitrack.r","optitrack.roll"
    ]

    print("=" * 70)
    print("\nRegression for longitudinal maneuvres\n")

    # Load and process all dataframes
    dfs = []
    for data_name, row_slice in dataframes.items():
        file_path = DATA_DIR / data_name / f"{data_name}-processed.csv"
        df = pd.read_csv(file_path, usecols=columns).iloc[row_slice].copy()
        
        df["dihedral"] = df["optitrack.dihedral.right"]
        df["grad_dihedral"] = np.gradient(df["dihedral"], df["time"])
        
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Extract variables
    dihedral = combined_df["dihedral"]
    grad_dihedral = combined_df["grad_dihedral"]
    u_dot = combined_df["optitrack.acc.x"]
    u = combined_df["optitrack.vel.x"]
    w = combined_df["optitrack.vel.z"]
    p = combined_df["optitrack.roll"]
    r = combined_df["optitrack.r"]
    q = combined_df["optitrack.q"]
    q_dot_onboard = combined_df["onboard.q_dot"]
    q_dot_optitrack = combined_df["optitrack.q_dot"]
    theta = combined_df["optitrack.pitch"]
    freq_L = combined_df["optitrack.freq.left"]
    freq_R = combined_df["optitrack.freq.right"]
    lz = config_old.FLAPPER_DIMS["lz"]
    lw = config_old.FLAPPER_DIMS["lw"]
    m = config_old.MASS_WINGS
    Iyy = config_old.MMOI_WITH_WINGS_XY["Iyy"]

    # Compute state variables
    alpha_L = compute_angle_attack(dihedral, 0, "L")
    alpha_R = compute_angle_attack(dihedral, 0, "R")

    thrust_left = compute_thrust(thrust_coeffs, freq_L)
    thrust_right = compute_thrust(thrust_coeffs, freq_R)

    z_L = freq_L * (w - lw * np.sin(dihedral)*q + lw * np.cos(dihedral) * p)
    z_R = freq_R * (w - lw * np.sin(dihedral)*q - lw * np.cos(dihedral) * p)

    # Forces in body x-axis regression
    b_forces = m * (u_dot + g0 * np.sin(theta) + w*q)
    A_forces = np.vstack([-(freq_L + freq_R)*(u - lz * q + lw * grad_dihedral*np.sin(dihedral)), -(thrust_left * np.sin(alpha_L) + thrust_right * np.sin(alpha_R))]).T

    coeffs_forces, residuals_forces, rank_forces, s_forces = np.linalg.lstsq(A_forces, b_forces, rcond=None)
    b_pred_forces = A_forces @ coeffs_forces
    r2_forces = r2_score(b_forces, b_pred_forces)

    # Pitch moments regression -> set p, r to zero

    spline = UnivariateSpline(combined_df["time"], q, s=10.0)

    # Evaluate smoothed curve
    q_smooth = spline(combined_df["time"])

    # Derivative
    q_dot_smooth = spline.derivative()(combined_df["time"])

    # dihedral neutral is to 0.187

    neutral_pos = 0.187

    k_M1_term = - (freq_L + freq_R) * (u - lz * q_smooth + lw * grad_dihedral * np.sin(dihedral - neutral_pos)) * lz

    k_M2_term = (thrust_left * np.sin(alpha_L) + thrust_right * np.sin(alpha_R)) * lz 
    
    k_M3_term = - (z_L + z_R) * lw * np.sin(dihedral - neutral_pos)

    # k_M4_term = (thrust_right * np.cos(alpha_R) + thrust_left * np.cos(alpha_L))*np.sin(dihedral)*lw
    k_M4_term = (thrust_right + thrust_left)* lw * np.sin(dihedral - neutral_pos)

    b_moment = Iyy * q_dot_smooth

    A_moment = np.vstack([k_M1_term, k_M2_term, k_M3_term, k_M4_term]).T
    
    coeffs_moment, residuals_moment, rank_moment, s_moment = np.linalg.lstsq(A_moment, b_moment, rcond=False)

    b_pred_moment = A_moment @ coeffs_moment
    r2_moment = r2_score(b_moment, b_pred_moment)

    print(f"The R^2 regression score for the longitudinal regression is {r2_forces:.4f}")
    print(f"Linear regression parameters are: k_zx = {coeffs_forces[0]:.6f}\n")

    print("="*70)

    print(f"\nThe R^2 regression score for the longitudinal regression moments is {r2_moment:.4f}")
    print(f"Linear regression parameters moments are: k_1M = {coeffs_moment[0]:.6f}, k_2M = {coeffs_moment[1]:.6f}, k_3M = {coeffs_moment[2]:.6f}, k_4M = {coeffs_moment[3]:.6f},")
    
    # Plot predicted vs actual forces
    plt.figure()
    plt.scatter(b_forces, b_pred_forces, alpha=0.5)
    plt.plot([b_forces.min(), b_forces.max()], [b_forces.min(), b_forces.max()], 'r--', label='Perfect fit')
    plt.xlabel('Actual Force (N)')
    plt.ylabel('Predicted Force (N)')
    plt.title(rf'Model Fit ($R^2$ = {r2_forces:.4f})')
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.scatter(b_moment, b_pred_moment, alpha=0.5)
    plt.plot([b_moment.min(), b_moment.max()], [b_moment.min(), b_moment.max()], 'r--', label='Perfect fit')
    plt.xlabel('Actual Force (N)')
    plt.ylabel('Predicted Force (N)')
    plt.title(rf'Model Fit ($R^2$ = {r2_moment:.4f})')
    plt.legend()
    plt.tight_layout()

    # Plot predicted vs actual moments
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True)

    # --------------------------
    # Top subplot: Moments
    # --------------------------
    ax.plot(combined_df["time"], b_pred_moment, label='Predicted Moment')
    ax.plot(combined_df["time"], Iyy * q_dot_smooth, label='Smoothed q̇ * Iyy')
    ax.plot(combined_df["time"], Iyy * q_dot_optitrack, alpha=0.3, label='Optitrack q̇ * Iyy')
    ax.axvline(x = 11.68, )
    ax.axvline(x = 17.90, )
    ax.set_ylabel('Moment (Nm)')
    ax.set_title(rf'Model Fit ($R^2$ = {r2_moment:.4f})')
    ax.legend()
    ax.grid(True)

    # # --------------------------
    # # Bottom subplot: Frequencies
    # # --------------------------
    # ax[1].plot(combined_df["time"], u, alpha=0.3, label='Left Frequency (scaled)')
    # ax[1].set_xlabel('Time (s)')
    # ax[1].set_ylabel('Frequency (scaled)')
    # ax[1].legend()
    # ax[1].axvline(x = 11.68, )
    # ax[1].axvline(x = 17.90, )
    # ax[1].grid(True)

    plt.tight_layout()


    if save_fig:
        plt.savefig("outputs/longitudinal_regression.png")



    print("                                                                      ")
    print("======================================================================")
    return coeffs_moment, r2_forces

def regression_lateral_forces():
    dataframes = {"lateral1": slice(727, 5163)}#, "lateral2": slice(849, 5305)}

    columns = ["time", "onboard.p", "optitrack.freq.left", "optitrack.freq.right", "optitrack.acc.y", "optitrack.vel.y", 
               "optitrack.roll", "optitrack.vel.z", "optitrack.p_dot", "optitrack.p", "optitrack.q", "optitrack.dihedral.right"]
    
    print("======================================================================")
    print("                                                                      ")
    print("Regression for lateral forces")
    
    dfs = []
    for data_name, row_slice in dataframes.items():
        file_path = DATA_DIR / data_name / f"{data_name}-processed.csv"
        df = pd.read_csv(file_path, usecols=columns)
        
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    v_dot = combined_df["optitrack.acc.y"]
    v = combined_df["optitrack.vel.y"]
    phi = combined_df["optitrack.roll"]
    w = combined_df["optitrack.vel.z"]
    p = combined_df["optitrack.p"]
    p_onboard = combined_df["onboard.p"]
    q = combined_df["optitrack.q"]
    p_dot = combined_df["optitrack.p_dot"]
    freq_L = combined_df["optitrack.freq.left"]
    freq_R = combined_df["optitrack.freq.right"]
    dihedral = combined_df["optitrack.dihedral.right"]


    lz = config_old.FLAPPER_DIMS["lz"]
    lw = config_old.FLAPPER_DIMS["lw"]
    m = config_old.MASS_WINGS
    Iyy = config_old.MMOI_WITH_WINGS_XY["Iyy"]


    # Compute state variables
    alpha_L = compute_angle_attack(dihedral, 0, "L")
    alpha_R = compute_angle_attack(dihedral, 0, "R")

    thrust_left = compute_thrust(thrust_coeffs, freq_L)
    thrust_right = compute_thrust(thrust_coeffs, freq_R)

    z_L = freq_L * (w - lw * np.sin(dihedral)*q + lw * np.cos(dihedral) * p)
    z_R = freq_R * (w - lw * np.sin(dihedral)*q - lw * np.cos(dihedral) * p)

    lz = config_old.FLAPPER_DIMS["lz"]


    b = m * (v_dot - g0 * np.sin(phi) - w * p)
    Y = - (freq_L + freq_R) * (v - lz * p)
    A = np.vstack([Y]).T

    coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    b_pred = A @ coeffs
    r2 = r2_score(b, b_pred)

    spline = UnivariateSpline(combined_df["time"], p_onboard, s=50.0)

    # Evaluate smoothed curve
    p_smooth = spline(combined_df["time"])

    # Derivative
    p_dot_smooth = spline.derivative()(combined_df["time"])

    k_L1_moment = - (z_L - z_R) * lw * np.cos(dihedral)

    k_L2_moment = Y * lz 

    k_L3_moment = lw * np.sin(dihedral) * (thrust_left * np.cos(alpha_L) - thrust_right*np.cos(alpha_R))


    b_moment = Iyy * p_dot_smooth

    A_moment = np.vstack([k_L1_moment, k_L2_moment, k_L3_moment]).T
    
    coeffs_moment, residuals_moment, rank_moment, s_moment = np.linalg.lstsq(A_moment, b_moment, rcond=False)

    b_pred_moment = A_moment @ coeffs_moment
    r2_moment = r2_score(b_moment, b_pred_moment)

    print(f"The R^2 regression score for the lateral regression is {r2:.4f}")
    print(f"Linear regression parameters are: k_zx = {coeffs[0]:.6f}")
    
    print("="*70)

    print(f"\nThe R^2 regression score for the longitudinal regression moments is {r2_moment:.4f}")
    print(f"Linear regression parameters moments are: k_1M = {coeffs_moment[0]:.6f}, k_2M = {coeffs_moment[1]:.6f}, k_3M = {coeffs_moment[2]:.6f}")

    # Plot predicted vs actual
    plt.figure()
    
    plt.scatter(b, b_pred, alpha=0.5)
    plt.plot([b.min(), b.max()], [b.min(), b.max()], 'r--', label='Perfect fit')
    plt.xlabel('Actual Force (N)')
    plt.ylabel('Predicted Force (N)')
    plt.title(f'Model Fit (R² = {r2:.4f})')
    plt.legend()
    plt.tight_layout()


    plt.figure()
    plt.scatter(b_moment, b_pred_moment, alpha=0.5)
    plt.plot([b_moment.min(), b_moment.max()], [b_moment.min(), b_moment.max()], 'r--', label='Perfect fit')
    plt.xlabel('Actual Force (N)')
    plt.ylabel('Predicted Force (N)')
    plt.title(rf'Model Fit ($R^2$ = {r2_moment:.4f})')
    plt.legend()
    plt.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True)

    # --------------------------
    # Top subplot: Moments
    # --------------------------
    ax.plot(combined_df["time"], Iyy * p_dot, label='p_dot unsmoothed')
    ax.plot(combined_df["time"], Iyy * p_dot_smooth, label='Smoothed q̇ * Iyy')
    ax.plot(combined_df["time"], b_pred_moment, alpha=1, label='Optitrack q̇ * Iyy')
    ax.axvline(x = 11.68, )
    ax.axvline(x = 17.90, )
    ax.set_ylabel('Moment (Nm)')
    ax.set_title(rf'Model Fit ($R^2$ = {r2_moment:.4f})')
    ax.legend()
    ax.grid(True)
    if save_fig:
        plt.savefig("outputs/lateral_regression.png")

    print("                                                                      ")
    print("======================================================================")
    return coeffs, r2


if __name__ == "__main__":
    # regression_pwm_frequency()
    thrust_coeffs, _ = regression_vertical_forces()
    regression_longitudinal_forces(thrust_coeffs)
    # regression_lateral_forces()


    # ideally compile this coefficients into a json file for nn
    plt.show()
