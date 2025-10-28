import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from utils import config

WORKING_DIR = Path.cwd()
DATA_DIR = WORKING_DIR / 'data'
g0 = 9.80665

def compute_thrust(coeffs, f):
    """T = m*f + c"""
    return coeffs[0]*f + coeffs[1]

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
    plt.title(r'PWM to $f$ relationship (R² = {r2:.4f})')
    plt.xlabel('pwm values')
    plt.ylabel(r'$f$ (Hz)')
    plt.legend()

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
    # print("Using a linear model for the thrust; T(f) = k_zw * w * f + c1 * f + c2")
    
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
    b = config.MASS_WINGS * (w_dot + g0)
    A = np.vstack([- w * (freq_L + freq_R), -(freq_L + freq_R), -2 * np.ones_like(w)]).T
    
    coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    b_pred = A @ coeffs
    r2 = r2_score(b, b_pred)
    
    print(f"The R^2 regression score for the vertical regression is {r2:.4f}")
    print(f"Linear regression parameters are: k_zw = {coeffs[0]:.6f}, c1 = {coeffs[1]:.6f}, c2 = {coeffs[2]:.6f}")
    
    # Plot predicted vs actual
    plt.figure(figsize=(10, 5))
    
    plt.scatter(b, b_pred, alpha=0.5)
    plt.plot([b.min(), b.max()], [b.min(), b.max()], 'r--', label='Perfect fit')
    plt.xlabel('Actual Force (N)')
    plt.ylabel('Predicted Force (N)')
    plt.title(f'Model Fit (R² = {r2:.4f})')
    plt.legend()

    print("                                                                      ")
    print("======================================================================")
    return coeffs, r2


def regression_longitudinal(c1=0, c2=0):
    dataframes = {"longitudinal1": slice(720, 5229), "longitudinal2": slice(608, 5151)}

    columns = ["time", "optitrack.freq.left", "optitrack.freq.right", "optitrack.acc.x", "optitrack.vel.x", "optitrack.vel.z", 
               "optitrack.dihedral.left", "optitrack.dihedral.right", "optitrack.q" ,"optitrack.pitch"]
    
    print("======================================================================")
    print("                                                                      ")
    print("Regression for longitudinal maneuvres")
    
    dfs = []
    for data_name, row_slice in dataframes.items():
        file_path = DATA_DIR / data_name / f"{data_name}-processed.csv"
        df = pd.read_csv(file_path, usecols=columns)
        df = df.iloc[row_slice].copy()

        df["dihedral"] = (df["optitrack.dihedral.left"] + df["optitrack.dihedral.right"]) / 2
        df["grad_dihedral"] = np.gradient(df["dihedral"], df["time"])
        
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    dihedral = combined_df["dihedral"]

    grad_dihedral = combined_df["grad_dihedral"]

    u_dot = combined_df["optitrack.acc.x"]
    u = combined_df["optitrack.vel.x"]
    w = combined_df["optitrack.vel.z"]
    q = combined_df["optitrack.q"]
    theta = combined_df["optitrack.pitch"]
    freq_L = combined_df["optitrack.freq.left"]
    freq_R = combined_df["optitrack.freq.right"]

    lz = config.FLAPPER_DIMS["lz"]
    lw = config.FLAPPER_DIMS["lw"]

    b = config.MASS_WINGS * (u_dot + g0 * np.sin(theta) + w*q)

    A = np.vstack([-(freq_L + freq_R)*(u - lz * q + lw*grad_dihedral*np.sin(dihedral))]).T

    coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    b_pred = A @ coeffs
    r2 = r2_score(b, b_pred)


    print(f"The R^2 regression score for the longitudinal regression is {r2:.4f}")
    print(f"Linear regression parameters are: k_zx = {coeffs[0]:.6f}")
    
    # Plot predicted vs actual
    plt.figure(figsize=(10, 5))
    
    plt.scatter(b, b_pred, alpha=0.5)
    plt.plot([b.min(), b.max()], [b.min(), b.max()], 'r--', label='Perfect fit')
    plt.xlabel('Actual Force (N)')
    plt.ylabel('Predicted Force (N)')
    plt.title(f'Model Fit (R² = {r2:.4f})')
    plt.legend()


    print("                                                                      ")
    print("======================================================================")
    return coeffs, r2

def regression_lateral_forces():
    dataframes = {"lateral1": slice(720, 5229), "lateral2": slice(608, 5151)}

    columns = []
    
    print("======================================================================")
    print("                                                                      ")
    print("Regression for longitudinal maneuvres")
    
    dfs = []
    for data_name, row_slice in dataframes.items():
        file_path = DATA_DIR / data_name / f"{data_name}-processed.csv"
        df = pd.read_csv(file_path, usecols=columns)
        
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)



    b = config.MASS_WINGS * (g0)

    A = np.vstack([]).T


    coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    b_pred = A @ coeffs
    r2 = r2_score(b, b_pred)


    print(f"The R^2 regression score for the longitudinal regression is {r2:.4f}")
    print(f"Linear regression parameters are: k_zx = {coeffs[0]:.6f}")
    
    # Plot predicted vs actual
    plt.figure(figsize=(10, 5))
    
    plt.scatter(b, b_pred, alpha=0.5)
    plt.plot([b.min(), b.max()], [b.min(), b.max()], 'r--', label='Perfect fit')
    plt.xlabel('Actual Force (N)')
    plt.ylabel('Predicted Force (N)')
    plt.title(f'Model Fit (R² = {r2:.4f})')
    plt.legend()


    print("                                                                      ")
    print("======================================================================")
    return coeffs, r2


if __name__ == "__main__":
    regression_pwm_frequency()
    coeffs_vertical = regression_vertical_forces()
    regression_longitudinal()
    # plt.show()
