import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils import config

WORKING_DIR = Path.cwd()
DATA_DIR = WORKING_DIR / 'data' / config.PLATFORM

def regression_pwm_frequency():
    columns_to_keep = ['optitrack.freq.left', 'onboard.motor.m2', 'optitrack.freq.right', 'onboard.motor.m4']

    dfs = []

    for dir_path in DATA_DIR.iterdir():
        if dir_path.is_dir():
            for file_path in dir_path.glob("*_processed.csv"):
                print(f"Found dataset: {file_path.name}")
                # Find files ending with _processed.csv
                df = pd.read_csv(file_path, usecols=columns_to_keep)
                dfs.append(df)

    if dfs:  # Check if list is not empty
        combined_df = pd.concat(dfs, ignore_index=True)
    else:
        print("No _processed.csv files found.")

    mask = (combined_df['onboard.motor.m2'] > 1e4) & (combined_df['onboard.motor.m4'] > 1e4)

    masked_df = combined_df.loc[mask, :]

    motor_inputs = pd.concat([masked_df['onboard.motor.m2'], masked_df['onboard.motor.m4']], axis=0).to_numpy()

    frequency = pd.concat([masked_df['optitrack.freq.left'], masked_df['optitrack.freq.right']], axis=0)

    A = np.vstack([motor_inputs, np.ones(len(motor_inputs))]).T

    b = frequency

    m, c = np.linalg.lstsq(A, b)[0]

    print("Linear regression parameters are ", m, " * x + ", c)
    plt.scatter(masked_df['onboard.motor.m2'], masked_df['optitrack.freq.left'], label='freq_left', alpha = 0.5)
    plt.scatter(masked_df['onboard.motor.m4'], masked_df['optitrack.freq.right'], label='freq_right', alpha=0.5)
    plt.plot(masked_df['onboard.motor.m2'], m*masked_df['onboard.motor.m2'] + c, label='fit line', color='lightgreen')
    plt.title(r'PWM to $f$ relationship')
    plt.xlabel('pwm values')
    plt.ylabel(r'$f$ (Hz)')
    plt.legend()
    plt.show()

def regression_thrust_coeffs():
    columns_to_keep = ['optitrack.freq.left', 'optitrack.freq.right', 'onboard.vel.z','onboard.acc.z']
    
    dfs = []

    for dir_path in DATA_DIR.iterdir():
        if dir_path.is_dir():
            for file_path in dir_path.glob("*_processed.csv"):
                print(f"Found dataset: {file_path.name}")
                df = pd.read_csv(file_path, usecols=columns_to_keep)

                # Find files ending with _processed.csv
                if file_path.name == 'flight_001_processed.csv':
                    df = df.iloc[1200: 1300]
                elif file_path.name == 'flight_002_processed.csv':
                    df = df.iloc[1268: 1400]
                
                dfs.append(df)

    if dfs:  # Check if list is not empty
        combined_df = pd.concat(dfs, ignore_index=True)
    else:
        print("No _processed.csv files found.")

    average_freq = (
        combined_df['optitrack.freq.left'] + combined_df['optitrack.freq.right']
    ) / 2

    frequency = pd.concat([average_freq], axis=0).to_numpy()

    # vertical z equation of motion -> m*a_z = - 2 * (c1 * f + c2) - f * w * bz

    thrust = -0.5 * combined_df['onboard.acc.z'] * config.MASS_WINGS  - config.MASS_WINGS * 9.81 # - 0.5 * average_freq * combined_df['onboard.vel.z'] * 9.14e-3

    A = np.vstack([frequency, np.ones(len(frequency))]).T

    b = thrust

    m, c = np.linalg.lstsq(A, b)[0]
        
    print("Linear regression parameters are ", m, " * x + ", c)
    plt.scatter(frequency, thrust, alpha=0.5)
    plt.plot(frequency, m*frequency + c, label='fit line', color='lightgreen')
    plt.title(r'$f$ to $T$ relationship')
    plt.xlabel(r'$f$ values')
    plt.ylabel(r'T (N)')
    plt.show()



if __name__ == "__main__":
    regression_pwm_frequency()
    regression_thrust_coeffs()
