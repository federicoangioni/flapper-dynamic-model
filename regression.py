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
    columns_to_keep = ['optitrack.freq.left', 'optitrack.freq.right', 'onboard.acc.z']
    
    dfs = []

    for dir_path in DATA_DIR.iterdir():
        if dir_path.is_dir():
            for file_path in dir_path.glob("*_processed.csv"):
                print(f"Found dataset: {file_path.name}")
                # Find files ending with _processed.csv
                df = pd.read_csv(file_path, usecols=columns_to_keep)
                dfs.append(df)




    pass



if __name__ == "__main__":
    regression_pwm_frequency()