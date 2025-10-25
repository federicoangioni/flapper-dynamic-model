import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from utils import config

WORKING_DIR = Path.cwd()
DATA_DIR = WORKING_DIR / 'data' 

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
    dataframes = {"hover1":slice(627, 3131), "hover2":slice(690,2595), "climb1":slice(1002, 5070), "climb2":slice(613, 5104)}
    columns = ["optitrack.freq.left", "optitrack.freq.right", "optitrack.acc.z", "onboard.vel.z"]
    dfs = []

    for data in dataframes.keys():
        file_path = DATA_DIR / data /f"{data}-processed.csv"

        df = pd.read_csv(file_path, usecols=columns)

        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    w_dot = combined_df["optitrack.acc.z"]
    w = combined_df["onboard.vel.z"]
    freq_L = combined_df["optitrack.freq.left"]
    freq_R = combined_df["optitrack.freq.right"]
    
    b = config.MASS_WINGS*(w_dot + config.g0)

    A = np.vstack([-w * (freq_L + freq_R), -(freq_L + freq_R), -2 * np.ones_like(w)]).T



    coeffs, R, *_ = np.linalg.lstsq(A, b)

    b_pred = A @ coeffs

    r2 = r2_score(b, b_pred)

    print(r2)

    # print("Linear regression parameters are ", m, " * x + ", c)
    # plt.scatter(frequency, thrust, alpha=0.5)
    # plt.plot(frequency, m*frequency + c, label='fit line', color='lightgreen')
    plt.title(r'$f$ to $T$ relationship')
    plt.xlabel(r'$f$ values')
    plt.ylabel(r'T (N)')
    plt.show()



if __name__ == "__main__":
    # regression_pwm_frequency()
    regression_thrust_coeffs()
