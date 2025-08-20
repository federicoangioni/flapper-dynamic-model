import pandas as pd
import matplotlib.pyplot as plt
from time import time
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy import signal

# Local imports
from controller import PID_controller
from state_estimator import MahonyIMU

# TODO
# Verify pitch, yaw and roll angles are estimated correctly, compare with optitrack -> they are correct

flight_exp = "flight_001"

# Attitude PID definitions
freq_attitude = 500  # Hz

pitch_kp = 13
pitch_ki = 0
pitch_kd = 1
pitch_kff = 0

roll_kp = 10
roll_ki = 0
roll_kd = 0.2
roll_kff = 0

yaw_kp = 8
yaw_ki = 0
yaw_kd = 0.35
yaw_kff = 0

# Rate PID definitions
freq_attitude_rate = 500  # Hz

pitchrate_kp = 50
pitchrate_ki = 0
pitchrate_kd = 0
pitchrate_kff = 0

rollrate_kp = 50
rollrate_ki = 0
rollrate_kd = 0
rollrate_kff = 0

yawrate_kp = 80
yawrate_ki = 0
yawrate_kd = 0
yawrate_kff = 220

# Ouput of attitude controllers
controller_pitch = []
controller_roll = []
controller_yaw = []

# Ouput of attitude rate controllers
controller_pitchrate = []
controller_rollrate = []
controller_yawrate = []

sensfusion = MahonyIMU()

# Instantiate PID attitude controllers
pitch_pid = PID_controller(pitch_kp, pitch_ki, pitch_kd)
roll_pid = PID_controller(roll_kp, roll_ki, roll_kd)
yaw_pid = PID_controller(yaw_kp, yaw_ki, yaw_kd)

# Instantiate PID attitude rate controllers
pitchrate_pid = PID_controller(pitchrate_kp, pitchrate_ki, pitchrate_kd)
rollrate_pid = PID_controller(rollrate_kp, rollrate_ki, rollrate_kd)
yawrate_pid = PID_controller(yawrate_kp, yawrate_ki, yawrate_kd)


def attitude_controller(onboard, i, dt):
    gx, gy, gz = onboard.loc[i, "gyro.x":"gyro.z"].to_numpy().T
    ax, ay, az = onboard.loc[i, "acc.x":"acc.z"].to_numpy().T
    pitch_sp, roll_sp, yaw_sp = (
        onboard.loc[i, "controller.pitch":"controller.yaw"]
    )  # .pitch, .roll, .yaw is the setpoint according to craziflie docs, check controller_pid.c

    qx, qy, qz, qw = sensfusion.sensfusion6Update(gx, gy, gz, ax, ay, az, dt)

    yaw, pitch, roll = np.degrees(R.from_quat([qx, qy, qz, qw]).as_euler(
        "ZYX"
    ))  # - - +, in radians

    controller_pitch.append(pitch_pid.compute(pitch, pitch_sp, dt))
    controller_roll.append(roll_pid.compute(-roll, roll_sp, dt))
    controller_yaw.append(yaw_pid.compute(-yaw, yaw_sp, dt))


def rate_controller(onboard, i, dt):
    gx, gy, gz = np.radians(onboard.loc[i, "gyro.x":"gyro.z"].to_numpy().T)
    pitch_sp, roll_sp, yaw_sp = (
        onboard.loc[i, "controller.pitchRate":"controller.yawRate"]
    )  # .pitchRate, .rollRate, .yawRate is the setpoint according to craziflie docs, check controller_pid.c


    pitchrate_sp = controller_pitch[-1]
    rollrate_sp = controller_roll[-1]
    yawrate_sp = controller_yaw[-1]

    controller_pitchrate.append(pitchrate_pid.compute(gy, pitchrate_sp, dt))
    controller_rollrate.append(pitchrate_pid.compute(gx, rollrate_sp, dt))
    controller_yawrate.append(pitchrate_pid.compute(gz, yawrate_sp, dt))




if __name__ == "__main__":
    start = time()

    # Declare data file paths
    data_dir = f"data/{flight_exp}/{flight_exp}"
    optitrack_csv = f"{data_dir}_optitrack.csv"
    onboard_csv = f"{data_dir}_flapper.csv"
    processed_csv = f"{data_dir}_processed.csv"

    # Load onboard data
    onboard_data = pd.read_csv(onboard_csv)
    processed_data = pd.read_csv(processed_csv)

    for i in range(len(onboard_data)):
        attitude_controller(onboard_data, i, 1 / freq_attitude)
        rate_controller(onboard_data, i, 1 / freq_attitude)

    end = time()

    print(f"Process run in {round(end - start, 3)} s")