import pandas as pd
import matplotlib.pyplot as plt
from time import time
from scipy.spatial.transform import Rotation as R
import numpy as np


# Local imports
from utils.controller import PID_controller
from utils.state_estimator import MahonyIMU

show = True
flight_exp = "flight_001"

# Attitude PID definitions
freq_attitude = 500  # Hz

roll_kp = 10
roll_ki = 0
roll_kd = 0.2
roll_kff = 0
roll_integration_limit = 20.0

pitch_kp = 13
pitch_ki = 0
pitch_kd = 1
pitch_kff = 0
pitch_integration_limit = 20.0

yaw_kp = 8
yaw_ki = 0
yaw_kd = 0.35
yaw_kff = 0
yaw_integration_limit = 360.0

# Rate PID definitions
freq_attitude_rate = 500  # Hz

rollrate_kp = 50
rollrate_ki = 0
rollrate_kd = 0
rollrate_kff = 0
omxFiltCut = 20
rollrate_integration_limit = 33.3

pitchrate_kp = 50
pitchrate_ki = 0
pitchrate_kd = 0
pitchrate_kff = 0
omyFiltCut = 20
pitchrate_integration_limit = 33.3

yawrate_kp = 80
yawrate_ki = 0
yawrate_kd = 0
yawrate_kff = 220
omzFiltCut = 5
yawrate_integration_limit = 166.7

# -----------------------------------------------------------
# Ouput of attitude rate controllers
cmd_roll = []
cmd_pitch = []
cmd_yaw = []

sensfusion = MahonyIMU()

# Instantiate PID attitude controllers
roll_pid = PID_controller(roll_kp, roll_ki, roll_kd, roll_kff, roll_integration_limit, 1 / freq_attitude, freq_attitude, 0, False)
pitch_pid = PID_controller(pitch_kp, pitch_ki, pitch_kd, pitch_kff, pitch_integration_limit, 1 / freq_attitude, freq_attitude, 0, False)
yaw_pid = PID_controller(yaw_kp, yaw_ki, yaw_kd, yaw_kff, yaw_integration_limit, 1 / freq_attitude, freq_attitude, 0, False)

# Instantiate PID attitude rate controllers
rollrate_pid = PID_controller(rollrate_kp, rollrate_ki, rollrate_kd, rollrate_kff, rollrate_integration_limit, 1 / freq_attitude_rate, freq_attitude_rate, omxFiltCut, True)
pitchrate_pid = PID_controller(pitchrate_kp, pitchrate_ki, pitchrate_kd, pitchrate_kff, pitchrate_integration_limit, 1 / freq_attitude_rate, freq_attitude_rate, omyFiltCut, True)
yawrate_pid = PID_controller(yawrate_kp, yawrate_ki, yawrate_kd, yawrate_kff, yawrate_integration_limit, 1 / freq_attitude_rate, freq_attitude_rate, omzFiltCut, True)


def state_estimation(gx, gy, gz, ax, ay, az, dt):
    qx, qy, qz, qw = sensfusion.sensfusion6Update(gx, gy, gz, ax, ay, az, dt)

    yaw_i, pitch_i, roll_i = np.degrees(R.from_quat([qx, qy, qz, qw]).as_euler("ZYX"))  # - - +, in radians

    return roll_i, pitch_i, yaw_i


def attitude_controller(attitude_measured, attitude_sp):
    roll_sp, pitch_sp, yaw_sp = attitude_sp

    roll_i, pitch_i, yaw_i = attitude_measured

    controller_rollrate_i = roll_pid.compute(roll_i, roll_sp, False)
    controller_pitchrate_i = pitch_pid.compute(pitch_i, pitch_sp, False)
    controller_yawrate_i = yaw_pid.compute(yaw_i, yaw_sp, True)

    return controller_rollrate_i, controller_pitchrate_i, controller_yawrate_i


def rate_controller(attitude_rate_measured, attitude_rate_sp):
    rollrate_sp, pitchrate_sp, yawrate_sp = attitude_rate_sp

    gx, gy, gz = attitude_rate_measured

    cmd_roll_i = rollrate_pid.compute(gx, rollrate_sp, False)
    cmd_pitch_i = pitchrate_pid.compute(gy, pitchrate_sp, False)
    cmd_yaw_i = yawrate_pid.compute(gz, yawrate_sp, False)

    return cmd_roll_i, cmd_pitch_i, cmd_yaw_i


def controller_pid(sensor_rates, acc, attitude_sp, dt_imu):
    gx, gy, gz = sensor_rates
    ax, ay, az = acc
    attitude_measured = state_estimation(gx, gy, gz, ax, ay, az, dt_imu)

    controller_rate_sp = attitude_controller(attitude_measured, attitude_sp)

    cmd_roll_i, cmd_pitch_i, cmd_yaw_i = rate_controller(sensor_rates, controller_rate_sp)

    return cmd_roll_i, cmd_pitch_i, cmd_yaw_i


def power_distribution(onboard, i, dt):
    raise NotImplementedError

def open_loop():
    raise NotImplementedError


def flapper(onboard, i, dt):
    sensor_rates = onboard.loc[i, ["gyro.x", "gyro.y", "gyro.z"]].to_numpy().T
    acc = onboard.loc[i, ["acc.x", "acc.y" ,"acc.z"]].to_numpy().T
    attitude_sp = onboard.loc[i, ["controller.roll", "controller.pitch", "controller.yaw"]].to_numpy().T

    cmd_roll_i, cmd_pitch_i, cmd_yaw_i = controller_pid(sensor_rates, acc, attitude_sp, dt)

    cmd_roll.append(cmd_roll_i)
    cmd_pitch.append(cmd_pitch_i)
    cmd_yaw.append(cmd_yaw_i)


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
        flapper(onboard_data, i, 1 / freq_attitude)

    end = time()

    print(f"Process run in {round(end - start, 3)} s")

    if show:
        print("Showing the outputs in plots")
        fig, axs = plt.subplots(nrows=3, ncols=3)

        axs[0, 0].set_title("Pitch command from rate PID")
        axs[0, 0].plot(cmd_pitch)
        axs[0, 0].plot(onboard_data["controller.cmd_pitch"], alpha=0.5)

        axs[1, 0].set_title("Roll command from rate PID")
        axs[1, 0].plot(cmd_roll)
        axs[1, 0].plot(onboard_data["controller.cmd_roll"], alpha=0.5)

        axs[2, 0].set_title("Yaw command from rate PID")
        axs[2, 0].plot(cmd_yaw)
        axs[2, 0].plot(onboard_data["controller.cmd_yaw"], alpha=0.5)
        plt.show()
