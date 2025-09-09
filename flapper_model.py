import pandas as pd
import matplotlib.pyplot as plt
from time import time
from scipy.spatial.transform import Rotation as R
import numpy as np
from rich import print
from rich.progress import track

# Local imports
from utils.controller import PID_controller
from utils.state_estimator import MahonyIMU
from utils.power_distribution import power_distribution
from utils.open_loop import FlapperModel

show = True
flight_exp = "flight_001"
use_open_loop = False


# Choose frequency to run the controllers
if use_open_loop:
    freq_attitude = 500  # Hz
    freq_attitude_rate = 500  # Hz
    prefix_data = "onboard."
else:
    freq_attitude = 500  # Hz
    freq_attitude_rate = 500  # Hz
    prefix_data = ""

# Attitude PID definitions
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
yaw_max_delta = 30.0

# -----------------------------------------------------------
# Ouput of attitude rate controllers
cmd_roll = []
cmd_pitch = []
cmd_yaw = []

# Handle YAW rate controllers
yawrate_sp = []

# Pre-define the necessary dictionaries for the PID and power_distribution
attitude_measured = {"roll": 0, "pitch": 0, "yaw": 0}
attitude_desired = {"roll": 0, "pitch": 0, "yaw": 0}
attituderate_desired = {"rollrate": 0, "pitchrate": 0, "yawrate": 0}
controls = {"thrust": 0, "roll": 0, "pitch": 0, "yaw": 0}
motors_list = {"m1": [], "m2": [], "m3": [], "m4": []}

# Instantiate the sensor fusion filter
sensfusion = MahonyIMU()

# Instantiate PID attitude controllers
roll_pid = PID_controller(roll_kp, roll_ki, roll_kd, roll_kff, roll_integration_limit, 1 / freq_attitude, freq_attitude, 0, False)
pitch_pid = PID_controller(pitch_kp, pitch_ki, pitch_kd, pitch_kff, pitch_integration_limit, 1 / freq_attitude, freq_attitude, 0, False)
yaw_pid = PID_controller(yaw_kp, yaw_ki, yaw_kd, yaw_kff, yaw_integration_limit, 1 / freq_attitude, freq_attitude, 0, False)

# Instantiate PID attitude rate controllers
rollrate_pid = PID_controller(rollrate_kp, rollrate_ki, rollrate_kd, rollrate_kff, rollrate_integration_limit, 1 / freq_attitude_rate, freq_attitude_rate, omxFiltCut, True)
pitchrate_pid = PID_controller(pitchrate_kp, pitchrate_ki, pitchrate_kd, pitchrate_kff, pitchrate_integration_limit, 1 / freq_attitude_rate, freq_attitude_rate, omyFiltCut, True)
yawrate_pid = PID_controller(yawrate_kp, yawrate_ki, yawrate_kd, yawrate_kff, yawrate_integration_limit, 1 / freq_attitude_rate, freq_attitude_rate, omzFiltCut, True, 32767.0)

# Instantiate the open loop model
flapper_model = FlapperModel(freq_attitude)

def capAngle(angle):
    result = angle
    while result > 180.0:
        result -= 360.0
    while result < -180.0:
        result += 360.0
    return result


def state_estimation(rates, acc, dt):
    """
    Estimates the current attitude through a Mahony filter

    Parameters:
    -----------
        rates: list
            Angular rates in deg / s as recorded from the IMU
        acc: list
            Acceleration as recorded from the IMU (in g)
        dt: float
            IMU delta t
        
    Returns:
    --------
        roll_i, -pitch_i, yaw_i: floats
            Estimated angular attitude in deg
    """
    gx, gy, gz = rates
    ax, ay, az = acc

    qx, qy, qz, qw = sensfusion.sensfusion6Update(gx, gy, gz, ax, ay, az, dt)

    yaw_i, pitch_i, roll_i = np.degrees(R.from_quat([qx, qy, qz, qw]).as_euler("ZYX"))

    return roll_i, -pitch_i, yaw_i


def attitude_controller(roll_i, pitch_i, yaw_i, roll_sp_i, pitch_sp_i, yaw_sp_i):
    controller_rollrate_i = roll_pid.compute(roll_i, roll_sp_i, False)
    controller_pitchrate_i = pitch_pid.compute(pitch_i, pitch_sp_i, False)
    controller_yawrate_i = yaw_pid.compute(yaw_i, yaw_sp_i, True)

    return controller_rollrate_i, controller_pitchrate_i, controller_yawrate_i


def rate_controller(attitude_rate_measured, rollrate_sp, pitchrate_sp, yawrate_sp):
    gx, gy, gz = attitude_rate_measured

    cmd_roll_i = rollrate_pid.compute(gx, rollrate_sp, False)
    cmd_pitch_i = pitchrate_pid.compute(-gy, pitchrate_sp, False)
    cmd_yaw_i = yawrate_pid.compute(-gz, yawrate_sp, False)

    return cmd_roll_i, cmd_pitch_i, cmd_yaw_i


def controller_pid(attitude, rates, setpoints, dt_imu, yaw_max_delta, yaw_mode="velocity"):

    attitude_measured["roll"], attitude_measured["pitch"], attitude_measured["yaw"] = attitude

    if yaw_mode == "velocity":
        attitude_desired["yaw"] = capAngle(attitude_desired["yaw"] + setpoints["yawrate"] * dt_imu)

        if yaw_max_delta != 0.0:
            delta = capAngle(attitude_desired["yaw"] - attitude_measured["yaw"])

            if delta > yaw_max_delta:
                attitude_desired["yaw"] = attitude_measured["yaw"] + yaw_max_delta
            elif delta < -yaw_max_delta:
                attitude_desired["yaw"] = attitude_measured["yaw"] - yaw_max_delta

        attitude_desired["roll"] = setpoints["roll"]
        attitude_desired["pitch"] = setpoints["pitch"]
        attitude_desired["yaw"] = capAngle(attitude_desired["yaw"])

    elif yaw_mode == "manual":
        attitude_desired["yaw"] = setpoints["yaw"]
        attitude_desired["roll"] = setpoints["roll"]
        attitude_desired["pitch"] = setpoints["pitch"]

    controller_rate_sp = attitude_controller(
        attitude_measured["roll"], attitude_measured["pitch"], attitude_measured["yaw"], attitude_desired["roll"], attitude_desired["pitch"], attitude_desired["yaw"]
    )

    attituderate_desired["rollrate"] = controller_rate_sp[0]
    attituderate_desired["pitchrate"] = controller_rate_sp[1]
    attituderate_desired["yawrate"] = controller_rate_sp[2]

    cmd_roll_i, cmd_pitch_i, cmd_yaw_i = rate_controller(rates, attituderate_desired["rollrate"], attituderate_desired["pitchrate"], attituderate_desired["yawrate"])

    return cmd_roll_i, cmd_pitch_i, cmd_yaw_i


def simulate_flapper(data, i, dt, use_model : bool):

    if use_model:
        print("[red]You thought we implemented that haha![/red]")
    else:
        # Fetch data from onboard (unprocessed, for now) .csv
        rates = data.loc[i, [f"{prefix_data}gyro.x", f"{prefix_data}gyro.y", f"{prefix_data}gyro.z"]].to_numpy().T
        acc = data.loc[i, [f"{prefix_data}acc.x", f"{prefix_data}acc.y", f"{prefix_data}acc.z"]].to_numpy().T
        
        # Calculate estimated attitude through Mahony filter
        attitude = state_estimation(rates, acc, dt)
    

    setpoints = {"roll": data.loc[i, f"{prefix_data}controller.roll"], "pitch": data.loc[i, f"{prefix_data}controller.pitch"], "yaw": data.loc[i, f"{prefix_data}controller.yaw"], "yawrate": data.loc[i, f"{prefix_data}controller.yawRate"]}
    cmd_thrust = data.loc[i, f"{prefix_data}controller.cmd_thrust"]

    # Run the PID cascade
    cmd_roll_i, cmd_pitch_i, cmd_yaw_i = controller_pid(attitude, rates, setpoints, dt, yaw_max_delta, yaw_mode="manual")

    controls["thrust"] = cmd_thrust
    controls["pitch"] = cmd_pitch_i
    controls["roll"] = cmd_roll_i
    controls["yaw"] = -cmd_yaw_i
    
    motors = power_distribution(controls)

    # For now save only the cmd outputs and the relative motor outputss
    cmd_roll.append(cmd_roll_i)
    cmd_pitch.append(cmd_pitch_i)
    cmd_yaw.append(-cmd_yaw_i)
    motors_list["m1"].append(motors["m1"])
    motors_list["m2"].append(motors["m2"])
    motors_list["m3"].append(motors["m3"])
    motors_list["m4"].append(motors["m4"])


"""
The main loop runs at 1000 hz, 
attitude and position at 500 hz
"""

if __name__ == "__main__":
    start = time()

    # Declare data file paths
    data_dir = f"data/raw/{flight_exp}/{flight_exp}"
    onboard_csv = f"{data_dir}_flapper.csv"
    processed_csv = f"{data_dir}_oriented_processed.csv"

    # Load onboard data
    if use_open_loop:
        print("[bold red]I'm still implementing this feature![/bold red]")
        exit()
        data = 0 # pd.read_csv(processed_csv)
    else:
        print("[bold thistle1]Running the controllers with the recorded data, here no open loop models are run. The data recorded from the IMU gets fed back to the controllers. [/bold thistle1]")
        data = pd.read_csv(onboard_csv)
        # data = pd.read_csv(processed_csv)
    

    print("[bold green]Starting the simulation[/bold green]")
    for i in track(range(len(data)), description="Processing..."):
        simulate_flapper(data, i, 1 / freq_attitude, use_open_loop)

    end = time()

    print(f"[magenta]Process run in {round(end - start, 3)} s[/magenta]")

    if show:
        print("Showing the outputs in plots")

        # First figure
        fig1, axs1 = plt.subplots(nrows=3, ncols=1)

        axs1[0].set_title("Pitch angle command from rate PID")
        axs1[0].plot(cmd_pitch, label="simulated")
        axs1[0].plot(data[f"{prefix_data}controller.cmd_pitch"], alpha=0.5, label="recorded")
        axs1[0].legend()

        axs1[1].set_title("Roll angle command from rate PID")
        axs1[1].plot(cmd_roll)
        axs1[1].plot(data[f"{prefix_data}controller.cmd_roll"], alpha=0.5)

        axs1[2].set_title("Yaw angle command from rate PID")
        axs1[2].plot(cmd_yaw)
        axs1[2].plot(data[f"{prefix_data}controller.cmd_yaw"], alpha=0.5)
        axs1[2].set_ylabel("time (s)")
        axs1[2].set_ylim(-32767, 32767)

        plt.tight_layout()

        # Second figure
        fig2, axs2 = plt.subplots(nrows=4, ncols=1)

        axs2[0].set_title("motor commands m1")
        axs2[0].plot(motors_list["m1"], label="simulated")
        axs2[0].plot(data[f"{prefix_data}motor.m1"], label="recorded", alpha=0.5)

        axs2[1].set_title("motor commands m2")
        axs2[1].plot(motors_list["m2"])
        axs2[1].plot(data[f"{prefix_data}motor.m2"], alpha=0.5)

        axs2[2].set_title("motor commands m3")
        axs2[2].plot(motors_list["m3"])
        axs2[2].plot(data[f"{prefix_data}motor.m3"], alpha=0.5)

        axs2[3].set_title("motor commands m4")
        axs2[3].plot(motors_list["m4"])
        axs2[3].plot(data[f"{prefix_data}motor.m4"], alpha=0.5)

        plt.tight_layout()

        # Third figure, plot angles, rates and velocities
        # fig3, axs3 = plt.subplots(nrows=4, ncols=4)
        
        # axs3[0, 0].plot(data[f"{prefix_data}pitch"])

        plt.tight_layout()


        plt.show()
