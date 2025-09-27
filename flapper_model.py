import matplotlib.pyplot as plt
from time import time
from scipy.spatial.transform import Rotation as R
import numpy as np
from rich import print
from rich.progress import track

# Local imports
import utils.config as config
from utils.controller import PID_controller
from utils.state_estimator import MahonyIMU
from utils.power_distribution import power_distribution
from utils.open_loop import FlapperModel
from utils.data_loader import load_data


show = True
flight_exp = "flight_001"


# Choose frequency to run the controllers

freq_attitude = 500  # Hz
freq_attitude_rate = 500  # Hz
prefix_data = ""


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
roll_pid = PID_controller(
    config.ROLL_KP, config.ROLL_KI, config.ROLL_KD, config.ROLL_KFF,
    config.ROLL_INTEGRATION_LIMIT, 1 / freq_attitude, freq_attitude, 0, False
)
pitch_pid = PID_controller(
    config.PITCH_KP, config.PITCH_KI, config.PITCH_KD, config.PITCH_KFF,
    config.PITCH_INTEGRATION_LIMIT, 1 / freq_attitude, freq_attitude, 0, False
)
yaw_pid = PID_controller(
    config.YAW_KP, config.YAW_KI, config.YAW_KD, config.YAW_KFF,
    config.YAW_INTEGRATION_LIMIT, 1 / freq_attitude, freq_attitude, 0, False
)

# Instantiate PID attitude rate controllers
rollrate_pid = PID_controller(
    config.ROLLRATE_KP, config.ROLLRATE_KI, config.ROLLRATE_KD, config.ROLLRATE_KFF,
    config.ROLLRATE_INTEGRATION_LIMIT, 1 / freq_attitude_rate, freq_attitude_rate,
    config.OMX_FILT_CUT, True
)
pitchrate_pid = PID_controller(
    config.PITCHRATE_KP, config.PITCHRATE_KI, config.PITCHRATE_KD, config.PITCHRATE_KFF,
    config.PITCHRATE_INTEGRATION_LIMIT, 1 / freq_attitude_rate, freq_attitude_rate,
    config.OMY_FILT_CUT, True
)
yawrate_pid = PID_controller(
    config.YAWRATE_KP, config.YAWRATE_KI, config.YAWRATE_KD, config.YAWRATE_KFF,
    config.YAWRATE_INTEGRATION_LIMIT, 1 / freq_attitude_rate, freq_attitude_rate,
    config.OMZ_FILT_CUT, True, 32767.0
)

# Instantiate the open loop model
# flapper_model = FlapperModel(freq_attitude)

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
    p, q, r = rates
    ax, ay, az = acc

    qx, qy, qz, qw = sensfusion.sensfusion6Update(p, -q, -r, ax, ay, az, dt)

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
    cmd_pitch_i = pitchrate_pid.compute(gy, pitchrate_sp, False)
    cmd_yaw_i = yawrate_pid.compute(gz, yawrate_sp, False)

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
        attitude_desired["roll"] = setpoints["roll"]
        attitude_desired["pitch"] = setpoints["pitch"]
        attitude_desired["yaw"] = setpoints["yaw"]

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
        rates = data.loc[i, [f"{prefix_data}p", f"{prefix_data}q", f"{prefix_data}r"]].to_numpy().T
        acc = data.loc[i, [f"{prefix_data}acc.x", f"{prefix_data}acc.y", f"{prefix_data}acc.z"]].to_numpy().T
        
        # Calculate estimated attitude through Mahony filter
        attitude = state_estimation(rates, acc, dt)
    

    setpoints = {"roll": data.loc[i, f"{prefix_data}controller.roll"], "pitch": data.loc[i, f"{prefix_data}controller.pitch"], "yaw": data.loc[i, f"{prefix_data}controller.yaw"], "yawrate": data.loc[i, f"{prefix_data}controller.yawRate"]}
    cmd_thrust = data.loc[i, f"{prefix_data}controller.cmd_thrust"]

    # Run the PID cascade
    cmd_roll_i, cmd_pitch_i, cmd_yaw_i = controller_pid(attitude, rates, setpoints, dt, config.YAW_MAX_DELTA, yaw_mode="manual")

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
    data = load_data(config.PLATFORM)

    # Load onboard data
    if config.USE_OPEN_LOOP:
        print("[bold red]I'm still implementing this feature![/bold red]")
        exit()
    else:
        print("[bold thistle1]Running the controllers with the recorded data, here no open loop models are run. The data recorded from the IMU gets fed back to the controllers. [/bold thistle1]")
        
    

    print("[bold green]Starting the simulation[/bold green]")
    for i in track(range(len(data)), description="Processing..."):
        simulate_flapper(data, i, 1 / freq_attitude, config.USE_OPEN_LOOP)

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
