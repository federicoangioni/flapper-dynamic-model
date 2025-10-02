import matplotlib.pyplot as plt
from time import time
from scipy.spatial.transform import Rotation as R
import numpy as np
from rich import print
from rich.progress import track

# Local imports
import utils.config as config
from utils.transform_to_global import to_global
from utils.controller import PID_controller
from utils.state_estimator import MahonyIMU
from utils.power_distribution import power_distribution
from utils.open_loop import FlapperModel
from utils.data_loader import load_data


show = True
flight_exp = "flight_002"


prefix_data = ""

class Simulation():
    def __init__(self):

        self.cmd_roll = []
        self.cmd_pitch = []
        self.cmd_yaw = []

        # Output of the model
        self.phi = []
        self.theta = []
        self.psi = []
        self.freqs = []


        # Handle YAW rate controllers
        self.yawrate_sp = []

        # Pre-define the necessary dictionaries for the PID and power_distribution
        self.attitude_measured = {"roll": 0, "pitch": 0, "yaw": 0}
        self.attitude_desired = {"roll": 0, "pitch": 0, "yaw": 0}
        self.attituderate_desired = {"rollrate": 0, "pitchrate": 0, "yawrate": 0}
        self.controls = {"thrust": 0, "roll": 0, "pitch": 0, "yaw": 0}
        self.motors_list = {"m1": [config.MID_PWM['m1']], "m2": [0], "m3": [config.MID_PWM['m3']], "m4": [0]}

        self.current_pos = np.array([[0.0], [0.0], [0.0]])

        self.pos_list = [[], [], []]

        self.accelerations_list = {'u_dot': [], 'v_dot': [], 'w_dot': []}

        # Instantiate the sensor fusion filter
        self.sensfusion = MahonyIMU()

        # Instantiate PID attitude controllers
        self.roll_pid = PID_controller(
            config.ROLL_KP, config.ROLL_KI, config.ROLL_KD, config.ROLL_KFF,
            config.ROLL_INTEGRATION_LIMIT, 1 / config.FREQ_ATTITUDE, config.FREQ_ATTITUDE, 0, False
        )
        self.pitch_pid = PID_controller(
            config.PITCH_KP, config.PITCH_KI, config.PITCH_KD, config.PITCH_KFF,
            config.PITCH_INTEGRATION_LIMIT, 1 / config.FREQ_ATTITUDE, config.FREQ_ATTITUDE, 0, False
        )
        self.yaw_pid = PID_controller(
            config.YAW_KP, config.YAW_KI, config.YAW_KD, config.YAW_KFF,
            config.YAW_INTEGRATION_LIMIT, 1 / config.FREQ_ATTITUDE, config.FREQ_ATTITUDE, 0, False
        )

        # Instantiate PID attitude rate controllers
        self.rollrate_pid = PID_controller(
            config.ROLLRATE_KP, config.ROLLRATE_KI, config.ROLLRATE_KD, config.ROLLRATE_KFF,
            config.ROLLRATE_INTEGRATION_LIMIT, 1 / config.FREQ_ATTITUDE_RATE, config.FREQ_ATTITUDE_RATE,
            config.OMX_FILT_CUT, True
        )
        self.pitchrate_pid = PID_controller(
            config.PITCHRATE_KP, config.PITCHRATE_KI, config.PITCHRATE_KD, config.PITCHRATE_KFF,
            config.PITCHRATE_INTEGRATION_LIMIT, 1 / config.FREQ_ATTITUDE_RATE, config.FREQ_ATTITUDE_RATE,
            config.OMY_FILT_CUT, True
        )
        self.yawrate_pid = PID_controller(
            config.YAWRATE_KP, config.YAWRATE_KI, config.YAWRATE_KD, config.YAWRATE_KFF,
            config.YAWRATE_INTEGRATION_LIMIT, 1 / config.FREQ_ATTITUDE_RATE, config.FREQ_ATTITUDE_RATE,
            config.OMZ_FILT_CUT, True, 32767.0
        )

        # Instantiate the open loop model
        self.Flapper = FlapperModel(1 / config.FREQ_ATTITUDE, config.MMOI_WITH_WINGS_XY, config.MASS_WINGS, config.MODEL_COEFFS, 
                            config.THRUST_COEFFS, config.FLAPPER_DIMS, config.TF_COEFFS, config.MAX_PWM, config.MID_PWM, config.MIN_PWM, 
                            config.MAX_ACT_STATE)

    def capAngle(self, angle):
        result = angle
        while result > 180.0:
            result -= 360.0
        while result < -180.0:
            result += 360.0
        return result


    def state_estimation(self, rates, acc, dt):
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

        qx, qy, qz, qw = self.sensfusion.sensfusion6Update(p, -q, -r, ax, ay, az, dt)

        yaw_i, pitch_i, roll_i = np.degrees(R.from_quat([qx, qy, qz, qw]).as_euler("ZYX"))

        return roll_i, -pitch_i, yaw_i


    def attitude_controller(self, roll_i, pitch_i, yaw_i, roll_sp_i, pitch_sp_i, yaw_sp_i):
        controller_rollrate_i = self.roll_pid.compute(roll_i, roll_sp_i, False)
        controller_pitchrate_i = self.pitch_pid.compute(pitch_i, pitch_sp_i, False)
        controller_yawrate_i = self.yaw_pid.compute(yaw_i, yaw_sp_i, True)

        return controller_rollrate_i, controller_pitchrate_i, controller_yawrate_i


    def rate_controller(self, attitude_rate_measured, rollrate_sp, pitchrate_sp, yawrate_sp):
        gx, gy, gz = attitude_rate_measured

        cmd_roll_i = self.rollrate_pid.compute(gx, rollrate_sp, False)
        cmd_pitch_i = self.pitchrate_pid.compute(gy, pitchrate_sp, False)
        cmd_yaw_i = self.yawrate_pid.compute(gz, yawrate_sp, False)

        return cmd_roll_i, cmd_pitch_i, cmd_yaw_i


    def controller_pid(self, attitude, rates, setpoints, dt_imu, yaw_max_delta, yaw_mode="velocity"):

        self.attitude_measured["roll"], self.attitude_measured["pitch"], self.attitude_measured["yaw"] = attitude

        if yaw_mode == "velocity":
            self.attitude_desired["yaw"] = self.capAngle(self.attitude_desired["yaw"] + setpoints["yawrate"] * dt_imu)

            if yaw_max_delta != 0.0:
                delta = self.capAngle(self.attitude_desired["yaw"] - self.attitude_measured["yaw"])

                if delta > yaw_max_delta:
                    self.attitude_desired["yaw"] = self.attitude_measured["yaw"] + yaw_max_delta
                elif delta < -yaw_max_delta:
                    self.attitude_desired["yaw"] = self.attitude_measured["yaw"] - yaw_max_delta

            self.attitude_desired["roll"] = setpoints["roll"]
            self.attitude_desired["pitch"] = setpoints["pitch"]
            self.attitude_desired["yaw"] = self.capAngle(self.attitude_desired["yaw"])

        elif yaw_mode == "manual":
            self.attitude_desired["roll"] = setpoints["roll"]
            self.attitude_desired["pitch"] = setpoints["pitch"]
            self.attitude_desired["yaw"] = setpoints["yaw"]

        controller_rate_sp = self.attitude_controller(
            self.attitude_measured["roll"], self.attitude_measured["pitch"], self.attitude_measured["yaw"], self.attitude_desired["roll"], self.attitude_desired["pitch"], self.attitude_desired["yaw"]
        )

        self.attituderate_desired["rollrate"] = controller_rate_sp[0]
        self.attituderate_desired["pitchrate"] = controller_rate_sp[1]
        self.attituderate_desired["yawrate"] = controller_rate_sp[2]

        cmd_roll_i, cmd_pitch_i, cmd_yaw_i = self.rate_controller(rates, self.attituderate_desired["rollrate"], self.attituderate_desired["pitchrate"], self.attituderate_desired["yawrate"])

        return cmd_roll_i, cmd_pitch_i, cmd_yaw_i


    def simulate_flapper(self, dt, setpoints, cmd_thrust, use_model : bool, data = None, i = None,):

        if use_model:
            # Use previous motor command
            pwm_m1 = self.motors_list["m1"][-1]
            pwm_m2 = self.motors_list["m2"][-1]
            pwm_m3 = self.motors_list["m3"][-1]
            pwm_m4 = self.motors_list["m4"][-1]

            pwm_signals = {'m1': pwm_m1, 'm2':pwm_m2, 'm3':pwm_m3, 'm4':pwm_m4}

            attitude, rates, velocity, accelerations, dihedral = self.Flapper.update(pwm_signals)

            rates = np.degrees(rates[0]), np.degrees(rates[1]), np.degrees(rates[2])
            attitude = np.degrees(attitude[0]), np.degrees(attitude[1]), np.degrees(attitude[2])
            
            self.accelerations_list['u_dot'].append(accelerations[0])
            self.accelerations_list['v_dot'].append(accelerations[1])
            self.accelerations_list['w_dot'].append(accelerations[2])
            self.current_dihedral = dihedral
            self.current_attitude = attitude
            self.current_acceleration = accelerations
            
            self.current_pos, attitude = to_global(self.current_pos, dt, accelerations, velocity, 0, rates, attitude)

        else:
            # Fetch data from onboard (unprocessed, for now) .csv
            rates = data.loc[i, [f"{prefix_data}p", f"{prefix_data}q", f"{prefix_data}r"]].to_numpy().T
            acc = data.loc[i, [f"{prefix_data}acc.x", f"{prefix_data}acc.y", f"{prefix_data}acc.z"]].to_numpy().T
            
            # Calculate estimated attitude through Mahony filter
            attitude = self.state_estimation(rates, acc, dt)
        
        self.phi.append(attitude[0])
        self.theta.append(attitude[1])
        self.psi.append(attitude[2])

        # Run the PID cascade, input to this must be in degrees
        cmd_roll_i, cmd_pitch_i, cmd_yaw_i = self.controller_pid(attitude, rates, setpoints, dt, config.YAW_MAX_DELTA, yaw_mode="manual")
        self.controls["thrust"] = cmd_thrust
        self.controls["pitch"] = cmd_pitch_i
        self.controls["roll"] = cmd_roll_i
        self.controls["yaw"] = -cmd_yaw_i

        motors = power_distribution(self.controls)
        # For now save only the cmd outputs and the relative motor outputss
        self.cmd_roll.append(cmd_roll_i)
        self.cmd_pitch.append(cmd_pitch_i)
        self.cmd_yaw.append(-cmd_yaw_i)

        self.motors_list["m1"].append(motors["m1"])
        self.motors_list["m2"].append(motors["m2"])
        self.motors_list["m3"].append(motors["m3"])
        self.motors_list["m4"].append(motors["m4"])


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
        print("[bold thistle1]Running the modeled open loop.[/bold thistle1]")
    else:
        print("[bold thistle1]Running the controllers with the recorded data, here no open loop models are run. The data recorded from the IMU gets fed back to the controllers. [/bold thistle1]")
    
    simulation = Simulation()

    print("[bold green]Starting the simulation[/bold green]")

    if config.USE_OPEN_LOOP:
        for i in track(range(len(data)), description="Processing..."):
            setpoints = {"roll": 0, "pitch": -40, "yaw": 0, "yawrate": 0}

            cmd_thrust = 23000 
            
            simulation.simulate_flapper(1 / config.FREQ_ATTITUDE, setpoints, cmd_thrust, config.USE_OPEN_LOOP,)

    else:
        for i in track(range(len(data)), description="Processing..."):
            setpoints = {"roll": data.loc[i, f"{prefix_data}controller.roll"], "pitch": data.loc[i, f"{prefix_data}controller.pitch"], "yaw": data.loc[i, f"{prefix_data}controller.yaw"], "yawrate": data.loc[i, f"{prefix_data}controller.yawRate"]}

            cmd_thrust = data.loc[i, f"{prefix_data}controller.cmd_thrust"]
            simulation.simulate_flapper(1 / config.FREQ_ATTITUDE, setpoints, cmd_thrust, config.USE_OPEN_LOOP, data, i)

    end = time()

    print(f"[magenta]Process run in {round(end - start, 3)} s[/magenta]")

    if show:
        print("Showing the outputs in plots")

        # First figure
        fig1, axs1 = plt.subplots(nrows=4, ncols=1)

        axs1[0].set_title("Pitch angle command from rate PID")
        axs1[0].plot(simulation.cmd_pitch, label="simulated")
        axs1[0].plot(data[f"{prefix_data}controller.cmd_pitch"], alpha=0.5, label="recorded")
        axs1[0].legend()

        axs1[1].set_title("Roll angle command from rate PID")
        axs1[1].plot(simulation.cmd_roll)
        axs1[1].plot(data[f"{prefix_data}controller.cmd_roll"], alpha=0.5)

        axs1[2].set_title("Yaw angle command from rate PID")
        axs1[2].plot(simulation.cmd_yaw)
        axs1[2].plot(data[f"{prefix_data}controller.cmd_yaw"], alpha=0.5)
        axs1[2].set_ylabel("time (s)")
        axs1[2].set_ylim(-32767, 32767)

        axs1[3].set_title("CMD thrust directly from controller")
        axs1[3].plot(data[f"{prefix_data}controller.cmd_thrust"], alpha=0.5)
        axs1[3].set_ylabel("time (s)")

        plt.tight_layout()

        # Second figure
        fig2, axs2 = plt.subplots(nrows=4, ncols=1)

        axs2[0].set_title("motor commands m1")
        axs2[0].plot(simulation.motors_list["m1"], label="simulated")
        axs2[0].plot(data[f"{prefix_data}motor.m1"], label="recorded", alpha=0.5)

        axs2[1].set_title("motor commands m2")
        axs2[1].plot(simulation.motors_list["m2"])
        axs2[1].plot(data[f"{prefix_data}motor.m2"], alpha=0.5)

        axs2[2].set_title("motor commands m3")
        axs2[2].plot(simulation.motors_list["m3"])
        axs2[2].plot(data[f"{prefix_data}motor.m3"], alpha=0.5)

        axs2[3].set_title("motor commands m4")
        axs2[3].plot(simulation.motors_list["m4"])
        axs2[3].plot(data[f"{prefix_data}motor.m4"], alpha=0.5)

        plt.tight_layout()

        # Third figure, plot angles, rates and velocities
        fig3, axs3 = plt.subplots(nrows=4, ncols=1)

        axs3[0].set_title("Pitch")

        axs3[0].plot(simulation.theta, label="simulated")

        #axs3[0].plot(np.degrees(processed["onboard.pitch"]), label="recorded")
        axs3[0].set_ylim(-20, 20)
        axs3[0].legend()

        axs3[1].set_title("Roll")
        # axs3[1].plot(np.degrees(processed["onboard.roll"]))
        axs3[1].plot(simulation.phi)

        axs3[2].set_title("Yaw")
        axs3[2].plot()
        axs3[2].plot(simulation.psi)

        axs3[3].set_title("Freq left")
        axs3[3].plot()
        axs3[3].plot(simulation.freqs)

        plt.tight_layout()
        plt.show()
