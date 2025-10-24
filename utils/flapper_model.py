from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd

# Local imports
import utils.config as config
from utils.controller import PID_controller
from utils.state_estimator import MahonyIMU
from utils.power_distribution import power_distribution, capAngle
from utils.open_loop import FlapperModel



prefix_data = ""

class Simulation():
    def __init__(self, dt, use_open_loop):

        self.dt = dt

        self.use_open_loop = use_open_loop

        self.motors_list = [{"m1": config.MID_PWM['m1'], "m2": 0, "m3": config.MID_PWM['m3'], "m4": 0},]

        self.controls_list = [{"thrust": 0, "roll": 0, "pitch": 0, "yaw": 0},]

        self.flapper_state = [{
                            # attitude
                            "phi": 0.0,
                            "theta": 0.0,
                            "psi": 0.0,

                            # rates
                            "p": 0.0,
                            "q": 0.0,
                            "r": 0.0,

                            # rates derivatives
                            "p_dot": 0.0,
                            "q_dot": 0.0,
                            "r_dot": 0.0,

                            # global position
                            "x_glob": 0.0,
                            "y_glob": 0.0,
                            "z_glob": 0.0,

                            # velocities
                            "vel.x": 0.0,
                            "vel.y": 0.0,
                            "vel.z": 0.0,

                            # accelerations
                            "acc.x": 0.0,
                            "acc.y": 0.0,
                            "acc.z": 0.0,

                            # control inputs
                            "freq.left": 0.0,
                            "freq.right": 0.0,
                            "dihedral": 0.0,
                            "yaw_servo": 0.0
                        }]

        # Pre-define the necessary dictionaries for the PID and power_distribution -> do they actually have to be class wise variables?
        self.attitude_measured = {"roll": 0, "pitch": 0, "yaw": 0}
        self.attitude_desired = {"roll": 0, "pitch": 0, "yaw": 0}


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


    def state_estimation(self, rates, acc):
        p, q, r = rates
        ax, ay, az = acc

        qx, qy, qz, qw = self.sensfusion.sensfusion6Update(p, -q, -r, ax, ay, az, self.dt)

        yaw, pitch, roll = np.degrees(R.from_quat([qx, qy, qz, qw]).as_euler("ZYX"))

        return roll, -pitch, yaw


    def attitude_controllers(self, attitude, attitude_desired):
        roll, pitch, yaw = attitude
        roll_sp, pitch_sp, yaw_sp = attitude_desired

        controller_rollrate = self.roll_pid.compute(roll, roll_sp, False)
        controller_pitchrate = self.pitch_pid.compute(pitch, pitch_sp, False)
        controller_yawrate = self.yaw_pid.compute(yaw, yaw_sp, True)

        return controller_rollrate, controller_pitchrate, controller_yawrate


    def rate_controllers(self, attitude_rate_measured, rate_desired):
        gx, gy, gz = attitude_rate_measured
        rollrate_sp, pitchrate_sp, yawrate_sp = rate_desired
        
        cmd_roll = self.rollrate_pid.compute(gx, rollrate_sp, False)
        cmd_pitch = self.pitchrate_pid.compute(gy, pitchrate_sp, False)
        cmd_yaw = self.yawrate_pid.compute(gz, yawrate_sp, False)

        return cmd_roll, cmd_pitch, cmd_yaw


    def controllers_pid(self, attitude, rates, setpoints, yaw_mode="velocity"):

        self.attitude_measured["roll"], self.attitude_measured["pitch"], self.attitude_measured["yaw"] = attitude

        if yaw_mode == "velocity":
            self.attitude_desired["yaw"] = capAngle(self.attitude_desired["yaw"] + setpoints["yawrate"] * self.dt)

            if config.YAW_MAX_DELTA != 0.0:
                delta = capAngle(self.attitude_desired["yaw"] - self.attitude_measured["yaw"])

                if delta > config.YAW_MAX_DELTA:
                    self.attitude_desired["yaw"] = self.attitude_measured["yaw"] + config.YAW_MAX_DELTA
                elif delta < -config.YAW_MAX_DELTA:
                    self.attitude_desired["yaw"] = self.attitude_measured["yaw"] - config.YAW_MAX_DELTA

            self.attitude_desired["roll"] = setpoints["roll"]
            self.attitude_desired["pitch"] = setpoints["pitch"]
            self.attitude_desired["yaw"] = capAngle(self.attitude_desired["yaw"])

        elif yaw_mode == "manual":
            self.attitude_desired["roll"] = setpoints["roll"]
            self.attitude_desired["pitch"] = setpoints["pitch"]
            self.attitude_desired["yaw"] = setpoints["yaw"]

        attitude_desired = np.array([self.attitude_desired["roll"], self.attitude_desired["pitch"], self.attitude_desired["yaw"]])

        rate_desired = self.attitude_controllers(attitude, attitude_desired)

        cmd_roll_i, cmd_pitch_i, cmd_yaw_i = self.rate_controllers(rates, rate_desired)

        return cmd_roll_i, cmd_pitch_i, cmd_yaw_i


    def simulate_flapper(self, setpoints, cmd_thrust_i, i, data = None):

        if self.use_open_loop:
            # Use previous motor command

            attitude, rates, alphas, velocity, accelerations, dihedral, freq_left, yaw_servo, freq_right = self.Flapper.update(self.motors_list[-1])

            rates = np.degrees(rates)
            attitude = np.degrees(attitude)
            
            self.flapper_state.append({
                                    # attitude (phi, theta, psi)
                                    "phi": attitude[0],
                                    "theta": attitude[1],
                                    "psi": attitude[2],

                                    # rates (p, q, r)
                                    "p": rates[0],
                                    "q": rates[1],
                                    "r": rates[2],

                                    # rates derivatives 
                                    "p_dot": alphas[0],
                                    "q_dot": alphas[1],
                                    "r_dot": alphas[2],

                                    # global position
                                    "x_glob": 0,
                                    "y_glob": 0,
                                    "z_glob": 0,

                                    # velocities
                                    "vel.x": velocity[0],
                                    "vel.y": velocity[1],
                                    "vel.z": velocity[2],

                                    # accelerations
                                    "acc.x": accelerations[0],
                                    "acc.y": accelerations[1],
                                    "acc.z": accelerations[2],

                                    # control inputs
                                    "freq.left": freq_left,
                                    "freq.right": freq_right,
                                    "dihedral": dihedral,
                                    "yaw_servo": yaw_servo
                                })



        else:
            # Fetch data from onboard (unprocessed, for now) .csv
            rates = data.loc[i, ["onboard.p", "onboard.q", "onboard.r"]].to_numpy().T
            acc = data.loc[i, ["onboard.acc.x", "onboard.acc.y", "onboard.acc.z"]].to_numpy().T
            
            # Calculate estimated attitude through Mahony filter
            attitude = self.state_estimation(rates, acc)
        
        # Run the PID cascade, input to this must be in degrees, setpoints, attitude and rates
        cmd_roll_i, cmd_pitch_i, cmd_yaw_i = self.controllers_pid(attitude, rates, setpoints, yaw_mode="manual")

        self.controls_list.append({"cmd_thrust": cmd_thrust_i, 
                                   "cmd_roll": cmd_roll_i, 
                                   "cmd_pitch": cmd_pitch_i, 
                                   "cmd_yaw": cmd_yaw_i})

        motors = power_distribution(self.controls_list[-1])

        self.motors_list.append({
                    "m1": motors[0],
                    "m2": motors[1],
                    "m3": motors[2],
                    "m4": motors[3]
                })

    def save_simulation(self):

        '''
        TODO:
        
        write code to translate into global coordinates for rerun visualisation
        '''

        # Dataframes to fill the values at the end
        flapper_state_df = pd.DataFrame(self.flapper_state)
        
        controls_df = pd.DataFrame(self.controls_list)

        motors_df = pd.DataFrame(self.motors_list)


        return flapper_state_df, controls_df, motors_df
        