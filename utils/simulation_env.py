from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd

# Local imports
import utils.config_old as config_old
from utils.controller import PID_controller
from utils.state_estimator import MahonyIMU
from utils.power_distribution import power_distribution, capAngle
from utils.dynamic_model import DynamicModel



prefix_data = ""

class Simulation():
    def __init__(self, dt, use_open_loop):
        self.time_elapsed = 0

        self.dt = dt

        self.use_open_loop = use_open_loop

        self.motors_list = [{"m1": config_old.MID_PWM['m1'], "m2": 0, "m3": config_old.MID_PWM['m3'], "m4": 0},]

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
        self.attitude_desired = np.array([0.0, 0.0, 0.0])

        self.attitude=np.array([0, 0, 0])
        self.rates = np.array([0, 0, 0])

        self.bias_buffer = []
        self.accel_bias = np.array([0.0, 0.0, 0.0])
        self.calibration_samples = 6000

        # Instantiate the sensor fusion filter
        self.sensfusion = MahonyIMU()

        # Instantiate PID attitude controllers
        self.roll_pid = PID_controller(
            config_old.ROLL_KP, config_old.ROLL_KI, config_old.ROLL_KD, config_old.ROLL_KFF,
            config_old.ROLL_INTEGRATION_LIMIT, 1 / config_old.FREQ_ATTITUDE, config_old.FREQ_ATTITUDE, 0, False
        )
        self.pitch_pid = PID_controller(
            config_old.PITCH_KP, config_old.PITCH_KI, config_old.PITCH_KD, config_old.PITCH_KFF,
            config_old.PITCH_INTEGRATION_LIMIT, 1 / config_old.FREQ_ATTITUDE, config_old.FREQ_ATTITUDE, 0, False
        )
        self.yaw_pid = PID_controller(
            config_old.YAW_KP, config_old.YAW_KI, config_old.YAW_KD, config_old.YAW_KFF,
            config_old.YAW_INTEGRATION_LIMIT, 1 / config_old.FREQ_ATTITUDE, config_old.FREQ_ATTITUDE, 0, False
        )

        # Instantiate PID attitude rate controllers
        self.rollrate_pid = PID_controller(
            config_old.ROLLRATE_KP, config_old.ROLLRATE_KI, config_old.ROLLRATE_KD, config_old.ROLLRATE_KFF,
            config_old.ROLLRATE_INTEGRATION_LIMIT, 1 / config_old.FREQ_ATTITUDE_RATE, config_old.FREQ_ATTITUDE_RATE,
            config_old.OMX_FILT_CUT, True
        )
        self.pitchrate_pid = PID_controller(
            config_old.PITCHRATE_KP, config_old.PITCHRATE_KI, config_old.PITCHRATE_KD, config_old.PITCHRATE_KFF,
            config_old.PITCHRATE_INTEGRATION_LIMIT, 1 / config_old.FREQ_ATTITUDE_RATE, config_old.FREQ_ATTITUDE_RATE,
            config_old.OMY_FILT_CUT, True
        )
        self.yawrate_pid = PID_controller(
            config_old.YAWRATE_KP, config_old.YAWRATE_KI, config_old.YAWRATE_KD, config_old.YAWRATE_KFF,
            config_old.YAWRATE_INTEGRATION_LIMIT, 1 / config_old.FREQ_ATTITUDE_RATE, config_old.FREQ_ATTITUDE_RATE,
            config_old.OMZ_FILT_CUT, True, 32767
        )

        # Instantiate the open loop model
        # self.Flapper = DynamicModel(1 / config_old.FREQ_ATTITUDE, config_old.MMOI_WITH_WINGS_XY, config_old.MASS_WINGS, config_old.MODEL_COEFFS, 
        #                     config_old.THRUST_COEFFS, config_old.FLAPPER_DIMS, config_old.TF_COEFFS, config_old.MAX_PWM, config_old.MID_PWM, config_old.MIN_PWM, 
        #                     config_old.MAX_ACT_STATE)


    def state_estimation(self, rates, acc):
        p, q, r = rates
        ax, ay, az = acc
    
        if self.time_elapsed % (1/250):

            self.sensfusion.sensfusion6UpdateQ(p, -q, -r, ax, ay, az, self.dt)

            self.sensfusion.sensfusion6GetEulerRPY()

            az = self.sensfusion.sensfusion6GetAccZWithoutGravity(ax, ay, az)

        return self.sensfusion.roll, self.sensfusion.pitch, self.sensfusion.yaw


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


    def controllers_pid(self, setpoints, yaw_mode="velocity"):

        if yaw_mode == "velocity":
            self.attitude_desired[2] = capAngle(self.attitude_desired[2] + setpoints["yawrate"] * self.dt)

            if config_old.YAW_MAX_DELTA != 0.0:
                delta = capAngle(self.attitude_desired[2] - self.attitude[2])

                if delta > config_old.YAW_MAX_DELTA:
                    self.attitude_desired[2] = self.attitude[2] + config_old.YAW_MAX_DELTA
                elif delta < -config_old.YAW_MAX_DELTA:
                    self.attitude_desired[2] = self.attitude[2] - config_old.YAW_MAX_DELTA

            self.attitude_desired[0] = setpoints["roll"]
            self.attitude_desired[1] = setpoints["pitch"]
            self.attitude_desired[2] = capAngle(self.attitude_desired[2])

        elif yaw_mode == "manual":
            self.attitude_desired[0] = setpoints["roll"]
            self.attitude_desired[1] = setpoints["pitch"]
            self.attitude_desired[2] = setpoints["yaw"]

        rate_desired = self.attitude_controllers(self.attitude, self.attitude_desired)

        cmd_roll_i, cmd_pitch_i, cmd_yaw_i = self.rate_controllers(self.rates, rate_desired)

        return cmd_roll_i, cmd_pitch_i, cmd_yaw_i


    def simulate_flapper(self, setpoints, cmd_thrust_i, i, data = None):

         # Run the PID cascade, input to this must be in degrees, setpoints, attitude and rates
        cmd_roll_i, cmd_pitch_i, cmd_yaw_i = self.controllers_pid(setpoints, yaw_mode="manual")

        self.controls_list.append({"cmd_thrust": cmd_thrust_i, 
                                   "cmd_roll": cmd_roll_i, 
                                   "cmd_pitch": cmd_pitch_i, 
                                   "cmd_yaw": -cmd_yaw_i})

        motors = power_distribution(self.controls_list[-1])

        self.motors_list.append({
                    "m1": motors[0],
                    "m2": motors[1],
                    "m3": motors[2],
                    "m4": motors[3]
                })
        

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
            self.rates = data.loc[i, ["p", "q", "r"]].to_numpy().T
            self.acc = data.loc[i, ["acc.x", "acc.y", "acc.z"]].to_numpy().T

            if i < self.calibration_samples:
                self.bias_buffer.append((self.acc[0], self.acc[1], self.acc[2]))
                self.attitude = [0.0, 0.0, 0.0]
            elif i == self.calibration_samples:
                buf = np.array(self.bias_buffer)
                self.accel_bias = np.array([buf[:, 0].mean(), buf[:, 1].mean(), buf[:, 2].mean() - 1.0])
                self.attitude = [0.0, 0.0, 0.0]
            else:

                # Apply bias correction
                acc_corrected = self.acc - self.accel_bias

                # Run state estimation with corrected accelerometer
                self.attitude = self.state_estimation(self.rates, acc_corrected)
            

            self.flapper_state.append({
                                    # attitude (phi, theta, psi)
                                    "phi": self.attitude[0],
                                    "theta": self.attitude[1],
                                    "psi": self.attitude[2],

                                    # rates (p, q, r)
                                    "p": self.rates[0],
                                    "q": self.rates[1],
                                    "r": self.rates[2],

                                    # rates derivatives 
                                    "p_dot": 0,
                                    "q_dot": 0,
                                    "r_dot": 0,

                                    # global position
                                    "x_glob": 0,
                                    "y_glob": 0,
                                    "z_glob": 0,

                                    # velocities
                                    "vel.x": 0,
                                    "vel.y": 0,
                                    "vel.z": 0,

                                    # accelerations
                                    "acc.x": self.acc[0],
                                    "acc.y": self.acc[1],
                                    "acc.z": self.acc[2],

                                    # control inputs
                                    "freq.left": 0,
                                    "freq.right": 0,
                                    "dihedral": 0,
                                    "yaw_servo": 0
                                })
            
        self.time_elapsed += self.dt
        
       

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
        