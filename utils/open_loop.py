import numpy as np
from scipy.signal import lti
from . import config

g0 = 9.80665  # m/s^2

class DynamicModel:
    def __init__(self, dt, inertia, mass, model_coeffs, thrust_coeffs, flapper_dims, tf_coeffs, max_pwm, mid_pwm, min_pwm, max_act_state,):
        """Assume everything is given in standard SI units"""

        self.Ixx = inertia["Ixx"]
        self.Iyy = inertia["Iyy"]
        self.Izz = inertia["Izz"]

        self.m = mass

        self.dt = dt

        self.c1 = thrust_coeffs['c1']
        self.c2 = thrust_coeffs['c2']

        self.k_xu, self.k_yv, self.k_zw, self.k_N = model_coeffs['k_xu'], model_coeffs['k_yv'], model_coeffs['k_zw'], model_coeffs['k_N']

        self.lw, self.lz, self.lk, self.ly, self.R, self.l_hinge = flapper_dims['lw'], flapper_dims['lz'], flapper_dims['lk'], flapper_dims['ly'], flapper_dims['R'], flapper_dims['l_hinge']

        self.flapping_max, self.dihedral_max, self.yaw_max = max_act_state["flapping_max"], max_act_state["dihedral_max"], max_act_state["yaw_max"]

        self.max_pwm_m1, self.max_pwm_m2, self.max_pwm_m3, self.max_pwm_m4 =  max_pwm["m1"], max_pwm["m2"], max_pwm["m3"], max_pwm["m4"]

        self.min_pwm_m1, self.min_pwm_m2, self.min_pwm_m3, self.min_pwm_m4 =  min_pwm["m1"], min_pwm["m2"], min_pwm["m3"], min_pwm["m4"]

        self.mid_pwm_m1, self.mid_pwm_m2, self.mid_pwm_m3, self.mid_pwm_m4 =  mid_pwm["m1"], mid_pwm["m2"], mid_pwm["m3"], mid_pwm["m4"]

        tau_flapping, omega_dihedral, zeta_dihedral, omega_yaw, zeta_yaw = (tf_coeffs["tau_flapping"], tf_coeffs["omega_dihedral"], tf_coeffs["zeta_dihedral"], tf_coeffs["omega_yaw"], tf_coeffs["zeta_yaw"])

        self.ss_flapping, self.ss_dihedral, self.ss_yaw, x_freq_left, x_freq_right, x_dihedral, x_yaw = self.init_state_space(tau_flapping, omega_dihedral, zeta_dihedral, omega_yaw, zeta_yaw)

        # state vector for the flapper
        self.flapper_state = {"phi": 0, "theta": 0, "psi": 0, "p": 0, "q": 0, "r": 0, "u": 0, "v": 0, "w": 0, "freq_left": 0, "freq_right": 0, "dihedral": 0, "yaw_actuator_angle": 0}

        # state vectors for the actuators, they are all initialised at zero
        self.actuator_state = {'x_freq_left': x_freq_left, 'x_freq_right': x_freq_right, 'x_dihedral': x_dihedral, 'x_yaw': x_yaw}

    def calculate_accelerations(self, pwm_signals):

        dihedral, freq_left, yaw_angle, freq_right = self.compute_control_states(pwm_signals['m1'], pwm_signals['m2'], pwm_signals['m3'], pwm_signals['m4'])
    
    
        thrust_left = self.compute_thrust(freq_left)
        thrust_right = self.compute_thrust(freq_right)

        u, v, w = self.flapper_state["u"], self.flapper_state["v"], self.flapper_state["w"]
        p, q, r = self.flapper_state["p"], self.flapper_state["q"], self.flapper_state["r"]
        phi, theta, _ = self.flapper_state["phi"], self.flapper_state["theta"], self.flapper_state["psi"]

        ld = self.lw * np.sin(dihedral)

        ld_dot = 0 # np.gradient(ld)

        alpha_L = np.round(np.arcsin((- self.l_hinge * np.sin(dihedral) + self.ly*np.sin(yaw_angle)) / self.lk), 7)
        alpha_R = np.round(np.arcsin((- self.l_hinge * np.sin(dihedral) - self.ly*np.sin(yaw_angle)) / self.lk), 7)

        # Full model

        # Z_L = freq_left  * (w - ld* q + self.lw * np.cos(dihedral) * p)
        # Z_R = freq_right * (w - ld * q - self.lw * np.cos(dihedral) * p)

        # X = - self.k_xu * (freq_left + freq_right) * (u - self.lz * q + self.lw * ld_dot) - thrust_left * np.sin(alpha_L) - thrust_right * np.sin(alpha_R)
        # Y = - self.k_yv * (freq_left + freq_right) * (v + self.lz * p)
        # Z = - self.k_zw * (Z_L + Z_R) - (thrust_right*np.cos(alpha_R) + thrust_left * np.cos(alpha_L))
        # L = - self.k_zw * (Z_L - Z_R) * self.lw * np.cos(dihedral) + Y * self.lz + self.lw*np.cos(dihedral)*(thrust_left*np.cos(alpha_L) - thrust_right*np.cos(alpha_R))
        # M = - self.k_xu * (freq_left + freq_right) * (u - self.lz * q + self.lw * ld_dot) * self.lz + (thrust_left * np.sin(alpha_L) + thrust_right * np.sin(alpha_R)) * self.lz + self.k_zw*(Z_L + Z_R) * ld + (thrust_right * np.cos(alpha_R) + thrust_left * np.cos(alpha_L))*ld
        # N = -self.k_N * ((freq_left + freq_right) * self.R * r + (freq_left - freq_right)*u + (freq_left + freq_right) * dihedral*v) + self.lw * np.cos(dihedral)*(thrust_right * np.sin(alpha_R) - thrust_left * np.cos(alpha_L))

        

        Z_L = freq_left  * (w - ld* q + self.lw * np.cos(dihedral) * p)
        Z_R = freq_right * (w - ld* q - self.lw * np.cos(dihedral) * p)


        X = - self.k_xu * (freq_left + freq_right) * (u - self.lz * q + self.lw * ld_dot) - thrust_left * np.sin(alpha_L) - thrust_right * np.sin(alpha_R)
        Y = - self.k_yv * (freq_left + freq_right) * (v + self.lz * p) 
        Z = - self.k_zw * (Z_L + Z_R) - (thrust_right*np.cos(alpha_R) + thrust_left * np.cos(alpha_L))

        L = (
                    (self.k_zw * (Z_L - Z_R) * self.lw * np.cos(dihedral))
                    + (Y * self.lz)
                    + (self.lw * np.cos(dihedral) * (thrust_left - thrust_right))
                )
        
        M = X * self.lz + ld * self.k_zw * (Z_L + Z_R) + (thrust_right * np.cos(alpha_R) + thrust_left * np.cos(alpha_L)) * ld + (thrust_left * np.sin(alpha_L) + thrust_right * np.sin(alpha_R)) * self.lz 

        N = - self.k_N * ((freq_left + freq_right) * self.R * r + (freq_left - freq_right)*u + (freq_left + freq_right) * dihedral*v) # + self.lw * np.cos(dihedral)*(thrust_right * np.sin(alpha_R) - thrust_left * np.cos(alpha_L))

        # Newton-Euler equations of motion from 'A Mathematical Introduction to Robotic Manipulation'
        u_dot = -(w * q - v * r) + X / self.m + g0 * np.sin(theta)
        v_dot = -(u * r - w * p) + Y / self.m - g0 * np.cos(theta) * np.sin(phi)
        w_dot = -(v * p - u * q) + Z / self.m - g0 * np.cos(theta) * np.cos(phi)
        p_dot = ((self.Iyy - self.Izz) * q * r + L) / self.Ixx
        q_dot = ((self.Izz - self.Ixx) * p * r + M) / self.Iyy
        r_dot = ((self.Ixx - self.Iyy) * p * q + N) / self.Izz
        accelerations = {"u_dot": u_dot, "v_dot": v_dot, "w_dot": w_dot, "p_dot": p_dot, "q_dot": q_dot, "r_dot": r_dot}

        return accelerations, dihedral, freq_left, yaw_angle, freq_right

    def compute_thrust(self, f):
        return self.c1 * f + self.c2

    def update(self, pwm_signals):
        phi, theta, _ = self.flapper_state["phi"], self.flapper_state["theta"], self.flapper_state["psi"]

        accelerations,  dihedral, freq_left, yaw_angle, freq_right = self.calculate_accelerations(pwm_signals)

        # Here integrate the values obtined from the equations of motion
        self.flapper_state["u"] += accelerations["u_dot"] * self.dt
        self.flapper_state["v"] += accelerations["v_dot"] * self.dt
        self.flapper_state["w"] += accelerations["w_dot"] * self.dt
        self.flapper_state["p"] += accelerations["p_dot"] * self.dt
        self.flapper_state["q"] += accelerations["q_dot"] * self.dt
        self.flapper_state["r"] += accelerations["r_dot"] * self.dt

        phi_dot = self.flapper_state["p"] + self.flapper_state["q"] * np.sin(phi) * np.tan(theta) + self.flapper_state["r"] * np.cos(phi) * np.tan(theta)
        theta_dot = self.flapper_state["q"] * np.cos(phi) - self.flapper_state["r"] * np.sin(phi)
        psi_dot = self.flapper_state["q"] * np.sin(phi) / np.cos(theta) + self.flapper_state["r"] * np.cos(phi) / np.cos(theta)

        self.flapper_state["phi"] += phi_dot * self.dt
        self.flapper_state["theta"] += theta_dot * self.dt
        self.flapper_state["psi"] += psi_dot * self.dt

        rates =  np.array([float(self.flapper_state['p']), float(self.flapper_state['q']), float(self.flapper_state['r'])])

        alphas =  np.array([float(accelerations['p_dot']), float(accelerations['q_dot']), float(accelerations['r_dot'])])

        attitude = np.array([float(self.flapper_state['phi']), float(self.flapper_state['theta']), float(self.flapper_state['r'])])

        accelerations = np.array([float(accelerations["u_dot"]), float(accelerations["v_dot"]), float(accelerations["w_dot"])])

        velocity = np.array([float(self.flapper_state['u']), float(self.flapper_state['v']), float(self.flapper_state['w'])])

        # return whatever is needed
        return attitude, rates, alphas, velocity, accelerations, dihedral, freq_left, yaw_angle, freq_right

    
    def init_state_space(self, tau_flapping, omega_dihedral, zeta_dihedral, omega_yaw, zeta_yaw):
        
        num_flapping = [1]
        den_flapping = [tau_flapping, 1]

        sys_flapping = lti(num_flapping, den_flapping).to_ss()

        A_flapping, B_flapping, C_flapping, D_flapping = sys_flapping.A, sys_flapping.B, sys_flapping.C, sys_flapping.D
        
        num_dihedral = [omega_dihedral**2]
        den_dihedral = [1, 2*zeta_dihedral*omega_dihedral, omega_dihedral**2]

        sys_dihedral = lti(num_dihedral, den_dihedral).to_ss()

        A_dihedral, B_dihedral, C_dihedral, D_dihedral = sys_dihedral.A, sys_dihedral.B, sys_dihedral.C, sys_dihedral.D

        num_yaw = [omega_yaw**2]
        den_yaw = [1, 2*zeta_yaw*omega_yaw, omega_yaw**2]

        sys_yaw = lti(num_yaw, den_yaw).to_ss()

        A_yaw, B_yaw, C_yaw, D_yaw = sys_yaw.A, sys_yaw.B, sys_yaw.C, sys_yaw.D

        ss_flapping = {'A': A_flapping, 'B': B_flapping, 'C': C_flapping, 'D': D_flapping}
        ss_dihedral = {'A': A_dihedral, 'B': B_dihedral, 'C': C_dihedral, 'D': D_dihedral}
        ss_yaw = {'A': A_yaw, 'B': B_yaw, 'C': C_yaw, 'D': D_yaw}

        # initialise state vectors for each single servo/motor
        x_freq_left = np.zeros(A_flapping.shape[0])
        x_freq_right = np.zeros(A_flapping.shape[0])
        x_dihedral = np.zeros(A_dihedral.shape[0])
        x_yaw = np.zeros(A_yaw.shape[0])

        return ss_flapping, ss_dihedral, ss_yaw, x_freq_left, x_freq_right, x_dihedral, x_yaw

    def compute_control_states(self, pwm_m1, pwm_m2, pwm_m3, pwm_m4):
        """
        Model the servos and motors, through different transfer functions to find the control inputs.

        m2: left flapping motor
        m4: right flapping motor
        m1: pitch servo
        m3: yaw servo

        """

        dihedral_control, freq_left_control, yaw_control, freq_right_control = self.pwm_to_angle(pwm_m1, pwm_m2, pwm_m3, pwm_m4)

        x_dihedral_new, y_dihedral = self.step_system(self.actuator_state['x_dihedral'], self.ss_dihedral, dihedral_control, self.dt)
            
        x_freq_left_new, y_freq_left = self.step_system(self.actuator_state['x_freq_left'], self.ss_flapping, freq_left_control, self.dt)

        x_yaw_new, y_yaw = self.step_system(self.actuator_state['x_yaw'], self.ss_yaw, yaw_control, self.dt)

        x_freq_right_new, y_freq_right = self.step_system(self.actuator_state['x_freq_right'], self.ss_flapping, freq_right_control, self.dt)

        self.actuator_state['x_dihedral'] = x_dihedral_new

        self.actuator_state['x_freq_left'] = x_freq_left_new

        self.actuator_state['x_yaw'] = x_yaw_new

        self.actuator_state['x_freq_right'] = x_freq_right_new
        
        return y_dihedral, y_freq_left, y_yaw, y_freq_right

    def pwm_to_angle(self, pwm_m1, pwm_m2, pwm_m3, pwm_m4): 

        dihedral_control = self.compute_control_inputs(pwm_m1, self.min_pwm_m1, self.mid_pwm_m1, self.max_pwm_m1, self.dihedral_max)

        freq_left_control = config.PWM_TO_FREQ['m'] * pwm_m2 + config.PWM_TO_FREQ['c']

        yaw_control = self.compute_control_inputs(pwm_m3, self.min_pwm_m3, self.mid_pwm_m3, self.max_pwm_m3, self.yaw_max)

        freq_right_control = config.PWM_TO_FREQ['m'] * pwm_m4 + config.PWM_TO_FREQ['c']

        return dihedral_control, freq_left_control, yaw_control, freq_right_control
    
    def compute_control_inputs(self, pwm, min_pwm, mid_pwm, max_pwm, max_state):
        if pwm >= mid_pwm:
            state_value = abs((pwm - mid_pwm) / (max_pwm - mid_pwm) * max_state)
        else:
            state_value = abs((pwm - mid_pwm) / (mid_pwm - min_pwm) * max_state)
        
        state_value = max(-max_state, min(state_value, max_state))
        return state_value
    
    def step_system(self, x, ss, u, dt):
        u = np.atleast_1d(u)
        y = ss['C'] @ x + ss['D'] @ u
        x_new = x + dt * (ss['A'] @ x + ss['B'] @ u)
        return x_new, y



