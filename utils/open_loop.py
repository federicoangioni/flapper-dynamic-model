import numpy as np
# mass moments of inertia
mmoi_with_wings = {"Ixx": 5.6e-4, "Iyy": 3.4e-4, "Izz": 3.6e-4}  # kg/m^2

mmoi_with_wings_XY = {"Ixx": 5.6e-4, "Iyy": 3.4e-4, "Izz": 2.8e-4}  # kg/m^2 # with wings oriented in XY plane

mmoi_no_wings = {"Ixx": 3.1e-4, "Iyy": 3.0e-4, "Izz": 2.7e-4}  # kg/m^2

mass_with_wings = 102.6  # g

mass_no_wings = 94.3  # g

g0 = 9.80665  # m/s²

# Drag coefficients
damping_coefficients = {"kx": None, "ky": None, "kz": None}

# Thrust coefficients
thrust_coefficients = {"c1": None, "c2": None}

# Distances necessary of the flapper
"""
lw: distance from the z-axis of the flapper to the center of pressure along the y-axis [m]
"""
flapper_parameters = {"lw": None, "lz": None}


class FlapperModel:
    def __init__(self, dt: float, inertia: dict , mass: float, damping_coeffs, thrust_coeffs, flapper_params):
        self.Ixx = inertia["Ixx"]
        self.Iyy = inertia["Iyy"]
        self.Izz = inertia["Izz"]

        self.m = mass / 1e3  # convert to kg

        self.dt = dt

        """Pay attention here it still needs to be implemented"""
        # Define the thrust coeffecients
        self.c1 = thrust_coefficients["c1"]
        self.c2 = thrust_coefficients["c2"]

        self.kx = damping_coeffs["kx"]

        self.lw= flapper_params["lw"]


        # Inizialize state variables and dictionaries that get updated every timestep
        self.state = {"phi": 0, "theta": 0, "psi": 0, "p": 0, "q": 0, "r": 0, "u": 0, "v": 0, "w": 0}

        self.control_inputs = {"freq_left": 0, "freq_right": 0, "dihedral": 0, "yaw_actuator_angle": 0}

    def calculate_accelerations(self, pwm_signals):
        # pwm -> self.control_inputs with func(pwms_to_control)

        u, v, w = self.state["u", "v", "w"]
        p, q, r = self.state["p", "q", "r"]
        phi, theta, psi = self.state["phi", "theta", "psi"]
        freq_left, freq_right, dihedral, yaw_actuator_angle = self.control_inputs["freq_left", "freq_right", "dihedral", "yaw_actuator_angle"]

        # Model assuming CD is constant at a range of angle of attacks

        X = 0
        Y = 0
        Z = 0
        L = 0
        M = 0
        N = 0

        # Newton-Euler equations of motion from 'A Mathematical Introduction to Robotic Manipulation'
        u_dot = -(w * q - v * r) + X / self.m + g0 * np.sin(theta)
        v_dot = -(u * r - w * p) + Y / self.m - g0 * np.cos(theta) * np.sin(phi)
        w_dot = -(v * p - u * q) + Z / self.m - g0 * np.cos(theta) * np.cos(phi)
        p_dot = ((self.Iyy - self.Izz) * q * r + L) / self.Ixx
        q_dot = ((self.Izz - self.Ixx) * p * r + M) / self.Iyy
        r_dot = ((self.Ixx - self.Iyy) * q * q + N) / self.Izz

        accelerations = {"u_dot": u_dot, "v_dot": v_dot, "w_dot": w_dot, "p_dot": p_dot, "q_dot": q_dot, "r_dot": r_dot}

        return accelerations

    def pwms_to_control(self, pwm_signals: dict):
        """
        Model the servos and motors, through different transfer functions to find the control inputs.

        m2: left flapping motor
        m4: right flapping motor
        m1: pitch servo
        m3: yaw servo

        """
        # For now write the pwms to control inputs as in the nimble, this will be changed with future experiments
        motors_m1 = pwm_signals["m1"]
        motors_m2 = pwm_signals["m2"]
        motors_m3 = pwm_signals["m3"]
        motors_m4 = pwm_signals["m4"]

        # Update the control inputs
        self.control_inputs["freq_left"]
        self.control_inputs["freq_right"]
        self.control_inputs["dihedral"]
        self.control_inputs["yaw_actuator_angle"]
        self.ld = self.lw * np.sin(self.control_inputs["dihedral"])

    def compute_thrust(self, f):
        return self.c1 * f + self.c2

    def update(self, accelerations):
        phi, theta, psi = self.state["phi", "theta", "psi"]

        # Here integrate the values obtined from the equations of motion
        self.state["u"] += accelerations["u_dot"] * self.dt
        self.state["v"] += accelerations["v_dot"] * self.dt
        self.state["w"] += accelerations["w_dot"] * self.dt
        self.state["p"] += accelerations["p_dot"] * self.dt
        self.state["q"] += accelerations["q_dot"] * self.dt
        self.state["r"] += accelerations["r_dot"] * self.dt

        phi_dot = self.state["p"] + self.state["q"] * np.sin(phi) * np.tan(theta) + self.state["r"] * np.cos(phi) * np.tan(theta)
        theta_dot = self.state["q"] * np.cos(phi) - self.state["r"] * np.sin(phi)
        psi_dot = self.state["q"] * np.sin(phi) / np.cos(theta) + self.state["r"] * np.cos(phi) / np.cos(theta)

        self.state["phi"] += phi_dot * self.dt
        self.state["theta"] += theta_dot * self.dt
        self.state["psi"] += psi_dot * self.dt
