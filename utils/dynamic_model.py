from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np
from scipy.signal import lti
from . import config_old

g0 = 9.80665  # m/s^2

@dataclass
class FlapperState:
    phi: float = 0.0
    theta: float = 0.0
    psi: float = 0.0
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0
    u: float = 0.0
    v: float = 0.0
    w: float = 0.0

@dataclass
class ActuatorState:
    x_freq_left: np.ndarray = field(default_factory=lambda: np.zeros(0))
    x_freq_right: np.ndarray = field(default_factory=lambda: np.zeros(0))
    x_dihedral: np.ndarray = field(default_factory=lambda: np.zeros(0))
    x_yaw: np.ndarray = field(default_factory=lambda: np.zeros(0))

class DynamicModel:
    def __init__(
        self,
        dt,
        inertia: Dict[str, float],
        mass: float,
        model_coeffs: Dict[str, float],
        thrust_coeffs: Dict[str, float],
        flapper_dims: Dict[str, float],
        tf_coeffs: Dict[str, float],
        max_pwm: Dict[str, float],
        mid_pwm: Dict[str, float],
        min_pwm: Dict[str, float],
        max_act_state: Dict[str, float],
    ):
        """Assume everything is given in standard SI units"""

        self.Ixx = inertia["Ixx"]
        self.Iyy = inertia["Iyy"]
        self.Izz = inertia["Izz"]

        self.m = mass
        self.dt = dt

        self.c1 = thrust_coeffs["c1"]
        self.c2 = thrust_coeffs["c2"]

        self.k_xu = model_coeffs["k_xu"]
        self.k_yv = model_coeffs["k_yv"]
        self.k_zw = model_coeffs["k_zw"]
        self.k_N = model_coeffs["k_N"]

        self.lw = flapper_dims["lw"]
        self.lz = flapper_dims["lz"]
        self.lk = flapper_dims["lk"]
        self.ly = flapper_dims["ly"]
        self.R = flapper_dims["R"]
        self.l_hinge = flapper_dims["l_hinge"]

        self.flapping_max = max_act_state["flapping_max"]
        self.dihedral_max = max_act_state["dihedral_max"]
        self.yaw_max = max_act_state["yaw_max"]

        # store pwm limits grouped (keeps API close to original)
        self.max_pwm = max_pwm
        self.min_pwm = min_pwm
        self.mid_pwm = mid_pwm

        tau_flapping = tf_coeffs["tau_flapping"]
        omega_dihedral = tf_coeffs["omega_dihedral"]
        zeta_dihedral = tf_coeffs["zeta_dihedral"]
        omega_yaw = tf_coeffs["omega_yaw"]
        zeta_yaw = tf_coeffs["zeta_yaw"]

        (
            self.ss_flapping,
            self.ss_dihedral,
            self.ss_yaw,
            x_freq_left,
            x_freq_right,
            x_dihedral,
            x_yaw,
        ) = self.init_state_space(tau_flapping, omega_dihedral, zeta_dihedral, omega_yaw, zeta_yaw)

        self.flapper_state = FlapperState()
        self.actuator_state = ActuatorState(
            x_freq_left=x_freq_left,
            x_freq_right=x_freq_right,
            x_dihedral=x_dihedral,
            x_yaw=x_yaw,
        )

    # ---------- helpers ----------
    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return float(np.clip(x, lo, hi))

    def _safe_arcsin(self, x: float) -> float:
        return float(np.arcsin(self._clamp(x, -1.0, 1.0)))

    def compute_thrust(self, f: float) -> float:
        return self.c1 * f + self.c2

    # ---------- state-space setup ----------
    def init_state_space(
        self, tau_flapping: float, omega_dihedral: float, zeta_dihedral: float, omega_yaw: float, zeta_yaw: float
    ):
        num_flapping = [1.0]
        den_flapping = [tau_flapping, 1.0]
        sys_flapping = lti(num_flapping, den_flapping).to_ss()
        A_flapping, B_flapping, C_flapping, D_flapping = (
            sys_flapping.A,
            sys_flapping.B,
            sys_flapping.C,
            sys_flapping.D,
        )

        num_dihedral = [omega_dihedral ** 2]
        den_dihedral = [1.0, 2.0 * zeta_dihedral * omega_dihedral, omega_dihedral ** 2]
        sys_dihedral = lti(num_dihedral, den_dihedral).to_ss()
        A_dihedral, B_dihedral, C_dihedral, D_dihedral = (
            sys_dihedral.A,
            sys_dihedral.B,
            sys_dihedral.C,
            sys_dihedral.D,
        )

        num_yaw = [omega_yaw ** 2]
        den_yaw = [1.0, 2.0 * zeta_yaw * omega_yaw, omega_yaw ** 2]
        sys_yaw = lti(num_yaw, den_yaw).to_ss()
        A_yaw, B_yaw, C_yaw, D_yaw = (sys_yaw.A, sys_yaw.B, sys_yaw.C, sys_yaw.D)

        ss_flapping = {"A": A_flapping, "B": B_flapping, "C": C_flapping, "D": D_flapping}
        ss_dihedral = {"A": A_dihedral, "B": B_dihedral, "C": C_dihedral, "D": D_dihedral}
        ss_yaw = {"A": A_yaw, "B": B_yaw, "C": C_yaw, "D": D_yaw}

        x_freq_left = np.zeros(A_flapping.shape[0])
        x_freq_right = np.zeros(A_flapping.shape[0])
        x_dihedral = np.zeros(A_dihedral.shape[0])
        x_yaw = np.zeros(A_yaw.shape[0])

        return ss_flapping, ss_dihedral, ss_yaw, x_freq_left, x_freq_right, x_dihedral, x_yaw

    # ---------- actuator/servo modelling ----------
    def step_system(self, x: np.ndarray, ss: Dict[str, np.ndarray], u: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        u_vec = np.atleast_1d(u).astype(float)
        y = ss["C"] @ x + ss["D"] @ u_vec
        x_new = x + dt * (ss["A"] @ x + ss["B"] @ u_vec)
        return x_new, np.atleast_1d(y)

    def compute_control_inputs(self, pwm: float, min_pwm: float, mid_pwm: float, max_pwm: float, max_state: float) -> float:
        # avoid division by zero if configuration is degenerate
        if pwm >= mid_pwm:
            denom = max_pwm - mid_pwm
            if denom == 0:
                return 0.0
            val = (pwm - mid_pwm) / denom * max_state
        else:
            denom = mid_pwm - min_pwm
            if denom == 0:
                return 0.0
            val = (pwm - mid_pwm) / denom * max_state

        # keep sign semantics and clamp
        return float(np.clip(val, -abs(max_state), abs(max_state)))

    def pwm_to_angle(self, pwm_m1: float, pwm_m2: float, pwm_m3: float, pwm_m4: float):
        dihedral_control = self.compute_control_inputs(
            pwm_m1, self.min_pwm["m1"], self.mid_pwm["m1"], self.max_pwm["m1"], self.dihedral_max
        )
        freq_left_control = config_old.PWM_TO_FREQ["m"] * pwm_m2 + config_old.PWM_TO_FREQ["c"]
        yaw_control = self.compute_control_inputs(
            pwm_m3, self.min_pwm["m3"], self.mid_pwm["m3"], self.max_pwm["m3"], self.yaw_max
        )
        freq_right_control = config_old.PWM_TO_FREQ["m"] * pwm_m4 + config_old.PWM_TO_FREQ["c"]
        return dihedral_control, freq_left_control, yaw_control, freq_right_control

    def compute_control_states(self, pwm_m1: float, pwm_m2: float, pwm_m3: float, pwm_m4: float):
        dihedral_control, freq_left_control, yaw_control, freq_right_control = self.pwm_to_angle(
            pwm_m1, pwm_m2, pwm_m3, pwm_m4
        )

        x_dihedral_new, y_dihedral = self.step_system(self.actuator_state.x_dihedral, self.ss_dihedral, dihedral_control, self.dt)
        x_freq_left_new, y_freq_left = self.step_system(self.actuator_state.x_freq_left, self.ss_flapping, freq_left_control, self.dt)
        x_yaw_new, y_yaw = self.step_system(self.actuator_state.x_yaw, self.ss_yaw, yaw_control, self.dt)
        x_freq_right_new, y_freq_right = self.step_system(self.actuator_state.x_freq_right, self.ss_flapping, freq_right_control, self.dt)

        self.actuator_state.x_dihedral = x_dihedral_new
        self.actuator_state.x_freq_left = x_freq_left_new
        self.actuator_state.x_yaw = x_yaw_new
        self.actuator_state.x_freq_right = x_freq_right_new

        return float(y_dihedral.squeeze()), float(y_freq_left.squeeze()), float(y_yaw.squeeze()), float(y_freq_right.squeeze())

    # ---------- dynamics ----------
    def calculate_accelerations(self, pwm_signals: Dict[str, float]):
        dihedral, freq_left, yaw_angle, freq_right = self.compute_control_states(pwm_signals["m1"], pwm_signals["m2"], pwm_signals["m3"], pwm_signals["m4"])

        thrust_left = self.compute_thrust(freq_left)
        thrust_right = self.compute_thrust(freq_right)

        s = self.flapper_state  # shorthand
        u, v, w = s.u, s.v, s.w
        p, q, r = s.p, s.q, s.r
        phi, theta = s.phi, s.theta

        ld = self.lw * np.sin(dihedral)
        ld_dot = 0.0

        arg_L = (-self.l_hinge * np.sin(dihedral) + self.ly * np.sin(yaw_angle)) / self.lk
        arg_R = (-self.l_hinge * np.sin(dihedral) - self.ly * np.sin(yaw_angle)) / self.lk
        alpha_L = self._safe_arcsin(arg_L)
        alpha_R = self._safe_arcsin(arg_R)

        Z_L = freq_left * (w - ld * q + self.lw * np.cos(dihedral) * p)
        Z_R = freq_right * (w - ld * q - self.lw * np.cos(dihedral) * p)

        X = -self.k_xu * (freq_left + freq_right) * (u - self.lz * q + self.lw * ld_dot) - thrust_left * np.sin(alpha_L) - thrust_right * np.sin(alpha_R)
        Y = -self.k_yv * (freq_left + freq_right) * (v + self.lz * p)
        Z = -self.k_zw * (Z_L + Z_R) - (thrust_right * np.cos(alpha_R) + thrust_left * np.cos(alpha_L))

        L = (self.k_zw * (Z_L - Z_R) * self.lw * np.cos(dihedral)) + (Y * self.lz) + (self.lw * np.cos(dihedral) * (thrust_left - thrust_right))

        M = X * self.lz + ld * self.k_zw * (Z_L + Z_R) + (thrust_right * np.cos(alpha_R) + thrust_left * np.cos(alpha_L)) * ld + (thrust_left * np.sin(alpha_L) + thrust_right * np.sin(alpha_R)) * self.lz

        N = -self.k_N * ((freq_left + freq_right) * self.R * r + (freq_left - freq_right) * u + (freq_left + freq_right) * dihedral * v)

        u_dot = -(w * q - v * r) + X / self.m + g0 * np.sin(theta)
        v_dot = -(u * r - w * p) + Y / self.m - g0 * np.cos(theta) * np.sin(phi)
        w_dot = -(v * p - u * q) + Z / self.m - g0 * np.cos(theta) * np.cos(phi)
        p_dot = ((self.Iyy - self.Izz) * q * r + L) / self.Ixx
        q_dot = ((self.Izz - self.Ixx) * p * r + M) / self.Iyy
        r_dot = ((self.Ixx - self.Iyy) * p * q + N) / self.Izz

        accelerations = {"u_dot": u_dot, "v_dot": v_dot, "w_dot": w_dot, "p_dot": p_dot, "q_dot": q_dot, "r_dot": r_dot}
        return accelerations, dihedral, freq_left, yaw_angle, freq_right

    def update(self, pwm_signals: Dict[str, float]):
        accelerations, dihedral, freq_left, yaw_angle, freq_right = self.calculate_accelerations(pwm_signals)

        s = self.flapper_state
        s.u += accelerations["u_dot"] * self.dt
        s.v += accelerations["v_dot"] * self.dt
        s.w += accelerations["w_dot"] * self.dt
        s.p += accelerations["p_dot"] * self.dt
        s.q += accelerations["q_dot"] * self.dt
        s.r += accelerations["r_dot"] * self.dt

        phi_dot = s.p + s.q * np.sin(s.phi) * np.tan(s.theta) + s.r * np.cos(s.phi) * np.tan(s.theta)
        theta_dot = s.q * np.cos(s.phi) - s.r * np.sin(s.phi)
        psi_dot = s.q * np.sin(s.phi) / np.cos(s.theta) + s.r * np.cos(s.phi) / np.cos(s.theta)

        s.phi += phi_dot * self.dt
        s.theta += theta_dot * self.dt
        s.psi += psi_dot * self.dt

        rates = np.array([float(s.p), float(s.q), float(s.r)])
        alphas = np.array([float(accelerations["p_dot"]), float(accelerations["q_dot"]), float(accelerations["r_dot"])])
        attitude = np.array([float(s.phi), float(s.theta), float(s.r)])
        accelerations_vec = np.array([float(accelerations["u_dot"]), float(accelerations["v_dot"]), float(accelerations["w_dot"])])
        velocity = np.array([float(s.u), float(s.v), float(s.w)])

        return attitude, rates, alphas, velocity, accelerations_vec, dihedral, freq_left, yaw_angle, freq_right