from pathlib import Path
import pandas as pd
import yaml
from dataclasses import dataclass
from typing import Dict, Any

ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / "data"


def load_data(experiment):

    FILE_PATH = DATA_DIR / experiment / f"{experiment}-onboard.csv"
    data = pd.read_csv(FILE_PATH)

    return data


# ======= PID STRUCTURES =======

@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float
    kff: float
    integration_limit: float


@dataclass
class RatePIDGains:
    kp: float
    ki: float
    kd: float
    kff: float
    filt_cut: float
    integration_limit: float
    max_delta: float | None = None


# ======= MMOI =======

@dataclass
class MMOI:
    ixx: float
    iyy: float
    izz: float


# ======= MASS =======

@dataclass
class MassConfig:
    wings: float
    no_wings: float


# ======= TRANSFER FUNCTION =======

@dataclass
class TransferFunction:
    tau_flapping: float
    omega_dihedral: float
    zeta_dihedral: float
    omega_yaw: float
    zeta_yaw: float


# ======= MOTOR & PWM =======

@dataclass
class PWMSettings:
    min: Dict[str, float]
    max: Dict[str, float]
    mid: Dict[str, float]
    pwm_to_freq: Dict[str, float]


@dataclass
class MotorSettings:
    max_thrust_pwm: int
    roll_bias: float
    pitch_servo_neutral: int
    yaw_servo_neutral: int


# ======= Dynamic model =======

@dataclass
class ThrustCoeffs:
    c1: float
    c2: float


@dataclass
class ModelCoeffs:
    k_xu: float
    k_yv: float
    k_zw: float
    k_n: float


@dataclass
class DynamicModel:
    thrust_coeffs: ThrustCoeffs
    model_coeffs: ModelCoeffs


# ======= Flapper dimensions =======

@dataclass
class FlapperDims:
    lw: float
    ly: float
    lk: float
    r: float
    lz: float
    l_hinge: float


# ======= Servo limits =======

@dataclass
class ServoMax:
    dihedral_max_deg: float
    flapping_max: float
    yaw_max_rad: float


# ======= Top-level config =======

@dataclass
class Config:
    use_dynamic_model: bool
    flight: str
    constants: Dict[str, float]
    frequencies: Dict[str, float]

    pid: Any
    mmoi: Any
    mass: MassConfig
    transfer_function: TransferFunction
    motor: MotorSettings
    pwm: PWMSettings
    dynamic_model: DynamicModel
    flapper_dims: FlapperDims
    servo: Dict[str, ServoMax]


# ======= Loader =======

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    pid_struct = {
        "attitude": {
            "roll": PIDGains(**data["pid"]["attitude"]["roll"]),
            "pitch": PIDGains(**data["pid"]["attitude"]["pitch"]),
            "yaw": PIDGains(**data["pid"]["attitude"]["yaw"]),
        },
        "rate": {
            "rollrate": RatePIDGains(**data["pid"]["rate"]["rollrate"]),
            "pitchrate": RatePIDGains(**data["pid"]["rate"]["pitchrate"]),
            "yawrate": RatePIDGains(**data["pid"]["rate"]["yawrate"]),
        },
    }

    mmoi = {
        "with_wings": MMOI(**data["mmoi"]["with_wings"]),
        "with_wings_xy": MMOI(**data["mmoi"]["with_wings_xy"]),
        "no_wings": MMOI(**data["mmoi"]["no_wings"]),
    }

    servo_struct = {
        "max_act_state": ServoMax(**data["servo"]["max_act_state"])
    }

    return Config(
        use_dynamic_model=data["use_dynamic_model"],
        flight=data["flight"],
        constants=data["constants"],
        frequencies=data["frequencies"],
        pid=pid_struct,
        mmoi=mmoi,
        mass=MassConfig(**data["mass"]),
        transfer_function=TransferFunction(**data["transfer_function"]),
        motor=MotorSettings(**data["motor"]),
        pwm=PWMSettings(**data["pwm"]),
        dynamic_model=DynamicModel(
            thrust_coeffs=ThrustCoeffs(**data["dynamic_model"]["thrust_coeffs"]),
            model_coeffs=ModelCoeffs(**data["dynamic_model"]["model_coeffs"]),
        ),
        flapper_dims=FlapperDims(**data["flapper_dims"]),
        servo=servo_struct,
    )

