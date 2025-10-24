import numpy as np

USE_DYNAMIC_MODEL = False

FLIGHT = "flight_002"

UINT_16B = 65535

FREQ_ATTITUDE = 500  # Hz
FREQ_ATTITUDE_RATE = 500  # Hz

# Attitude PID definitions
ROLL_KP = 10
ROLL_KI = 0
ROLL_KD = 0.2
ROLL_KFF = 0
ROLL_INTEGRATION_LIMIT = 20.0

PITCH_KP = 13
PITCH_KI = 0
PITCH_KD = 1
PITCH_KFF = 0
PITCH_INTEGRATION_LIMIT = 20.0

YAW_KP = 8
YAW_KI = 0
YAW_KD = 0.35
YAW_KFF = 0
YAW_INTEGRATION_LIMIT = 360.0

# Rate PID definitions
ROLLRATE_KP = 50
ROLLRATE_KI = 0
ROLLRATE_KD = 0
ROLLRATE_KFF = 0
OMX_FILT_CUT = 20
ROLLRATE_INTEGRATION_LIMIT = 33.3

PITCHRATE_KP = 50
PITCHRATE_KI = 0
PITCHRATE_KD = 0
PITCHRATE_KFF = 0
OMY_FILT_CUT = 20
PITCHRATE_INTEGRATION_LIMIT = 33.3

YAWRATE_KP = 80
YAWRATE_KI = 0
YAWRATE_KD = 0
YAWRATE_KFF = 220
OMZ_FILT_CUT = 5
YAWRATE_INTEGRATION_LIMIT = 166.7
YAW_MAX_DELTA = 30.0


# Mass moments of inertia

MMOI_WITH_WINGS = {"Ixx": 5.6e-4, "Iyy": 3.4e-4, "Izz": 3.6e-4}  # kg/m^2 # Flapper

MMOI_WITH_WINGS_XY = {"Ixx": 5.6e-4, "Iyy": 3.4e-4, "Izz": 2.8e-4}  # kg/m^2 # with wings oriented in XY plane

MMOI_NO_WINGS = {"Ixx": 3.1e-4, "Iyy": 3.0e-4, "Izz": 2.7e-4}  # kg/m^2

# Mass
MASS_WINGS = 0.103 # kg # Flapper

MASS_NO_WINGS = 0.0943  # kg

# Transfer function values
TAU_FLAPPING = 0.0796

OMEGA_DIHEDRAL = 40

ZETA_DIHEDRAL = 0.634

OMEGA_YAW = 10 # estimated general servo values 

ZETA_YAW = 0.7 # estimated general servo values


# Motor charactheristics

MAXTHRUST_PWM = 60000

ROLLBIAS = 0

PITCH_SERVO_NEUTRAL = 55

YAW_SERVO_NEUTRAL = 65

MIN_PWM = {'m1' : 0, 'm2': 0, 'm3': 0, 'm4': 0}

# Dihedral and yaw maximum angle estimated from pictures
MAX_ACT_STATE = {'dihedral_max' : np.deg2rad(18), 'flapping_max': 35, 'yaw_max':  0.786842889}

# Dynamic model specifics

THRUST_COEFFS = {'c1': 0.08, 'c2': -0.02} # Flapper +

MODEL_COEFFS = {'k_xu': 4.12, 'k_yv': 1.8, 'k_zw': 1.8e-1, 'k_N': 2.7e-3}

FLAPPER_DIMS = {'lw' : 0.08, 'ly': 0.036, 'lk': 0.1, 'R':0.098, 'lz' : 0.027, 'l_hinge': 0.035}

# Assembling necessary dictionaries
TF_COEFFS = {'tau_flapping': TAU_FLAPPING, 'omega_dihedral': OMEGA_DIHEDRAL, 'zeta_dihedral': ZETA_DIHEDRAL, 'omega_yaw': OMEGA_YAW, 'zeta_yaw': ZETA_YAW}

MAX_PWM = {'m1' : UINT_16B, 'm2':  MAXTHRUST_PWM, 'm3': UINT_16B, 'm4': MAXTHRUST_PWM}

FLAPPERCONFIG = {"pitchServoNeutral": PITCH_SERVO_NEUTRAL, "yawServoNeutral": YAW_SERVO_NEUTRAL, "rollBias": ROLLBIAS, "maxThrust": MAXTHRUST_PWM}

MID_PWM = {'m1': PITCH_SERVO_NEUTRAL * MAX_PWM['m1'] / 100, 'm2' : 0, 'm3': YAW_SERVO_NEUTRAL * MAX_PWM['m3'] / 100, 'm4': 0}

PWM_TO_FREQ = {'m' : 0.00038492781077195567, 'c': -0.13250131020527664}