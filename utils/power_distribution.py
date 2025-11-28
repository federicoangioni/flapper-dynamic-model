from . import config_old 

default_idle_thrust = 0

pitch_ampl = 0.4

def capAngle(angle):
    result = angle
    while result > 180.0:
        result -= 360.0
    while result < -180.0:
        result += 360.0
    return result

def limit_servo_neutral(value):
    if value > 75:
        value = 75
    elif value < 25:
        value = 25

    return value


def limit_roll_bias(value):
    if value > 25:
        value = 25
    elif value < -25:
        value = -25

    return value

def limit_thrust(value, min, max):
    if value < min:
        return min
    
    if value > max:
        return max
    
    return value


def power_distribution(controls,):
    """
    m2: left flapping motor
    m4: right flapping motor
    m1: pitch servo
    m3: yaw servo    
    """
    thrust = min(controls["cmd_thrust"], config_old.FLAPPERCONFIG["maxThrust"])

    pitch_neutral = limit_servo_neutral(config_old.FLAPPERCONFIG["pitchServoNeutral"])
    yaw_neutral = limit_servo_neutral(config_old.FLAPPERCONFIG["yawServoNeutral"])
    roll_bias = limit_roll_bias(config_old.FLAPPERCONFIG["rollBias"])

    motors_m1_uncapped = pitch_neutral * config_old.UINT_16B / 100 + pitch_ampl * controls["cmd_pitch"]
    motors_m3_uncapped = yaw_neutral * config_old.UINT_16B / 100 - controls["cmd_yaw"]

    motors_m2_uncapped = 0.5 * controls["cmd_roll"] + thrust * (1.0 + roll_bias / 100)
    motors_m4_uncapped = -0.5 * controls["cmd_roll"] + thrust * (1.0 - roll_bias / 100)


    m1 = limit_thrust(motors_m1_uncapped, 0, config_old.UINT_16B)
    m3 = limit_thrust(motors_m3_uncapped, 0, config_old.UINT_16B)
    m2 = limit_thrust(motors_m2_uncapped,  default_idle_thrust, config_old.UINT_16B)
    m4 = limit_thrust(motors_m4_uncapped,  default_idle_thrust, config_old.UINT_16B)
    
    return m1, m2, m3, m4
