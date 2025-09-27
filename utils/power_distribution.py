from . import config 

default_idle_thrust = 0

pitch_ampl = 0.4

motors = {"m1": 0, "m2": 0, "m3": 0, "m4": 0}

uint_16 = 65535

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
    thrust = min(controls["thrust"], config.FLAPPERCONFIG["maxThrust"])

    pitch_neutral = limit_servo_neutral(config.FLAPPERCONFIG["pitchServoNeutral"])
    yaw_neutral = limit_servo_neutral(config.FLAPPERCONFIG["yawServoNeutral"])
    roll_bias = limit_roll_bias(config.FLAPPERCONFIG["rollBias"])

    motors_m1_uncapped = pitch_neutral * uint_16 / 100 + pitch_ampl * controls["pitch"]
    motors_m3_uncapped = yaw_neutral * uint_16 / 100 - controls["yaw"]
    motors_m2_uncapped = 0.5 * controls["roll"] + thrust * (1.0 + roll_bias / 100)
    motors_m4_uncapped = -0.5 * controls["roll"] + thrust * (1.0 - roll_bias / 100)


    motors["m1"] = limit_thrust(motors_m1_uncapped, 0, uint_16)
    motors["m3"] = limit_thrust(motors_m3_uncapped, 0, uint_16)
    motors["m2"] = limit_thrust(motors_m2_uncapped,  default_idle_thrust, uint_16)
    motors["m4"] = limit_thrust(motors_m4_uncapped,  default_idle_thrust, uint_16)
    
    return motors
