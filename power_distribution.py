import numpy as np

flapperConfig = {"pitchServoNeutral": 50, 
                "yawServoNeutral": 50,
                "rollBias": 0,
                "maxThrust": 60000}

act_max = 65535

pitch_ampl = 0.4


def power_distribution(controls, ):

    thrust = min(controls["thrust"], flapperConfig["maxThrust"])
