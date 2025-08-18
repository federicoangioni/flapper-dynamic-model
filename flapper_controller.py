import pandas as pd
from controller import PID_controller
import matplotlib.pyplot as plt

flight_exp = "flight_001"

# Attitude PID definitions
freq_attitude = 500  # Hz

pitch_kp = 13
pitch_ki = 0
pitch_kd = 1
pitch_kff = 0

roll_kp = 10
roll_ki = 0
roll_kd = 0.2
roll_kff = 0

yaw_kp = 8
yaw_ki = 0
yaw_kd = 0.35
yaw_kff = 0

# Rate PID definitions
freq_attitude_rate = 500  # Hz

pitchrate_kp = 50
pitchrate_ki = 0
pitchrate_kd = 0
pitchrate_kff = 0

rollrate_kp = 50
rollrate_ki = 0
rollrate_kd = 0
rollrate_kff = 0

yawrate_kp = 80
yawrate_ki = 0
yawrate_kd = 0
yawrate_kff = 220


def main():
    pitch_controller = 0

    # Instantiate PID controllers
    pitch_pid = PID_controller(pitch_kp, pitch_ki, pitch_kd)




if __name__ == "__main__":

    # Declar data file paths
    data_dir = f"data/{flight_exp}/{flight_exp}"
    optitrack_csv = f"{data_dir}_optitrack.csv"
    onboard_csv = f"{data_dir}_flapper.csv"

    # Load onboard data
    onboard_data = pd.read_csv(onboard_csv)

    for i in range(len(onboard_data)):
        pass
        # pitch_pid.compute(onboard_data["controller.pitch"])

    print(len(onboard_data))
