"""
This file has the purpose of loading the files, taking the values that are relevant to the model,
and format them in a uniform way if the two platforms of 'nimble' or 'flapper' are used.

Below you can find some information on the nimble data, with the experiment number and the setpoint angle.

lateral data:

experiment  95: 15 degrees bank,
experiment 100: 30 degrees bank,
experiment  99: 45 degrees bank,
experiment  98: 60 degrees bank

longitudinal data:

MODE:
    - free flight
    - lateral
    - longitudinal
    - hover
    - yaw movements on hovering (change this to a more compact and representative name)
"""

from pathlib import Path
import pandas as pd
import scipy.io as sio
from . import config

ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / "data"
PLATFORM_DATA = DATA_DIR / config.PLATFORM


def load_data(drone_platform):
    if drone_platform == "nimble":
        # load the data
        DATA_FILE = PLATFORM_DATA / "science_paper.mat"
        data = sio.loadmat(DATA_FILE, squeeze_me=True, struct_as_record=False)

        nexp = 100
        experiments = data[f"experiment{nexp}"]

        # Obtain the setpoints
        setpoints_roll = experiments.onboard.angles_commands_setpoints.SETroll
        setpoints_pitch = experiments.onboard.angles_commands_setpoints.SETpitch
        setpoints_yaw = experiments.onboard.angles_commands_setpoints.SETyaw

        rollrate = experiments.onboard.rates.OMx_IMU
        pitchrate = experiments.onboard.rates.OMy_IMU
        yawrate = experiments.onboard.rates.OMz_IMU

        cmd_thrust = experiments.onboard.angles_commands_setpoints.CMDthrottle
        cmd_roll = experiments.onboard.angles_commands_setpoints.CMDroll
        cmd_pitch = experiments.onboard.angles_commands_setpoints.CMDpitch
        cmd_yaw = experiments.onboard.angles_commands_setpoints.CMDyaw

        acc_x = experiments.motion_tracking.DVEL_BODYx_filtered
        acc_y = experiments.motion_tracking.DVEL_BODYy_filtered
        acc_z = experiments.motion_tracking.DVEL_BODYz_filtered

        """
        CMD_dihed = np.radians(data.onboard_interpolated.CMDpitch_interp[nman] / 100 * 18)
        """

    elif drone_platform == "flapper":
        FILE_PATH = PLATFORM_DATA / "flight_001" / "flight_001_oriented_onboard.csv"
        data = pd.read_csv(FILE_PATH)

    return data
