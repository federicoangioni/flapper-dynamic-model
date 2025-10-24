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


def load_data():

    FILE_PATH = PLATFORM_DATA / "flight_001" / "flight_001_oriented_onboard.csv"
    data = pd.read_csv(FILE_PATH)

    return data
