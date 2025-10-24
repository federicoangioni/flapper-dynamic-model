from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / "data"


def load_data(experiment):

    FILE_PATH = DATA_DIR / experiment / f"{experiment}-onboard.csv"
    data = pd.read_csv(FILE_PATH)

    return data
