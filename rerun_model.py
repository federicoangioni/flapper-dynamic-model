import rerun as rr
from flapper_model import Simulation 
import rerun.blueprint as rrb
from time import time
import numpy as np
from rich import print
from rich.progress import track

from utils.data_loader import load_data
from utils import config

blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.Spatial3DView(origin="/flapper/", name="flapper",),
        rrb.TimeSeriesView(origin="/dihedral/", name="dihedral", visible=False),
        rrb.TimeSeriesView(origin="/rotations/", name="rotations", visible=True),
        rrb.TimeSeriesView(origin="/accelerations/", name="accelerations", visible=True),
    ),
    collapse_panels=False,
)

if __name__ == "__main__":

    pass