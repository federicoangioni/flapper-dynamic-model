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
        rrb.Spatial3DView(
            origin="/flapper/",
            name="flapper",
        ),
        rrb.TimeSeriesView(origin="/dihedral/", name="dihedral", visible=False),
        rrb.TimeSeriesView(origin="/rotations/", name="rotations", visible=True),
        rrb.TimeSeriesView(origin="/accelerations/", name="accelerations", visible=True),
    ),
    collapse_panels=False,
)

if __name__ == "__main__":
    start = time()

    # Declare data file paths
    data = load_data(config.PLATFORM)
   
    # Load onboard data
    if config.USE_OPEN_LOOP:
        print("[bold thistle1]Running the modeled open loop.[/bold thistle1]")
    else:
        print("[bold thistle1]Running the controllers with the recorded data, here no open loop models are run. The data recorded from the IMU gets fed back to the controllers. [/bold thistle1]")
    
    simulation = Simulation()

    print("[bold green]Starting the simulation[/bold green]")

    rr.init("rerun_flapper", spawn=True)
    rr.send_blueprint(blueprint)

    if config.USE_OPEN_LOOP:
        for i in track(range(len(data)), description="Processing..."):
            setpoints = {"roll": 0, "pitch": -12, "yaw": 0, "yawrate": 0}

            cmd_thrust = 23000 
            
            simulation.simulate_flapper(1 / config.FREQ_ATTITUDE, setpoints, cmd_thrust, config.USE_OPEN_LOOP,)

            rr.log("dihedral/dihedral", rr.Scalars(np.rad2deg(simulation.current_dihedral)))

            rr.log(
            "/flapper/fb_body",
            rr.Points3D(
                [
                    simulation.current_pos[0],
                    simulation.current_pos[1],
                    simulation.current_pos[2],
                ],
                colors=[0, 255, 0],
                radii=[5],
            ),
            )

            rr.log("/rotations/roll", rr.Scalars(simulation.current_attitude[0]))
            rr.log("/rotations/pitch", rr.Scalars(simulation.current_attitude[1]))
            rr.log("/rotations/yaw", rr.Scalars(simulation.current_attitude[2]))

            rr.log("/accelerations/u_acc", rr.Scalars(simulation.current_acceleration[0]))
            rr.log("/accelerations/v_acc", rr.Scalars(simulation.current_acceleration[1]))
            rr.log("/accelerations/w_acc", rr.Scalars(simulation.current_acceleration[2]))

            