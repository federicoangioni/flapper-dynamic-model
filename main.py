import rerun as rr
from utils.simulation_env import Simulation 
import rerun.blueprint as rrb
from time import time
import numpy as np
from rich import print
from rich.progress import track
import matplotlib.pyplot as plt

from utils.data_loader import load_data
from utils import config_old

blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.Spatial3DView(origin="/flapper/", name="flapper",),
        rrb.TimeSeriesView(origin="/dihedral/", name="dihedral", visible=False),
        rrb.TimeSeriesView(origin="/rotations/", name="rotations", visible=True),
        rrb.TimeSeriesView(origin="/accelerations/", name="accelerations", visible=True),
    ),
    collapse_panels=False,
)

def log_simulated_values(simulation_obj):
    rr.log('/rotations/roll', rr.Scalars(simulation_obj.flapper_state[-1]['phi']))
    rr.log('/rotations/pitch', rr.Scalars(simulation_obj.flapper_state[-1]['theta']))
    rr.log('/rotations/yaw', rr.Scalars(simulation_obj.flapper_state[-1]['psi']))


if __name__ == "__main__":
    start = time()

    # Declare data file paths
    data = load_data(config_old.FLIGHT)
   
    # Load onboard data
    if config_old.USE_DYNAMIC_MODEL:
        print("[bold thistle1]Running the modeled open loop.[/bold thistle1]")
    else:
        print("[bold thistle1]Running the controllers with the recorded data, here no open loop models are run. " \
        "The data recorded from the IMU gets fed back to the controllers. [/bold thistle1]")
    
    simulation = Simulation(1 / config_old.FREQ_ATTITUDE, config_old.USE_DYNAMIC_MODEL)

    print("[bold green]Starting the simulation[/bold green]")

    if config_old.USE_DYNAMIC_MODEL:
        for i in track(range(len(data)), description="Processing..."):
            setpoints = {"roll": 0, "pitch": -40, "yaw": 0,}

            cmd_thrust = 23000 
            
            simulation.simulate_flapper(setpoints, cmd_thrust, i)


    else:
        for i in track(range(len(data)), description="Processing..."):
            setpoints = {"roll": data.loc[i, "controller.roll"], "pitch": data.loc[i, "controller.pitch"], "yaw": data.loc[i, "controller.yaw"], "yawrate": data.loc[i, "controller.yawRate"]}

            cmd_thrust = data.loc[i, "controller.cmd_thrust"]

            simulation.simulate_flapper(setpoints, cmd_thrust, i, data)

            
    flapper_state_df, controls_df, motors_df = simulation.save_simulation()

    end = time()

    print(f"[magenta]Process run in {round(end - start, 3)} s[/magenta]")

    plt.plot(motors_df["m3"])
    plt.plot(data["motor.m3"])
    plt.show()


