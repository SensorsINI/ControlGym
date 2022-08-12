import csv
import os
import pandas as pd
from ControllersGym import Controller
from Utilities.utils import CurrentRunMemory, get_logger
log = get_logger(__name__)


def save_to_csv(config, controller: Controller, environment_name: str, path: str):
    os.makedirs(path, exist_ok=True)
    i = 0
    while os.path.isfile(os.path.join(path, f"Experiment-{i}.csv")):
        i += 1
    filename = os.path.join(path, f"Experiment-{i}.csv")
    log.info(f"Saving to the file {filename}")

    controller_outputs = controller.get_outputs()
    states = controller_outputs["s_logged"]
    inputs = controller_outputs["u_logged"]
    
    dt = config["2_environments"][environment_name]["dt"]
    df = pd.DataFrame({
        "time": [k * dt for k in range(states.shape[0])],
        **{f"x_{k}": states[:, k] for k in range(states.shape[1])},
        **{f"u_{k}": inputs[:, k] for k in range(inputs.shape[1])}
    })
    df = df.set_index("time")

    with open(filename, "a", newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow([f"# Gym Log"])
        writer.writerow([f"# {CurrentRunMemory.current_controller_name}"])
        writer.writerow([f"# Saving: {dt} s"])

    df.to_csv(filename, mode="a", header=True)

def load_from_csv(path):
    return pd.read_csv(path).to_numpy()