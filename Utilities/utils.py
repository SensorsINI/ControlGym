import os
from pathlib import Path


def get_output_path(timestamp: str, filename: str):
    folder = os.path.join("Output", f"{timestamp}_Experiment")
    Path(folder).mkdir(parents=True, exist_ok=True)
    return os.path.join(folder, f"{timestamp}_{filename}")
