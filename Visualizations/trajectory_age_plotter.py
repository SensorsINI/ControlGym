import os
import re
import numpy as np
from Utilities.utils import OutputPath
from Visualizations import Plotter
import matplotlib.pyplot as plt

plt.style.use(["science"])


class TrajectoryAgePlotter(Plotter):
    def plot(self, ages: "dict[str, list]", save_to_image):
        num_datapoints_per_experiment = [len(v) for v in list(ages.values())]
        num_experiments = len(ages.values())

        for name, data_series in ages.items():
            name = re.split("/|\\\\", name)[-2]
            if self.ax is None:
                self.fig, self.ax = plt.subplots(
                    figsize=(6, 5),
                    gridspec_kw={"wspace": 0.1, "top": 0.9, "bottom": 0.3},
                    dpi=300.0,
                )
            self.ax.clear()

            self.ax.hist(data_series, bins=20, edgecolor="black", color="white")
            
            self.ax.set_ylabel("Count")
            self.ax.set_title(
                f"Age of trajectory from which MPC control was selected, histogram, {num_datapoints_per_experiment[0]} control steps"
            )
            self.ax.set_xlabel("Age of trajectory")

            if save_to_image:
                path = os.path.join("Output", f"{self._timestamp}_trajectory_ages")
                if not os.path.exists(path):
                    os.makedirs(path)
                self.fig.savefig(
                    os.path.join(path, name),
                    bbox_inches="tight",
                )
            else:
                self.fig.show()
