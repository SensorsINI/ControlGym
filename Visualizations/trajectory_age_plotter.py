import os
import re
import numpy as np
from Utilities.utils import OutputPath
from Visualizations import Plotter
import matplotlib.pyplot as plt

plt.style.use(["science"])


class TrajectoryAgePlotter(Plotter):
    """
    This plotter is only relevant for MPC optimizers which use rollouts.
    It saves a histogram if the optimizer kept track of the age of its rollouts over time.
    The age is the rolling number of control steps in the past that each rollout was created.
    For an optimizer like RPGD, this can give insight into how often it replaces old input plans by new samples.
    """
    def plot(self, ages: "dict[str, list]", save_to_image):
        num_datapoints_per_experiment = [len(v) for v in list(ages.values())]
        num_experiments = len(ages.values())

        for name, data_series in ages.items():
            name = re.split("/|\\\\", name)[-2]
            if self.ax is None:
                self.fig, self.ax = plt.subplots(
                    figsize=(6, 4),
                    gridspec_kw={"wspace": 0.1, "top": 0.9, "bottom": 0.3},
                    dpi=300.0,
                )
            self.ax.clear()

            self.ax.hist(data_series, bins=np.max(data_series)-np.min(data_series), edgecolor="black", color="white", zorder=3)
            self.ax.grid(
                visible=True,
                which="major",
                axis="y",
                zorder=0,
            )
            
            self.ax.set_ylabel("Count")
            self.ax.set_title(
                f"Age of Trajectory from which MPC Control is Selected, Histogram"
            )
            self.ax.set_xlabel("Age of Input Plan")

            if save_to_image:
                path = os.path.join(self._path, f"trajectory_ages")
                if not os.path.exists(path):
                    os.makedirs(path)
                self.fig.savefig(
                    os.path.join(path, name),
                    bbox_inches="tight",
                )
            else:
                self.fig.show()
