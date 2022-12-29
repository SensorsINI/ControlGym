import os
import matplotlib.pyplot as plt
import numpy as np

from Visualizations import Plotter
from Utilities.utils import OutputPath

plt.style.use(["science"])


class HorizonCostPlotter(Plotter):
    """
    Saves a scatter plot. Each marker represents the horizon cost of a rollout.
    x-axis: The control iteration.
    y-axis: The costs of all the rollouts.
    Ideally, these rollout costs should tend to decrease going from left to right along the x-axis.
    """
    def plot(self, costs: np.ndarray, save_to_image: bool = True):
        if self.ax is None:
            self.fig, self.ax = plt.subplots(
                figsize=(10, 8),
                gridspec_kw={
                    "wspace": 0.1,
                    "left": 0.1,
                    "right": 0.9,
                    "top": 0.9,
                    "bottom": 0.1,
                },
                dpi=300.0,
            )
        self.ax.clear()
        num_steps, num_samples = costs.shape
        self.ax.scatter(
            np.repeat(np.arange(num_steps), num_samples),
            costs.ravel(order="C"),
            linewidths=0.4,
            marker="x",
            alpha=0.4,
        )
        self.ax.get_xaxis().set_major_locator(
            plt.MaxNLocator(nbins=10, integer=True, min_n_ticks=2)
        )
        self.ax.set_ylabel("Total horizon cost")
        self.ax.set_xlabel("Control iteration")
        self.ax.set_title("Total cost of input plans per global control iteration")

        self._display_some_config()
        
        c = 1
        p = f"J_logged_{c}.svg"
        while os.path.isfile(os.path.join(self._path, p)):
            c += 1
            p = f"J_logged_{c}.svg"

        if save_to_image:
            self.fig.savefig(
                os.path.join(self._path, p),
                bbox_inches="tight",
            )
        else:
            self.fig.show()
