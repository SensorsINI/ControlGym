import os
import numpy as np
from Utilities.utils import OutputPath
from Visualizations import Plotter
import matplotlib.pyplot as plt

plt.style.use(["science"])


def _build_color_seq(n):
    colors = ["pink", "lightblue", "lightgreen"]
    L = len(colors)
    return [colors[i % L] for i in range(n)]


class CostScatterPlotPlotter(Plotter):
    def plot(self, costs: "dict[str, list]", save_to_image):
        num_datapoints_per_experiment = len(list(costs.values())[0])
        num_experiments = len(costs.values())
        assert all(
            [len(c) == num_datapoints_per_experiment for c in costs.values()]
        ), "All compared experiment series should have same number of trials"

        if self.ax is None:
            self.fig, self.ax = plt.subplots(
                figsize=(num_experiments, 8), gridspec_kw={"wspace": 0.1, "top": 0.9, "bottom": 0.3}
            )
        self.ax.clear()
        data = np.array([c for c in costs.values()]).ravel()
        
        for k, (exp_name, data) in enumerate(costs.items()):
            self.ax.scatter(
                np.repeat(k+1, num_datapoints_per_experiment), data, marker="x", label=exp_name
            )

        self.ax.set_ylabel("Realized total cost per experiment")
        self.ax.set_title(
            f"Comparison of different control methods, N={num_datapoints_per_experiment}\nShowing median, Q1, Q3, IQR. Outliers not shown."
        )
        self.ax.set_xticklabels([])
        self.ax.set_xlim(0, num_experiments+1)

        self.ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.16),
        )

        if save_to_image:
            self.fig.savefig(
                os.path.join("Output", self._timestamp),
                bbox_inches="tight",
            )
        else:
            self.fig.show()
