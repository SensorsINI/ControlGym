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


class BoxPlotPlotter(Plotter):
    def plot(self, costs: "dict[str, list]", save_to_image):
        if self.ax is None:
            self.fig, self.ax = plt.subplots(
                figsize=(10, 8), gridspec_kw={"wspace": 0.1, "top": 0.9, "bottom": 0.3}
            )
        self.ax.clear()

        data = [c for c in costs.values()]
        num_datapoints_per_experiment = len(data[0])
        assert all(
            [len(c) == num_datapoints_per_experiment for c in data]
        ), "All compared experiment series should have same number of trials"
        bplot = self.ax.boxplot(
            data, notch=False, patch_artist=True, meanline=False, sym="", widths=0.3
        )

        for patch, color in zip(bplot["boxes"], _build_color_seq(len(costs.values()))):
            patch.set_facecolor(color)

        self.ax.set_ylabel("Realized average cost per experiment")
        self.ax.set_title(
            f"Comparison of different control methods, N={num_datapoints_per_experiment}\nShowing median, Q1, Q3, IQR. Outliers not shown."
        )
        self.ax.set_xticklabels([])

        self.ax.legend(
            bplot["boxes"],
            list(costs.keys()),
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
