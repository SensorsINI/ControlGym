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
    def plot(self, costs: "dict[str, list]", axis_info: dict, save_to_image):
        num_datapoints_per_experiment = [len(v) for v in list(costs.values())]
        num_experiments = len(costs.values())
        sweep_var, sweep_values = list(axis_info.keys())[0], list(map(lambda x: int(x) if x.isnumeric() else x, list(axis_info.values())[0]))
        assert all(
            v == num_datapoints_per_experiment[0] for v in num_datapoints_per_experiment
        ), "All compared experiment series should have same number of trials"

        if self.ax is None:
            self.fig, self.ax = plt.subplots(
                figsize=(num_experiments, 8),
                gridspec_kw={"wspace": 0.1, "top": 0.9, "bottom": 0.3},
                dpi=300.0,
            )
        self.ax.clear()
        
        x_loc = [k + 1 if isinstance(x, str) else x for k, x in enumerate(sweep_values)]
        # for k, (exp_name, data) in enumerate(costs.items()):
        #     self.ax.scatter(
        #         np.repeat(x_loc[k], num_datapoints_per_experiment[k]), data, marker="x", label=exp_name
        #     )
        self.ax.boxplot(np.array(list(costs.values())).T, positions=x_loc)

        self.ax.set_ylabel("Realized mean cost per experiment")
        self.ax.set_title(
            f"Comparison of different control methods, N={num_datapoints_per_experiment[0]}"
        )
        self.ax.set_xlabel(sweep_var)
        self.ax.set_xticks(x_loc, labels=sweep_values)
        self.ax.minorticks_off()
        min_x = 0 if isinstance(max(sweep_values), str) else min(sweep_values)*0.9
        max_x = num_experiments + 1 if isinstance(max(sweep_values), str) else max(sweep_values)*1.1
        self.ax.set_xlim(min_x, max_x)

        # self.ax.legend(
        #     loc="lower center",
        #     bbox_to_anchor=(0.5, -0.3),
        # )

        if save_to_image:
            self.fig.savefig(
                os.path.join("Output", self._timestamp),
                bbox_inches="tight",
            )
        else:
            self.fig.show()
