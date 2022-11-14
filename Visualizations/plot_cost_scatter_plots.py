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
        """_summary_

        :param costs: _description_
        :type costs: dict[str, list]
        :param axis_info: A dictionary with keys "description", "xlabel", "ylabel", "sweep_values", "boxcolors"
        :type axis_info: dict
        :param save_to_image: _description_
        :type save_to_image: _type_
        """
        num_datapoints_per_experiment = [len(v) for v in list(costs.values())]
        num_experiments = len(costs.values())
        sweep_values = list(map(lambda x: int(x) if x.isnumeric() else x, axis_info["sweep_values"]))
        assert all(
            v == num_datapoints_per_experiment[0] for v in num_datapoints_per_experiment
        ), "All compared experiment series should have same number of trials"

        if self.ax is None:
            self.fig, self.ax = plt.subplots(
                figsize=(4, 3),
                gridspec_kw={"wspace": 0.1, "top": 0.9, "bottom": 0.3},
                dpi=300.0,
            )
        self.ax.clear()
        
        # x_loc = [k + 1 if isinstance(x, str) else x for k, x in enumerate(sweep_values)]
        x_loc = list(range(1, num_experiments + 1))

        # for k, (exp_name, data) in enumerate(costs.items()):
        #     self.ax.scatter(
        #         np.repeat(x_loc[k], num_datapoints_per_experiment[k]), data, marker=".", label=exp_name, color="k"
        #     )
        c = np.array(list(costs.values())).T
        boxplot = self.ax.boxplot(c, positions=x_loc, patch_artist=True)#, flierprops={"markerfacecolor": "white"})
        for patch, color in zip(boxplot["boxes"], axis_info["boxcolors"]):
            patch.set_facecolor(color)
        self.ax.grid(visible=True, which="major", axis="y")

        self.ax.set_ylabel(axis_info["ylabel"], fontsize=14)
        # self.ax.set_title(
        #     f"Variation of {axis_info['description']}, {num_datapoints_per_experiment[0]} Random Trials"
        # )
        self.ax.set_xlabel(axis_info["xlabel"], fontsize=14)
        self.ax.set_xticks(x_loc, labels=sweep_values)
        self.ax.minorticks_off()
        # min_x = 0 if isinstance(max(sweep_values), str) else min(sweep_values)*0.9
        # max_x = num_experiments + 1 if isinstance(max(sweep_values), str) else max(sweep_values)*1.1
        min_x = 0
        max_x = num_experiments + 1
        
        self.ax.set_xlim(min_x, max_x)

        # self.ax.legend(
        #     loc="lower center",
        #     bbox_to_anchor=(0.5, -0.3),
        # )

        if save_to_image:
            # self.fig.savefig(
            #     os.path.join(self._path, f"cost_scatter_plot.eps"),
            #     bbox_inches="tight",
            #     format="eps",  # Save to EPS
            # )
            self.fig.savefig(
                os.path.join(self._path, f"cost_scatter_plot.png"),
                bbox_inches="tight",
            )
        else:
            self.fig.show()
