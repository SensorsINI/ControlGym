import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from Visualizations import Plotter
from Utilities.utils import OutputPath

plt.style.use(["science"])


class SummaryPlotter(Plotter):
    """
    Saves a plot of a control episode.
    The x-axis of each subplot is the control iteration.
    The y-axis is the value of a state/input variable.
    In the top row, it plots one subplot for each state variable. Going from left to right within the subplot, you can see how it evolves during the episode.
    In the bottom row, it plots one subplot for each input variable. Again, each subplot shows the input over time during the control episode.
    """
    def plot(self, states: np.ndarray, actions: np.ndarray, save_to_image: bool = True):
        assert states.shape[0] == actions.shape[0]

        num_steps, n = states.shape
        if actions.ndim == 1:
            actions = actions[:, np.newaxis]
        m = actions.shape[-1]
        
        if self.axs is None:
            self.fig = plt.figure(figsize=(12, 8))
            self.gs = gridspec.GridSpec(
                nrows=2, ncols=m * n, figure=self.fig, wspace=0.2, hspace=0.2
            )
            self.axs = [[], []]
            for i in range(n):
                self.axs[0].append(
                    self.fig.add_subplot(self.gs[0, (i * m) : ((i + 1) * m)])
                )
            for i in range(m):
                self.axs[1].append(
                    self.fig.add_subplot(self.gs[1, (i * n) : ((i + 1) * n)])
                )
        for ax in [_x1 for _x2 in self.axs for _x1 in _x2]:
            ax.clear()

        for i in range(n):
            self.axs[0][i].plot(
                np.arange(num_steps),
                states[:, i],
                linestyle="-",
                linewidth=0.5,
                marker="o",
                markersize=2,
                alpha=1.0,
                color="b",
            )
        for i in range(m):
            self.axs[1][i].plot(
                np.arange(num_steps),
                actions[:, i],
                linestyle="-",
                linewidth=0.5,
                marker="o",
                markersize=2,
                alpha=1.0,
                color="b",
            )

        for i in range(2):
            for j, ax in enumerate(self.axs[i]):
                ax.get_xaxis().set_major_locator(
                    plt.MaxNLocator(nbins=10, integer=True, min_n_ticks=2)
                )
                if i == 0:
                    ax.set_ylabel(f"$x_{j}$")
                elif i == 1:
                    ax.set_ylabel(f"$u_{j}$")
                ax.set_xlabel("Iteration")

        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.1)
        self.fig.suptitle("State 's' and Action 'u' Evolution")
        self._display_some_config()
        
        c = 1
        p = f"summary_logged_{c}.svg"
        while os.path.isfile(os.path.join(self._path, p)):
            c += 1
            p = f"summary_logged_{c}.svg"

        if save_to_image:
            self.fig.savefig(
                os.path.join(self._path, p),
                bbox_inches="tight",
            )
        else:
            self.fig.show()
