import matplotlib
matplotlib.use("Agg")

import itertools
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from Visualizations import Plotter
from Utilities.utils import OutputPath

plt.style.use(["science"])


def _build_color_seq(n):
    colors = ["r", "g", "b", "k"]
    L = len(colors)
    return [colors[i % L] for i in range(n)]


class InputPlanPlotter(Plotter):
    def plot(
        self,
        actions: np.ndarray,
        costs: np.ndarray,
        frames: "list[np.ndarray]",
        save_to_video: bool = True,
    ):
        actions = np.expand_dims(actions, 3) if actions.ndim == 3 else actions
        num_steps, num_samples, horizon_length, num_actions = actions.shape

        if self.fig is None:
            self.fig = plt.figure(figsize=(8, 4), layout="constrained")
            self.subfig_l, self.subfig_r = self.fig.subfigures(1, 2, wspace=0.1)
            self.axs_l = self.subfig_l.subplots(
                nrows=num_actions,
                ncols=1,
                sharex=True,
                squeeze=True,
                gridspec_kw={"top": 0.9, "bottom": 0.15},
            )
            self.axs_r = self.subfig_r.subplots(
                nrows=1,
                ncols=1,
                squeeze=True,
                gridspec_kw={"top": 0.9, "bottom": 0.15},
            )
            self.axs_l = (
                [self.axs_l] if isinstance(self.axs_l, plt.Axes) else self.axs_l
            )
            self.axs_r = (
                [self.axs_r] if isinstance(self.axs_r, plt.Axes) else self.axs_r
            )
        for ax in itertools.chain(self.axs_l, self.axs_r):
            ax.clear()
        self.axs_r[0].set_box_aspect(1)

        # One set of lines per action component
        lines = [
            [
                ax.plot(
                    [],
                    [],
                    linestyle="-",
                    linewidth=0.5,
                    marker="x",
                    alpha=1.0,
                    color="b",
                )[0]
                for _ in range(num_samples)
            ]
            for ax in self.axs_l
        ]
        im = self.axs_r[0].imshow(frames[0], animated=True)

        def init_animation():
            for m in lines:
                for line in m:
                    line.set_data([], [])
            return [im] + [line for sublist in lines for line in sublist]

        def animate(k):
            # Left side: Plot cost evolution
            a, b = np.min(costs[k, :]), np.max(costs[k, :])
            for na, p in enumerate(lines):
                for i, line in enumerate(p):
                    line.set_data(
                        np.arange(horizon_length),
                        actions[k, i, :, na],
                    )
                    line.set_alpha(
                        (1.0 - float((costs[k, i] - a) / max(0.01, b - a))) ** 2
                    )
                    if costs[k, i] == a:
                        line.set_color("r")
                        line.set_markerfacecolor("r")
                        line.set_zorder(3)
                    else:
                        line.set_color("b")
                        line.set_markerfacecolor("b")
                        line.set_zorder(2)

            for na in range(num_actions):
                self.axs_l[na].set_xlim(0, horizon_length)
                self.axs_l[na].set_ylim(np.min(actions[..., na]), np.max(actions[..., na]))

            # Right side: Plot state of environment
            im.set_array(frames[k])

            # Set figure title
            self.fig.suptitle(f"Frame {k}")
            return [im] + [line for p in lines for line in p]

        anim = animation.FuncAnimation(
            fig=self.fig,
            func=animate,
            init_func=init_animation,
            frames=num_steps,
            interval=500,
            blit=True,
            repeat=False,
        )

        self.axs_l[-1].get_xaxis().set_major_locator(
            plt.MaxNLocator(nbins=10, integer=True, min_n_ticks=2)
        )
        for k, ax in enumerate(self.axs_l):
            ax.set_ylabel(f"Control action u_{k}")
        self.axs_l[-1].set_xlabel("MPC horizon step")
        self.axs_l[-1].set_title("Input plans per control iteration")

        self.axs_r[0].get_xaxis().set_major_locator(plt.NullLocator())
        self.axs_r[0].get_yaxis().set_major_locator(plt.NullLocator())
        self.axs_r[0].set_title(f"Environment")

        self._display_some_config()

        if save_to_video:
            anim.save(
                OutputPath.get_output_path(self._timestamp, "Q_logged", ".mp4"),
                writer=animation.FFMpegWriter(fps=15),
            )
        else:
            self.fig.show()
