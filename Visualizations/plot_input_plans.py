import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from Visualizations import Plotter
from Utilities.utils import get_output_path

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
        frames: list[np.ndarray],
        save_to_video: bool = True,
    ):
        if self.axs is None:
            self.fig, self.axs = plt.subplots(
                nrows=1,
                ncols=2,
                figsize=(8, 4),
                gridspec_kw={"wspace": 0.1, "top": 0.9, "bottom": 0.1},
            )
        for ax in self.axs:
            ax.clear()
            ax.set_box_aspect(1)

        num_steps, num_samples, horizon_length = actions.shape

        lines = [
            self.axs[0].plot(
                [], [], linestyle="-", linewidth=0.5, marker="x", alpha=1.0, color="b"
            )[0]
            for i in range(num_samples)
        ]
        im = self.axs[1].imshow(frames[0], animated=True)

        def init_animation():
            for line in lines:
                line.set_data([], [])
            return lines

        def animate(k):
            # Left side: Plot cost evolution
            a, b = np.min(costs[k, :]), np.max(costs[k, :])
            for i, line in enumerate(lines):
                line.set_data(
                    np.arange(horizon_length),
                    actions[k, i, :],
                )
                line.set_alpha((1.0 - float((costs[k, i] - a) / (b - a)))**2)
                if costs[k, i] == a:
                    line.set_color("r")
                    line.set_markerfacecolor("r")
                    line.set_zorder(3)
                else:
                    line.set_color("b")
                    line.set_markerfacecolor("b")
                    line.set_zorder(2)
            self.axs[0].set_xlim(0, horizon_length)
            self.axs[0].set_ylim(np.min(actions), np.max(actions))

            # Right side: Plot state of environment
            im.set_array(frames[k])

            # Set figure title
            self.fig.suptitle(f"Frame {k}")
            return [im] + lines

        anim = animation.FuncAnimation(
            fig=self.fig,
            func=animate,
            init_func=init_animation,
            frames=num_steps,
            interval=500,
            blit=True,
            repeat=False,
        )

        self.axs[0].get_xaxis().set_major_locator(
            plt.MaxNLocator(nbins=10, integer=True, min_n_ticks=2)
        )
        self.axs[0].set_ylabel("Control action")
        self.axs[0].set_xlabel("MPC horizon step")
        self.axs[0].set_title("Input plans per control iteration")

        self.axs[1].get_xaxis().set_major_locator(plt.NullLocator())
        self.axs[1].get_yaxis().set_major_locator(plt.NullLocator())
        self.axs[1].set_title(f"Environment")

        if save_to_video:
            anim.save(
                get_output_path(self._timestamp, "Q_logged.mp4"),
                writer=animation.FFMpegWriter(fps=15),
            )
        else:
            self.fig.show()
