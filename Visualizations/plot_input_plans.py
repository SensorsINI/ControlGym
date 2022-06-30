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
    def plot(self, actions: np.ndarray, costs: np.ndarray, save_to_video: bool = True):
        if self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.clear()
        num_steps, num_samples, horizon_length = actions.shape

        c = _build_color_seq(num_samples)
        lines = [
            self.ax.plot(
                [], [], linestyle="-", linewidth=0.5, marker="x", alpha=1.0, color="b"
            )[0]
            for i in range(num_samples)
        ]

        def init_animation():
            for line in lines:
                line.set_data([], [])
            return lines

        def animate(k):
            a, b = np.min(costs[k, :]), np.max(costs[k, :])
            for i, line in enumerate(lines):
                line.set_data(
                    np.arange(horizon_length),
                    actions[k, i, :],
                )
                line.set_alpha(1.0 - float((costs[k, i] - a) / (b - a)))
            self.ax.set_ylabel(f"Control action, Iteration {k}")
            self.ax.set_xlim(0, horizon_length)
            self.ax.set_ylim(np.min(actions), np.max(actions))
            return lines

        anim = animation.FuncAnimation(
            fig=self.fig,
            func=animate,
            init_func=init_animation,
            frames=num_steps,
            interval=500,
            blit=True,
            repeat=False,
        )
        self.ax.get_xaxis().set_major_locator(plt.MaxNLocator(nbins=10, integer=True, min_n_ticks=2))
        self.ax.set_ylabel("Control action")
        self.ax.set_xlabel("MPC horizon step")
        self.ax.set_title("Input plans per control iteration")
        # self.fig.tight_layout()
        if save_to_video:
            anim.save(
                get_output_path(self._timestamp, "Q_logged.mp4"),
                writer=animation.FFMpegWriter(fps=1),
            )
        else:
            self.fig.show()
