import numpy as np
from Utilities.utils import get_output_path
from Visualizations import Plotter
import matplotlib.pyplot as plt
from matplotlib import animation


class EnvironmentPlotter(Plotter):
    def plot(self, frames: list[np.ndarray]):
        if self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.clear()

        im = self.ax.imshow(frames[0], animated=True)

        def animate(k):
            im.set_array(frames[k])
            self.ax.set_title(f"Environment, Frame {k}")
            return [im]

        anim = animation.FuncAnimation(
            fig=self.fig,
            func=animate,
            frames=len(frames),
            interval=200,
            blit=True,
            repeat=False,
        )
        self.ax.get_xaxis().set_major_locator(plt.NullLocator())
        self.ax.get_yaxis().set_major_locator(plt.NullLocator())
        anim.save(
            get_output_path(self._timestamp, "environment.mp4"),
            writer=animation.FFMpegWriter(fps=15),
        )
