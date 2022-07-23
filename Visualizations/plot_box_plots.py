import os
import numpy as np
from Utilities.utils import OutputPath
from Visualizations import Plotter
import matplotlib.pyplot as plt

class BoxPlotPlotter(Plotter):
    def plot(self, costs: np.ndarray, save_to_image):
        if self.ax is None:
            self.fig, self.ax = plt.subplots(
                figsize=(10, 8), gridspec_kw={"wspace": 0.1, "top": 0.9, "bottom": 0.1}
            )
        self.ax.clear()
        
        self.ax.boxplot([c for c in costs.values()])
        
        self.ax.set_ylabel("Total horizon cost")
        self.ax.set_xlabel("Control iteration")
        self.ax.set_title("Total cost of input plans per global control iteration")

        if save_to_image:
            self.fig.savefig(
                os.path.join("Output", self._timestamp),
                bbox_inches="tight",
            )
        else:
            self.fig.show()