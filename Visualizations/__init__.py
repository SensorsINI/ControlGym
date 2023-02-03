from Utilities.utils import CurrentRunMemory
import matplotlib.pyplot as plt
from matplotlib import gridspec


class Plotter:
    def __init__(self, path: str, run_config: dict, environment_config: dict, **kwargs) -> None:
        self.fig: plt.Figure = None
        self.ax: plt.Axes = None
        self.axs: "list[plt.Axes]" = None
        self._path = path
        self._config_to_disp: dict = {
            "Controller name": run_config["controller_name"],
            "Actuation stdev/range(actions)": environment_config["actuator_noise"],
        }

    def _display_some_config(self):
        self.fig.text(
            0,
            0,
            ", ".join(
                ["=".join([str(k), str(v)]) for k, v in self._config_to_disp.items()]
            ),
            horizontalalignment="left",
            verticalalignment="bottom",
        )
