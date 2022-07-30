from Utilities.utils import CurrentRunMemory
import matplotlib.pyplot as plt
from matplotlib import gridspec


class Plotter:
    def __init__(self, timestamp: str, config: dict, **kwargs) -> None:
        self.fig: plt.Figure = None
        self.ax: plt.Axes = None
        self.axs: "list[plt.Axes]" = None
        self._timestamp: str = timestamp
        self._config_to_disp: dict = {
            "Controller name": config["1_data_generation"]["controller_name"],
            "Actuation stdev/range(actions)": config["2_environments"][
                config["1_data_generation"]["environment_name"]
            ]["actuator_noise"],
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
