from Utilities.utils import CurrentRunMemory


class Plotter:
    def __init__(self, timestamp: str, config: dict, **kwargs) -> None:
        self.fig, self.ax, self.axs, self.gs = None, None, None, None
        self._timestamp: str = timestamp
        self._config_to_disp: dict = {
            "Controller name": CurrentRunMemory.current_controller_name,
            "Actuation stdev/range(actions)": config["2_environments"][
                CurrentRunMemory.current_environment_name
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
