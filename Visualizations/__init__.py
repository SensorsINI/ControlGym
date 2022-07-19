class Plotter:
    def __init__(self, timestamp: str, config: dict, **kwargs) -> None:
        self.fig, self.ax, self.axs, self.gs = None, None, None, None
        self._timestamp = timestamp
        self._config_to_disp = {
            "Controller name": config["data_generation"]["controller_name"],
            "Actuation stdev/range(actions)": config["environments"][
                config["data_generation"]["environment_name"]
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
