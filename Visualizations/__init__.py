from datetime import datetime


class Plotter:
    def __init__(self, **kwargs) -> None:
        self.fig, self.ax, self.axs, self.gs = None, None, None, None
        self._timestamp = kwargs.get(
            "timestamp_str", datetime.now().strftime("%Y%m%d-%H%M%S")
        )
