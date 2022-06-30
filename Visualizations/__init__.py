from datetime import datetime

import numpy as np


class Plotter:
    def __init__(self, **kwargs) -> None:
        self.fig, self.ax = None, None
        self._timestamp = kwargs.get(
            "timestamp_str", datetime.now().strftime("%Y%m%d-%H%M%S")
        )
