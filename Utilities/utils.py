import logging
import os
from importlib import import_module
from pathlib import Path
import platform
import tensorflow as tf
import torch

from yaml import FullLoader, load

config = load(open("config.yml", "r"), Loader=FullLoader)


class OutputPath:
    RUN_NUM = None

    @classmethod
    def get_output_path(cls, timestamp: str, filename: str, suffix: str):
        if config['controller_name'] == "ControllerCartPoleSimulationImport":
            config['controller_name'] = config["controllers"]["ControllerCartPoleSimulationImport"]["controller"].replace("-", "_")
        folder = os.path.join(
            "Output",
            f"{timestamp}_{config['controller_name']}_{config['environment_name']}".replace(
                "/", "_"
            ),
        )
        Path(folder).mkdir(parents=True, exist_ok=True)
        return os.path.join(
            folder,
            f"{timestamp}_{filename}{suffix}"
            if cls.RUN_NUM is None
            else f"{timestamp}_{filename}_{cls.RUN_NUM}{suffix}",
        )


# Below is copied from CartPole repo
class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(import_module("logging"), config["logging_level"]))
    # create console handler
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger

class SeedMemory:
    seeds = []


### Below is copied from CartPole repo


if config["debug"]:
    CompileTF = lambda func: func
    CompileTorch = lambda func: func
else:
    if (
        platform.machine() == "arm64" and platform.system() == "Darwin"
    ):  # For M1 Apple processor
        CompileTF = tf.function
    else:
        # tf.function: jit_compile=True uses nondeterministic random seeds, see https://tensorflow.org/xla/known_issues
        CompileTF = lambda func: tf.function(func=func)
    CompileTorch = torch.jit.script
