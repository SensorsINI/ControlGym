import logging
import os
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
import platform
import tensorflow as tf
import torch

from yaml import FullLoader, load

config = load(open("config.yml", "r"), Loader=FullLoader)


class CustomFormatter(logging.Formatter):
    # Copied from CartPole repo
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
    logger.setLevel(getattr(import_module("logging"), config["1_data_generation"]["logging_level"]))
    # create console handler
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger


logger = get_logger(__name__)


def get_name_of_controller_module(controller_name: str) -> str:
    """Check if the controller name specified a controller within
    CartPoleSimulation repo or an internal one

    :param controller_name: Name of controller as in config.ymnl
    :type controller_name: str
    :return: name of module where to find the right controller or wrapper
    :rtype: str
    """
    if find_spec(f"CartPoleSimulation.Controllers.{controller_name}") is not None:
        logger.info(f"Using a CartPoleSimulation controller: {controller_name}")
        return "ControllerCartPoleSimulationImport"
    elif find_spec(f"ControllersGym.{controller_name}") is not None:
        logger.info(f"Using a ControlGym controller: {controller_name}")
        return controller_name
    else:
        raise ValueError(f"Passed an unknown controller name {controller_name}")


class OutputPath:
    RUN_NUM = None

    @classmethod
    def get_output_path(cls, timestamp: str, filename: str, suffix: str):
        controller_name = config["1_data_generation"]["controller_name"]
        env_name = config["1_data_generation"]["environment_name"]
        predictor_name = config["4_controllers"][controller_name]["predictor_name"]
        folder = os.path.join(
            "Output",
            f"{timestamp}_{controller_name}_{env_name}_{predictor_name}".replace(
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



class SeedMemory:
    seeds = []
    
    @classmethod
    def set_seeds(cls, seeds):
        cls.seeds = seeds
    
    @classmethod
    def get_seeds(cls):
        if len(cls.seeds) == 0:
            logger = get_logger(__name__)
            logger.warn("SeedMemory has nothing saved. Filling with dummy seeds.")
            return [1, 2, 3]
        return cls.seeds


### Below is copied from CartPole repo


if config["1_data_generation"]["debug"]:
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
