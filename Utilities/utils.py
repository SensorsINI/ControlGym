import logging
import os
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
import platform
from typing import Any
import tensorflow as tf
import torch

from yaml import FullLoader, load
from Control_Toolkit.others.environment import EnvironmentBatched

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
    logger.setLevel(
        getattr(import_module("logging"), config["1_data_generation"]["logging_level"])
    )
    # create console handler
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger


logger = get_logger(__name__)


class OutputPath:
    RUN_NUM = 1
    collection_folder_name = ""

    @classmethod
    def get_output_path(cls, timestamp: str, env_name: str, controller_name: str, predictor_name: str, filename: str, suffix: str) -> str:
        folder = os.path.join(
            "Output",
            cls.collection_folder_name,
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


class CurrentRunMemory:
    current_controller_name: str
    current_environment_name: str
    current_environment: EnvironmentBatched


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
