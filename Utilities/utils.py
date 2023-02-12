from collections import OrderedDict
from glob import glob
import logging
import os
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
import platform
from typing import Any, Optional
import tensorflow as tf
import torch

from yaml import FullLoader, load, safe_load
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
        getattr(import_module("logging"), config["logging_level"])
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
    def get_output_path(cls, timestamp: str, file_name: Optional[str]=None) -> str:
        folder = os.path.join(
            "Output",
            cls.collection_folder_name,
            timestamp,
        )
        Path(folder).mkdir(parents=True, exist_ok=True)
        if file_name is not None:
            f, suffix = file_name.split(".")
            suffix = f".{suffix}"
            fn = f"{timestamp}_{f}_{suffix}" if cls.RUN_NUM is None else f"{timestamp}_{f}_{cls.RUN_NUM}{suffix}"
        else:
            fn = ""
        return os.path.join(folder, fn)


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
    current_optimizer_name: str
    current_environment_name: str
    current_environment: EnvironmentBatched
    

class ConfigManager:
    """Detects configuration files in project directory recursively within specified folders and spawns loaders for each"""
    def __init__(self, *paths_to_folders) -> None:
        config_files = []
        for path in paths_to_folders:
            config_files.extend(glob(os.path.join(path, "*config*.yml"), recursive=True))
        
        self._config_loaders = {str(os.path.basename(filename)).replace(".yml", ""): CustomLoader(filename) for filename in config_files}
        for name, custom_loader in self._config_loaders.items():
            setattr(self, name, custom_loader)
            
    def update_configs(self):
        for config_loader in self._config_loaders.values():
            config_loader.update_from_config()
    
    def set_config(self, config_name: str, updated_config: dict) -> None:
        config_loader: Optional[CustomLoader] = self._config_loaders.get(config_name, None)
        config_loader.config = updated_config
    
    @property
    def loaders(self):
        return self._config_loaders
    
    def __call__(self, config_name: str) -> Any:
        """Get the config with specified name (without '.yml' suffix) if it is loaded."""
        config_loader: Optional[CustomLoader] = self._config_loaders.get(config_name, None)
        if config_loader is None:
            raise ValueError(f"No configuration with name {config_name} loaded.")
        return config_loader.config


import ruamel.yaml
ruamel_yaml = ruamel.yaml.YAML()
class CustomLoader:
    """
    Class that loads a yaml, observes for changes and updates the output config
    """
    
    def __init__(self, path: str) -> None:
        self.path = path
        self.name = os.path.basename(path)
        self.load_config_from_file()
    
    def load(self):
        with open(self.path) as fp:
            data: ruamel.yaml.comments.CommentedMap = ruamel_yaml.load(fp)
        return data

    def overwrite_config(self, data):
        with open(self.path, "w") as fp:
            ruamel_yaml.dump(data, fp)
    
    @property
    def config(self):
        # self.load_config_from_file()
        return self._config
    
    @config.setter
    def config(self, new_config: dict):
        self._config = new_config.copy()
    
    def load_config_from_file(self):
        self._config = dict(safe_load(open(self.path, "r")))


def nested_conversion_to_ordereddict(d):
    if isinstance(d, dict):
        return OrderedDict({k: nested_conversion_to_ordereddict(v) for k, v in d.items()})
    else:
        return d


def nested_assignment_to_ordereddict(target: OrderedDict, source: dict):
    # In-place update of the OrderedDict `target` with values from the regular dictionary `source`
    for k, v in source.items():
        if k not in target:
            raise ValueError(f"Trying to re-assign target dictionary at key {k} which does not exist.")
        if isinstance(v, dict):
            nested_assignment_to_ordereddict(target[k], v)
        else:
            target[k] = v


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
