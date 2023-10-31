import os
import sys
import time
import csv
from datetime import datetime
from importlib import import_module


from typing import Any
import gymnasium as gym
import numpy as np
import tensorflow as tf
from numpy.random import SeedSequence
from yaml import dump

from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper
from Control_Toolkit.others.environment import EnvironmentBatched
from Environments import ENV_REGISTRY, register_envs
from SI_Toolkit.computation_library import TensorFlowLibrary
from Utilities.csv_helpers import save_to_csv
from Utilities.generate_plots import generate_experiment_plots
from Utilities.utils import ConfigManager, CurrentRunMemory, OutputPath, SeedMemory, get_logger, nested_assignment_to_ordereddict



sys.path.append(os.path.join(os.path.abspath("."),
                             "CartPoleSimulation"))  # Keep allowing absolute imports within CartPoleSimulation subgit
register_envs()  # Gym API: Register custom environments
logger = get_logger(__name__)

def prepare_run(path_to_controlgym="."):
    import ruamel.yaml

    # Create a config manager which looks for '.yml' files within the list of folders specified.
    # Rationale: We want GUILD AI to be able to update values in configs that we include in this list.
    # We might intentionally want to exclude the path to a folder which does contain configs but should not be overwritten by GUILD.
    config_default_locations = ["", "Control_Toolkit_ASF", "SI_Toolkit_ASF", "Environments"]
    config_locations = [os.path.join(path_to_controlgym, x) for x in config_default_locations]
    config_manager = ConfigManager(*config_locations)

    # Scan for any custom parameters that should overwrite the toolkits' config files:
    submodule_configs_default_locations = [".", "Control_Toolkit_ASF", "SI_Toolkit_ASF", "Environments"]
    submodule_configs_locations = [os.path.join(path_to_controlgym, x) for x in submodule_configs_default_locations]
    submodule_configs = ConfigManager(*submodule_configs_locations).loaders
    for base_name, loader in submodule_configs.items():
        if base_name in config_manager("config").get("custom_config_overwrites", {}):
            data: ruamel.yaml.comments.CommentedMap = loader.load()
            update_dict = config_manager("config")["custom_config_overwrites"][base_name]
            nested_assignment_to_ordereddict(data, update_dict)
            loader.overwrite_config(data)

    # Retrieve required parameters from config:
    CurrentRunMemory.current_controller_name = config_manager("config")["controller_name"]
    CurrentRunMemory.current_environment_name = config_manager("config")["environment_name"]

    return config_manager, CurrentRunMemory


class EnvManager:
    def __init__(self,
                 controller_name: str,
                 environment_name: str,
                 config_manager: ConfigManager,
                 run_for_ML_Pipeline=False,
                 record_path=None,
                 num_experiments=None,
                 num_iterations=None,
                 concat_state_and_attributes=False,
                 ):

        self.CRM = CurrentRunMemory
        self.controller_name = controller_name
        self.environment_name = environment_name
        self.config_manager = config_manager
        self.run_for_ML_Pipeline = run_for_ML_Pipeline
        self.record_path = record_path
        self.concat_state_and_attributes = concat_state_and_attributes

        # Generate seeds and set timestamp
        timestamp = datetime.now()
        seed_entropy = self.config_manager("config")["seed_entropy"]
        if seed_entropy is None:
            seed_entropy = int(timestamp.timestamp())
            logger.info("No seed entropy specified. Setting to posix timestamp.")

        if num_experiments is None:
            self.num_experiments = self.config_manager("config")["num_experiments"]
        else:
            self.num_experiments = num_experiments

        if num_iterations is None:
            self.num_iterations = self.config_manager("config")["num_iterations"]
        else:
            self.num_iterations = num_iterations

        self.seed_sequences = SeedSequence(entropy=seed_entropy).spawn(self.num_experiments)
        self.timestamp_str = timestamp.strftime("%Y%m%d-%H%M%S")

        if self.run_for_ML_Pipeline:
            # Get training/validation split
            self.frac_train, self.frac_val = self.config_manager("config")["split"]
            assert self.record_path is not None, "If ML mode is on, need to provide record_path."

        self.controller_short_name = self.controller_name.replace("controller_", "").replace("_", "-")
        self.optimizer_short_name = self.config_manager("config_controllers")[self.controller_short_name]["optimizer"]
        optimizer_name = "optimizer_" + self.optimizer_short_name.replace("-", "_")
        self.CRM.current_optimizer_name = optimizer_name
        self.all_metrics = dict(
            total_rewards=[],
            timeout=[],
            terminated=[],
            truncated=[],
        )

        self.i = 0
        self.experiment_step = 0

        self.frames = []
        self.start_time = None

        self.env = None
        self.controller = None

        self.sorted_attributes_keys = []

        self.config_controller = None

        self.obs = None
        self.truncated = None
        self.terminated = None

        self.all_rewards = []

    def reset(self):
        if self.i != 0:
            self.i += 1
        self.experiment_step = 0

        # Generate new seeds for environment and controller
        seeds = self.seed_sequences[self.i].generate_state(3)
        SeedMemory.set_seeds(seeds)

        self.config_controller = dict(self.config_manager("config_controllers")[self.controller_short_name])
        config_optimizer = dict(self.config_manager("config_optimizers")[self.optimizer_short_name])
        config_optimizer.update({"seed": int(seeds[1])})
        config_environment = dict(self.config_manager("config_environments")[self.environment_name])
        config_environment.update({"seed": int(seeds[0])})
        self.all_rewards = []

        ##### ----------------------------------------------- #####
        ##### ----------------- ENVIRONMENT ----------------- #####
        ##### --- Instantiate environment and call reset ---- #####
        if self.config_manager("config")["render_for_humans"]:
            render_mode = "human"
        elif self.config_manager("config")["save_plots_to_file"]:
            render_mode = "rgb_array"
        else:
            render_mode = None

        import matplotlib

        matplotlib.use("Agg")

        self.env: EnvironmentBatched = gym.make(
            self.environment_name,
            **config_environment,
            computation_lib=TensorFlowLibrary,
            render_mode=render_mode,
        )
        self.CRM.current_environment = self.env
        self.obs, obs_info = self.env.reset(seed=config_environment["seed"])
        assert len(
            self.env.action_space.shape) == 1, f"Action space needs to be a flat vector, is Box with shape {self.env.action_space.shape}"

        self.frames = []
        self.start_time = time.time()

        ##### ---------------------------------------------- #####
        ##### ----------------- CONTROLLER ----------------- #####
        controller_module = import_module(f"Control_Toolkit.Controllers.{self.controller_name}")
        self.controller: template_controller = getattr(controller_module, self.controller_name)(
            dt=self.env.dt,
            environment_name=ENV_REGISTRY[self.environment_name].split(":")[-1],
            control_limits=(self.env.action_space.low, self.env.action_space.high),
            initial_environment_attributes=self.env.environment_attributes,
        )
        self.controller.configure(optimizer_name=self.optimizer_short_name,
                             predictor_specification=self.config_controller["predictor_specification"])


    def step(self, action):
        if self.experiment_step != 0:
            self.experiment_step += 1
        new_obs, reward, self.terminated, self.truncated, info = self.env.step(action)
        c_fun: CostFunctionWrapper = getattr(self.controller, "cost_function", None)
        if c_fun is not None:
            assert isinstance(c_fun, CostFunctionWrapper)
            # Compute reward from the cost function that the controller optimized
            reward = -float(c_fun.get_stage_cost(
                tf.convert_to_tensor(new_obs[np.newaxis, np.newaxis, ...]),  # Add batch / MPC horizon dimensions
                tf.convert_to_tensor(action[np.newaxis, np.newaxis, ...]),
                None
            ))
            self.all_rewards.append(reward)
        if self.config_controller.get("controller_logging", False):
            self.controller.logs["realized_cost_logged"].append(np.array([-reward]).copy())
            self.env.set_logs(self.controller.logs)
        logger.debug(
            f"\nStep          : {self.experiment_step + 1}/{self.num_iterations}\nObservation   : {self.obs}\nPlanned Action: {action}\n"
        )
        self.obs = new_obs

        if self.concat_state_and_attributes:
            observation = np.copy(self.obs)
            self.sorted_attributes_keys = np.sort(self.env.environment_attributes.keys())
            for key in self.sorted_attributes_keys:
                np.append(observation, self.enc.environment_attributesp[key])
        else:
            observation = self.obs

        # If the episode is up, start a new experiment
        if self.truncated:
            logger.info(f"Episode truncated (failure)")
        elif self.terminated:
            logger.info(f"Episode terminated successfully")

        return observation, reward, self.terminated, self.truncated, info

    def render(self):
        if self.config_manager("config")["render_for_humans"]:
            self.env.render()
        elif self.config_manager("config")["save_plots_to_file"]:
            self.frames.append(self.env.render())

        time.sleep(1e-6)

    def close(self):
        # Print compute time statistics
        end_time = time.time()
        control_freq = self.num_iterations / (end_time - self.start_time)
        logger.debug(
            f"Achieved average control frequency of {round(control_freq, 2)}Hz ({round(1.0e3 / control_freq, 2)}ms per iteration)")

        # Close the env
        self.env.close()

        ##### ----------------------------------------------------- #####
        ##### ----------------- LOGGING AND PLOTS ----------------- #####
        OutputPath.RUN_NUM = self.i + 1
        controller_output = self.controller.get_outputs()
        self.all_metrics["total_rewards"].append(np.mean(self.all_rewards))
        self.all_metrics["timeout"].append(float(not (self.terminated or self.truncated)))
        self.all_metrics["terminated"].append(float(self.terminated))
        self.all_metrics["truncated"].append(float(self.truncated))

        if self.run_for_ML_Pipeline:
            # Save data as csv
            if self.i < int(self.frac_train * self.num_experiments):
                csv_path = os.path.join(self.record_path, "Train")
            elif self.i < int((self.frac_train + self.frac_val) * self.num_experiments):
                csv_path = os.path.join(self.record_path, "Validate")
            else:
                csv_path = os.path.join(self.record_path, "Test")
            os.makedirs(csv_path, exist_ok=True)
            save_to_csv(self.config_manager("config"), self.controller, self.environment_name, csv_path)
        elif self.config_controller.get("controller_logging", False):
            if self.config_manager("config")["save_plots_to_file"]:
                # Generate and save plots in default location
                generate_experiment_plots(
                    config=self.config_manager("config"),
                    environment_config=self.config_manager("config_environments")[self.environment_name],
                    controller_output=controller_output,
                    timestamp=self.timestamp_str,
                    frames=self.frames if len(self.frames) > 0 else None,
                )
            # Save .npy files
            for n, a in controller_output.items():
                with open(
                        OutputPath.get_output_path(self.timestamp_str, f"{str(n)}.npy"),
                        "wb",
                ) as f:
                    np.save(f, a)
            # Save configs
            for loader in self.config_manager.loaders.values():
                with open(
                        OutputPath.get_output_path(self.timestamp_str, loader.name), "w"
                ) as f:
                    dump(loader.config, f)

    def summary(self):
        # Dump all saved scalar metrics as csv
        with open(
                OutputPath.get_output_path(self.timestamp_str, f"output_scalars.csv"),
                "w",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(self.all_metrics.keys())
            writer.writerows(zip(*self.all_metrics.values()))
        # These output metrics are detected by GUILD AI and follow a "key: value" format
        print("Output metrics:")
        print(f"Mean total reward: {np.mean(self.all_metrics['total_rewards'])}")
        print(f"Stdev of reward: {np.std(self.all_metrics['total_rewards'])}")
        print(f"Timeout rate: {np.mean(self.all_metrics['timeout'])}")
        print(f"Terminated rate: {np.mean(self.all_metrics['terminated'])}")
        print(f"Truncated rate: {np.mean(self.all_metrics['truncated'])}")
