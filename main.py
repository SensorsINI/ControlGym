import os
import sys
import time
from datetime import datetime
from importlib import import_module
from typing import Any

import gym
import numpy as np
from numpy.random import SeedSequence
from yaml import FullLoader, dump, load

from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.others.environment import EnvironmentBatched
from Environments import ENV_REGISTRY, register_envs
from SI_Toolkit.computation_library import TensorFlowLibrary
from Utilities.csv_helpers import save_to_csv
from Utilities.generate_plots import generate_experiment_plots
from Utilities.utils import CurrentRunMemory, OutputPath, SeedMemory, get_logger

# Keep allowing absolute imports within CartPoleSimulation subgit
sys.path.append(os.path.join(os.path.abspath("."), "CartPoleSimulation"))

logger = get_logger(__name__)

# Gym API: Register custom environments
register_envs()


config_controllers = load(open(os.path.join("Control_Toolkit_ASF", "config_controllers.yml"), "r"), Loader=FullLoader)
config_cost_function = load(open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"), Loader=FullLoader)
config_optimizers = load(open(os.path.join("Control_Toolkit_ASF", "config_optimizers.yml"), "r"), Loader=FullLoader)


def run_data_generator(
    controller_name: str,
    environment_name: str,
    num_experiments: int,
    config: "dict[str, Any]",
    run_for_ML_Pipeline=False,
    record_path=None,
):
    # Generate seeds and set timestamp
    timestamp = datetime.now()
    seed_entropy = config["1_data_generation"]["seed_entropy"]
    if seed_entropy is None:
        seed_entropy = int(timestamp.timestamp())
        logger.info("No seed entropy specified. Setting to posix timestamp.")

    seed_sequences = SeedSequence(entropy=seed_entropy).spawn(num_experiments)
    timestamp_str = timestamp.strftime("%Y%m%d-%H%M%S")

    if run_for_ML_Pipeline:
        # Get training/validation split
        frac_train, frac_val = config["1_data_generation"]["split"]
        assert record_path is not None, "If ML mode is on, need to provide record_path."

    # Set current controller/env names in config which is saved later
    config["1_data_generation"].update(
        {"controller_name": controller_name, "environment_name": environment_name}
    )
    controller_short_name = controller_name.replace("controller_", "").replace("_", "-")

    # Loop through independent experiments
    for i in range(num_experiments):
        # Generate new seeds for environment and controller
        seeds = seed_sequences[i].generate_state(3)
        SeedMemory.set_seeds(seeds)
        
        config_controller = dict(config_controllers[controller_short_name])
        config_optimizer = dict(config_optimizers[config_controller["optimizer"]])
        config_optimizer.update({"seed": int(seeds[1])})
        config_environment = dict(config["2_environments"][environment_name])
        config_environment.update({"seed": int(seeds[0])})

        ##### ----------------------------------------------- #####
        ##### ----------------- ENVIRONMENT ----------------- #####
        ##### --- Instantiate environment and call reset ---- #####
        if config["1_data_generation"]["render_for_humans"]:
            render_mode = "human"
        elif config["1_data_generation"]["save_plots_to_file"]:
            render_mode = "rgb_array"
        else:
            render_mode = None

        import matplotlib

        matplotlib.use("Agg")

        env: EnvironmentBatched = gym.make(
            environment_name,
            **config_environment,
            computation_lib=TensorFlowLibrary,
            render_mode=render_mode,
        )
        CurrentRunMemory.current_environment = env
        obs, obs_info = env.reset(seed=config_environment["seed"])
        assert len(env.action_space.shape) == 1, f"Action space needs to be a flat vector, is Box with shape {env.action_space.shape}"
        
        ##### ---------------------------------------------- #####
        ##### ----------------- CONTROLLER ----------------- #####
        controller_module = import_module(f"Control_Toolkit.Controllers.{controller_name}")
        controller: template_controller = getattr(controller_module, controller_name)(
            dt=env.dt,
            environment_name=ENV_REGISTRY[environment_name].split(":")[-1],
            num_states=env.observation_space.shape[0],
            num_control_inputs=env.action_space.shape[0],
            control_limits=(env.action_space.low, env.action_space.high),
            initial_environment_attributes=env.environment_attributes,
        )
        controller.configure(optimizer_name=config_controller["optimizer"], predictor_specification=config_controller["predictor_specification"])

        ##### ----------------------------------------------------- #####
        ##### ----------------- MAIN CONTROL LOOP ----------------- #####
        frames = []
        start_time = time.time()
        num_iterations = config["1_data_generation"]["num_iterations"]
        for step in range(num_iterations):
            action = controller.step(obs, updated_attributes=env.environment_attributes)
            new_obs, reward, terminated, truncated, info = env.step(action)
            if config_controller.get("controller_logging", False):
                controller.logs["realized_cost_logged"].append(np.array([-reward]).copy())
                env.set_logs(controller.logs)
            if config["1_data_generation"]["render_for_humans"]:
                env.render()
            elif config["1_data_generation"]["save_plots_to_file"]:
                frames.append(env.render())

            time.sleep(1e-6)

            if terminated or truncated:
                # If the episode is up, start a new experiment
                break

            logger.debug(
                f"\nStep          : {step+1}/{num_iterations}\nObservation   : {obs}\nPlanned Action: {action}\n"
            )
            obs = new_obs
        
        # Print compute time statistics
        end_time = time.time()
        control_freq = num_iterations / (end_time - start_time)
        logger.debug(f"Achieved average control frequency of {round(control_freq, 2)}Hz ({round(1.0e3/control_freq, 2)}ms per iteration)")

        # Close the env
        env.close()

        ##### ----------------------------------------------------- #####
        ##### ----------------- LOGGING AND PLOTS ----------------- #####
        OutputPath.RUN_NUM = i + 1
        controller_output = controller.get_outputs()

        if run_for_ML_Pipeline:
            # Save data as csv
            if i < int(frac_train * num_experiments):
                csv = os.path.join(record_path, "Train")
            elif i < int((frac_train + frac_val) * num_experiments):
                csv = os.path.join(record_path, "Validate")
            else:
                csv = os.path.join(record_path, "Test")
            os.makedirs(csv, exist_ok=True)
            save_to_csv(config, controller, environment_name, csv)
        elif config_controller.get("controller_logging", False):
            if config["1_data_generation"]["save_plots_to_file"]:
                # Generate and save plots in default location
                generate_experiment_plots(
                    config=config,
                    controller_output=controller_output,
                    timestamp=timestamp_str,
                    frames=frames if len(frames) > 0 else None,
                )
            # Save .npy files 
            for n, a in controller_output.items():
                with open(
                    OutputPath.get_output_path(timestamp_str, environment_name, controller_name, config_controller["predictor_specification"], str(n), ".npy"),
                    "wb",
                ) as f:
                    np.save(f, a)
            # Save config
            with open(
                OutputPath.get_output_path(timestamp_str, environment_name, controller_name, config_controller["predictor_specification"], "config", ".yml"), "w"
            ) as f:
                dump(config, f)
