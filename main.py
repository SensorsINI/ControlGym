import os
import sys
import time
from datetime import datetime
from importlib import import_module
from typing import Any

import gym
import numpy as np
from numpy.random import SeedSequence
from yaml import dump

from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.others.environment import (EnvironmentBatched,
                                                TensorFlowLibrary)
from Environments import register_envs
from Utilities.csv_helpers import save_to_csv
from Utilities.generate_plots import generate_experiment_plots
from Utilities.utils import OutputPath, SeedMemory, get_logger

# Keep allowing absolute imports within CartPoleSimulation subgit
sys.path.append(os.path.join(os.path.abspath("."), "CartPoleSimulation"))

logger = get_logger(__name__)

# Gym API: Register custom environments
register_envs()


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

    # Loop through independent experiments
    for i in range(num_experiments):
        # Generate new seeds for environment and controller
        seeds = seed_sequences[i].generate_state(3)
        SeedMemory.set_seeds(seeds)
        
        controller_config = config["4_controllers"][controller_name]
        environment_config = config["2_environments"][environment_name]
        environment_config.update({"seed": int(seeds[0])})
        controller_config.update({"seed": int(seeds[2])})

        # Flatten global controller config values
        controller_config.update(
            {
                k: config["4_controllers"][k]
                for k in ["controller_logging", "mpc_horizon"]
            }
        )

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

        if render_mode == "human":
            matplotlib.use("Qt5Agg")
        else:
            matplotlib.use("Agg")

        env: EnvironmentBatched = gym.make(
            environment_name,
            **environment_config,
            computation_lib=TensorFlowLibrary,
            render_mode=render_mode,
        )
        obs = env.reset(seed=int(seeds[1]))
        assert len(env.action_space.shape) == 1, f"Action space needs to be a flat vector, is Box with shape {env.action_space.shape}"
        
        ##### --------------------------------------------- #####
        ##### ----------------- PREDICTOR ----------------- #####
        assert hasattr(env, "step_dynamics"), "Environment needs to have a stateless step_dynamics function"
        predictor_name = controller_config["predictor_name"]
        predictor_module = import_module(f"SI_Toolkit.Predictors.{predictor_name}")
        predictor = getattr(predictor_module, predictor_name)(
            horizon=controller_config["mpc_horizon"],
            dt=environment_config["dt"],
            intermediate_steps=controller_config["predictor_intermediate_steps"],
            disable_individual_compilation=True,
            batch_size=controller_config["num_rollouts"],
            net_name=controller_config["NET_NAME"],
            step_fun=env.step_dynamics,
        )
        
        ##### ------------------------------------------------- #####
        ##### ----------------- COST FUNCTION ----------------- #####
        cost_function_name = environment_config["cost_function"]
        cost_function_module = import_module(f"Control_Toolkit.Cost_Functions.{cost_function_name}")
        cost_function = getattr(cost_function_module, cost_function_name)(env)

        ##### ---------------------------------------------- #####
        ##### ----------------- CONTROLLER ----------------- #####
        controller_module = import_module(f"Control_Toolkit.Controllers.{controller_name}")
        controller: template_controller = getattr(controller_module, controller_name)(
            predictor=predictor,
            cost_function=cost_function,
            dt=environment_config["dt"],
            action_space=env.action_space,
            observation_space=env.observation_space,
            **controller_config,
        )

        ##### ----------------------------------------------------- #####
        ##### ----------------- MAIN CONTROL LOOP ----------------- #####
        frames = []
        start_time = time.time()
        num_iterations = config["1_data_generation"]["num_iterations"]
        for step in range(num_iterations):
            action = controller.step(obs)
            new_obs, reward, done, info = env.step(action)
            controller.current_log["realized_cost_logged"] = np.array([-reward])
            if config["4_controllers"]["controller_logging"]:
                controller.update_logs()
                env.set_logs(controller.logs)
            if config["1_data_generation"]["render_for_humans"]:
                env.render()
            elif config["1_data_generation"]["save_plots_to_file"]:
                frames.append(env.render())

            time.sleep(1e-6)

            if done:
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
        elif config["4_controllers"]["controller_logging"]:
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
                    OutputPath.get_output_path(timestamp_str, str(n), ".npy"),
                    "wb",
                ) as f:
                    np.save(f, a)
            # Save config
            with open(
                OutputPath.get_output_path(timestamp_str, "config", ".yml"), "w"
            ) as f:
                dump(config, f)
