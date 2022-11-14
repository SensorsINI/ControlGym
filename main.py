import os
import sys
import time
from datetime import datetime
from importlib import import_module

from typing import Any
import gym
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


sys.path.append(os.path.join(os.path.abspath("."), "CartPoleSimulation"))  # Keep allowing absolute imports within CartPoleSimulation subgit
register_envs()  # Gym API: Register custom environments
logger = get_logger(__name__)


def run_data_generator(
    controller_name: str,
    environment_name: str,
    config_manager: ConfigManager,
    run_for_ML_Pipeline=False,
    record_path=None,
):
    # Generate seeds and set timestamp
    timestamp = datetime.now()
    seed_entropy = config_manager("config")["seed_entropy"]
    if seed_entropy is None:
        seed_entropy = int(timestamp.timestamp())
        logger.info("No seed entropy specified. Setting to posix timestamp.")

    num_experiments = config_manager("config")["num_experiments"]
    seed_sequences = SeedSequence(entropy=seed_entropy).spawn(num_experiments)
    timestamp_str = timestamp.strftime("%Y%m%d-%H%M%S")

    if run_for_ML_Pipeline:
        # Get training/validation split
        frac_train, frac_val = config_manager("config")["split"]
        assert record_path is not None, "If ML mode is on, need to provide record_path."

    controller_short_name = controller_name.replace("controller_", "").replace("_", "-")
    optimizer_short_name = config_manager("config_controllers")[controller_short_name]["optimizer"]
    optimizer_name = "optimizer_" + optimizer_short_name.replace("-", "_")
    CurrentRunMemory.current_optimizer_name = optimizer_name
    all_mean_rewards = []
    all_steps_to_completion = []
    
    # Loop through independent experiments
    for i in range(num_experiments):
        # Generate new seeds for environment and controller
        seeds = seed_sequences[i].generate_state(3)
        SeedMemory.set_seeds(seeds)
        
        config_controller = dict(config_manager("config_controllers")[controller_short_name])
        config_optimizer = dict(config_manager("config_optimizers")[optimizer_short_name])
        config_optimizer.update({"seed": int(seeds[1])})
        config_environment = dict(config_manager("config_environments")[environment_name])
        config_environment.update({"seed": int(seeds[0])})
        all_rewards = []

        ##### ----------------------------------------------- #####
        ##### ----------------- ENVIRONMENT ----------------- #####
        ##### --- Instantiate environment and call reset ---- #####
        if config_manager("config")["render_for_humans"]:
            render_mode = "human"
        elif config_manager("config")["save_plots_to_file"]:
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
        controller.configure(optimizer_name=optimizer_short_name, predictor_specification=config_controller["predictor_specification"])

        ##### ----------------------------------------------------- #####
        ##### ----------------- MAIN CONTROL LOOP ----------------- #####
        frames = []
        start_time = time.time()
        num_iterations = config_manager("config")["num_iterations"]
        for step in range(num_iterations):
            action = controller.step(obs, updated_attributes=env.environment_attributes)
            new_obs, reward, terminated, truncated, info = env.step(action)
            c_fun: CostFunctionWrapper = getattr(controller, "cost_function", None)
            if c_fun is not None:
                assert isinstance(c_fun, CostFunctionWrapper)
                # Compute reward from the cost function that the controller optimized
                reward = -float(c_fun.get_stage_cost(
                    tf.convert_to_tensor(new_obs[np.newaxis, np.newaxis, ...]),  # Add batch / MPC horizon dimensions
                    tf.convert_to_tensor(action[np.newaxis, np.newaxis, ...]),
                    None
                ))
                all_rewards.append(reward)
            if config_controller.get("controller_logging", False):
                controller.logs["realized_cost_logged"].append(np.array([-reward]).copy())
                env.set_logs(controller.logs)
            if config_manager("config")["render_for_humans"]:
                env.render()
            elif config_manager("config")["save_plots_to_file"]:
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
        all_mean_rewards.append(np.mean(all_rewards))
        all_steps_to_completion.append(step + 1)

        if run_for_ML_Pipeline:
            # Save data as csv
            if i < int(frac_train * num_experiments):
                csv = os.path.join(record_path, "Train")
            elif i < int((frac_train + frac_val) * num_experiments):
                csv = os.path.join(record_path, "Validate")
            else:
                csv = os.path.join(record_path, "Test")
            os.makedirs(csv, exist_ok=True)
            save_to_csv(config_manager("config"), controller, environment_name, csv)
        elif config_controller.get("controller_logging", False):
            if config_manager("config")["save_plots_to_file"]:
                # Generate and save plots in default location
                generate_experiment_plots(
                    config=config_manager("config"),
                    environment_config=config_manager("config_environments")[environment_name],
                    controller_output=controller_output,
                    timestamp=timestamp_str,
                    frames=frames if len(frames) > 0 else None,
                )
            # Save .npy files 
            for n, a in controller_output.items():
                with open(
                    OutputPath.get_output_path(timestamp_str, f"{str(n)}.npy"),
                    "wb",
                ) as f:
                    np.save(f, a)
            # Save configs
            for loader in config_manager.loaders.values():
                with open(
                    OutputPath.get_output_path(timestamp_str, loader.name), "w"
                ) as f:
                    dump(loader.config, f)
    
    # These output metrics are detected by GUILD AI and follow a "key: value" format
    print("Output metrics:")
    print(f"mean_reward: {np.mean(all_mean_rewards)}")
    print(f"mean_steps_to_completion: {np.mean(all_steps_to_completion)}")


def prepare_and_run():
    import ruamel.yaml
    
    config_manager = ConfigManager(".", "Control_Toolkit_ASF", "SI_Toolkit_ASF", "Environments")
    # Scan for any custom parameters that should overwrite the toolkit configs:
    submodule_configs = ConfigManager("Control_Toolkit_ASF", "SI_Toolkit_ASF", "Environments").loaders
    for base_name, loader in submodule_configs.items():
        if base_name in config_manager("config").get("custom_config_overwrites", {}):
            data: ruamel.yaml.comments.CommentedMap = loader.load()
            update_dict = config_manager("config")["custom_config_overwrites"][base_name]
            nested_assignment_to_ordereddict(data, update_dict)
            loader.overwrite_config(data)
    
    # Retrieve required parameters from config:
    CurrentRunMemory.current_controller_name = config_manager("config")["controller_name"]
    CurrentRunMemory.current_environment_name = config_manager("config")["environment_name"]
    
    run_data_generator(
        controller_name=CurrentRunMemory.current_controller_name,
        environment_name=CurrentRunMemory.current_environment_name,
        config_manager=config_manager,
        run_for_ML_Pipeline=False,
    )

# TODO: Add a top-level dict that references all config files
if __name__ == "__main__":
    if os.getenv("GUILD_RUN") == "1":
        # Run as guild script
        from guild import ipy as guild
        guild.run(prepare_and_run)
    else:
        prepare_and_run()