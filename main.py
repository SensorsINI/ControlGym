import os
from tqdm import trange
from Utilities.utils import ConfigManager
from EnvManager import EnvManager, prepare_run

def run_data_generator(
    controller_name: str,
    environment_name: str,
    config_manager: ConfigManager,
    run_for_ML_Pipeline=False,
    record_path=None,
):

    Env = EnvManager(
        controller_name=controller_name,
        environment_name=environment_name,
        config_manager=config_manager,
        run_for_ML_Pipeline=run_for_ML_Pipeline,
        record_path=record_path,
    )

    # Loop through independent experiments
    for _ in trange(Env.num_experiments):
        Env.reset()
        for _ in range(Env.num_iterations):
            action = Env.controller.step(Env.obs, updated_attributes=Env.env.environment_attributes)
            obs, reward, terminated, truncated, info = Env.step(action)

            Env.render()

            # If the episode is up, start a new experiment
            if truncated:
                break
            elif terminated:
                break

        Env.close()

    Env.summary()

def prepare_and_run():

    config_manager, CurrentRunMemory = prepare_run()
    
    run_data_generator(
        controller_name=CurrentRunMemory.current_controller_name,
        environment_name=CurrentRunMemory.current_environment_name,
        config_manager=config_manager,
        run_for_ML_Pipeline=False,
    )

if __name__ == "__main__":
    if os.getenv("GUILD_RUN") == "1":
        # Run as guild script
        from guild import ipy as guild
        guild.run(prepare_and_run)
    else:
        prepare_and_run()