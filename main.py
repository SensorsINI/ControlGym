import os
from tqdm import trange
from Utilities.utils import ConfigManager,  CurrentRunMemory, nested_assignment_to_ordereddict
from EnvManager import EnvManager

def run_data_generator(
    controller_name: str,
    environment_name: str,
    config_manager: ConfigManager,
    run_for_ML_Pipeline=False,
    record_path=None,
):

    Env = EnvManager(
        CRM=CurrentRunMemory,
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
    import ruamel.yaml
    
    # Create a config manager which looks for '.yml' files within the list of folders specified.
    # Rationale: We want GUILD AI to be able to update values in configs that we include in this list.
    # We might intentionally want to exclude the path to a folder which does contain configs but should not be overwritten by GUILD. 
    config_manager = ConfigManager(".", "Control_Toolkit_ASF", "SI_Toolkit_ASF", "Environments")
    
    # Scan for any custom parameters that should overwrite the toolkits' config files:
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

if __name__ == "__main__":
    if os.getenv("GUILD_RUN") == "1":
        # Run as guild script
        from guild import ipy as guild
        guild.run(prepare_and_run)
    else:
        prepare_and_run()