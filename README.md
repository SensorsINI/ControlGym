# ControlGym - Repo to Test New Control Strategies

Main features:
* Conforms to OpenAI Gym API
* Implements batched versions of OpenAI Gym environments
* Can implement new controllers and MPC optimizers and test them
* Experiment management using GUILD AI

### New: Experiment Management with GUILD AI
We use the [GUILD AI](https://guild.ai) library to manage experiments and controllers. The difficulty lies in adapting the framework to suit the needs of this repository and the git submodules.

The following components are important to make GUILD AI work for your workflow:

* `config.yml` Main configuration file: The top section can be used to overwrite the controller, optimizer, cost function and predictor configurations. Below that, you can specify how long runs are, what to plot and how to save outputs.
* `guild.yml` GUILD AI configuration file: It instructs the framework to look in the `config.yml` for hyperparameters
* `main.py` script is called through GUILD AI

#### Important commands
* `guild ops`: List available scripts to run
* `guild run controller_mpc:run_control`: Run an MPC experiment using specification from `config.yml`.
* `guild run with extra args`
    * `guild run controller_mpc:run_control num_iterations=250`: Overwrite config value of num_iterations and run.
    * `guild run controller_mpc:run_control custom_config_overwrites.config_controllers.mpc.optimizer='[random-action-tf, cem-naive-grad-tf]' num_iterations=250`: Overwrite the controller config with a sweep over two different optimizers. The value `custom_config_overwrites.config_controllers.mpc.optimizer` needs to be a nested entry within the top section of `config.yml`.
* `guild runs`: List past runs and hyperparameters
* `guild cat <<run_id>>`: Print console outputs of a run
* `guild view`: Open GUILD AI dashboard
+ `guild run controller_mpc:run_control custom_config_overwrites.config_controllers.mpc.optimizer='rpgd-tf' custom_config_overwrites.config_optimizers.rpgd-tf.learning_rate='[0.0001:0.5]' --optimizer gp --max-trials 10 --maximize mean_reward`

### Reproduction of Simulation Results
* `config.yml` contains the right controller parameters
* To run a standard set of simulations:
    * At the top of `config.yml`: Set the list of environments and controllers to run
    * `python -m Utilities.run_mpc_sweep`
* To run a variation of a hyperparameter:
    * Set the variables (`CONTROLLER_TO_ANALYZE`, `PARAMETERS_TO_SWEEP`, `SWEEP_VALUES`) at the top of `Utilities/hyperparameter_sweep.py`
    * Select one environment in `config.yml`
    * `python -m Utilities.hyperparameter_sweep`
* To evaluate the results and generate plots:
    * Set `EXPERIMENT_FOLDER` (within `/Output`), `ENVIRONMENT_NAME` and `sweep_values` (for plot labelling only)
    * Run `python -m Utilities.generate_global_plots`

### Reproduction of Benchmarking Results
* Forces optimizer: select `optimizer: nlp-forces` in `Control_Toolikit_ASF/config_controllers.yml` and profile `step` function in `Control_Toolkit/optimizers/optimizer_nlp_forces.py`. Licensed ForcesPRO client is required in the folder forces/.
* RPGD optimizer: select `optimizer: rpgd-tf` in `Control_Toolikit_ASF/config_controllers.yml` and profile `step` function in `Control_Toolkit/optimizers/optimizer_rpgd_tf.py`. Note that in the first step of the RPGD optimizer, the Tensorflow computational graph is created, to get a coherent evaulation of the asyntotic computational time, it is needed to not get into account the first step.


### Plots you can generate
* If you used GUILD AI, you can plot the effect of a hyperparameter on the distribution of episode rewards (= negative costs)
    * To do so, open the `Utilities.generate_reward_distribution_plots` module
        * Set the `GUILD_AI_HASHES` of the runs you want to plot
        * Each hash refers to a set of episodes using the same hyperparameter specification
        * Set `fig_title`, `hp_name`, `hp_values` to set the plot annotations.
            * Example: `hp_name = "learning rate"` and `hp_values = [0.0, 0.1, 0.5]` when the `GUILD_AI_HASHES` contains three hashes to runs with learning rate taking those values
            * Specifying this annotation is currently done manually here, but one could implement a way to go into the guild recordings and retrieve that automatically
        * Make sure that the `PATHS_TO_EXPERIMENTS` links to the `.guild` folder within the right environment. This could be a local `.env` folder or a path to the conda environment used.
    * Then, run `python -m Utilities.generate_reward_distribution_plots`
    * Result looks like this:
        * <img src="Visualizations/sample_figures/sample_reward_distribution_plot.png" alt="sample reward distribution scatter plot" width="400"/>
* If you ran the `Utilities.run_mpc_sweep` module or the `main` module
    * You can have plots saved if you set `save_plots_to_file: true` in `config.yml`
    * Horizon cost plot:
        * <img src="Visualizations/sample_figures/sample_horizon_cost_plot.png" alt="sample horizon cost plot" width="400"/>
    * Summary plot:
        * <img src="Visualizations/sample_figures/sample_summary_plot.png" alt="sample summary plot" width="400"/>
    * A plot of the ages of rollout-inducing input plans before they are replaced by new input samples


# References
* [OpenAI Gym on GitHub](https://github.com/openai/gym)
* [Brockman et al. 2016, OpenAI Gym](https://arxiv.org/abs/1606.01540)


# Installation
* `git clone` this repository
* In the cloned `ControlGym` directory, run
    * `git submodule init`
    * `git submodule update` to load the submodules.
* `pip install -r requirements.txt`
* If you want GUI / rendering:
  * `pip install PyQt5` or `PyQt6`
  * `ffmpeg` and `latex`. Look up how to do that for your OS.
  
 
