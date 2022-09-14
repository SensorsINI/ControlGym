# ControlGym - Repo to Test New Control Strategies

> This branch (`reproduction_of_results_sep22`) belongs to the paper "A Small-Batch Parallel Gradient Descent Optimizer with Explorative Resampling for Nonlinear Model Predictive Control"

Main features:
* Conforms to OpenAI Gym API
* Implements batched versions of OpenAI Gym environments
* Can implement new controllers and test them

### Reproduction of Simulation Results
* `config.yml` contains the right controller parameters
* To run a standard set of simulations:
    * At the top of `config.yml`: Set the list of environments and controllers to run
    * `python -m Utilities.controller_comparison`
* To run a variation of a hyperparameter:
    * Set the variables (`CONTROLLER_TO_ANALYZE`, `PARAMETERS_TO_SWEEP`, `SWEEP_VALUES`) at the top of `Utilities/hyperparameter_sweep.py`
    * Select one environment in `config.yml`
    * `python -m Utilities.hyperparameter_sweep`
* To evaluate the results and generate plots:
    * Set `EXPERIMENT_FOLDER` (within `/Output`), `ENVIRONMENT_NAME` and `sweep_values` (for plot labelling only)
    * `python -m Utilities.generate_global_plots`

### References

* [OpenAI Gym on GitHub](https://github.com/openai/gym)
* [Brockman et al. 2016, OpenAI Gym](https://arxiv.org/abs/1606.01540)


### Required Installs

* `pip install -r requirements.txt`
* ffmpeg
* latex