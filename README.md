### Reproduction of Simulation Results of FORCESPRO vs RPGD
* `config.yml` contains the general controller parameters
    * In particular contains the choice for the environment 
* To select the optimizer:
    * In Control_Toolkit_ASF/config_controllers.yml:
        * uncomment `optimizer: nlp-forces` or   `optimizer: nlp-forces`
* Optimizers parameters like MPC Horizon are in Control_Toolkit_ASF/config_optimizers.yml
* To test nlp-forces:
    * To test Pendulum and MountainCar no further modifications are required
    * To test different configurations of obstacles for ObstacleAvoidance and DubinsCar:
        * Uncomment the wanted obstacles configuration in Control_Toolkit_ASF/config_environments.yml
        * Use the cost function with corresponding hardcoded obstacles:
            * Cost functions for FORCESPRO are in Control_Toolkit/others/cost_forces_interface.py
            * select the name of the wanted cost function in config_optimizers.yml: nlp-forces: -> obstacle_avoidance/dubins_car -> cost: ...
