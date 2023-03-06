# ControlGym - Repo to Test New Control Strategies

This branch is intended for the reproduction of the benchmarking comparison of ForcesPRO and RPGD.

### Reproduction of Benchmarking Results
* Forces optimizer: select `optimizer: nlp-forces` in `Control_Toolikit_ASF/config_controllers.yml` and profile `step` function in `Control_Toolkit/optimizers/optimizer_nlp_forces.py`. Licensed ForcesPRO client is required in the folder forces/.
* RPGD optimizer: select `optimizer: rpgd-tf` in `Control_Toolikit_ASF/config_controllers.yml` and profile `step` function in `Control_Toolkit/optimizers/optimizer_rpgd_tf.py`. Note that in the first step of the RPGD optimizer, the Tensorflow computational graph is created, to get a coherent evaulation of the asyntotic computational time, it is needed to not get into account the first step.

# References
* [OpenAI Gym on GitHub](https://github.com/openai/gym)
* [Brockman et al. 2016, OpenAI Gym](https://arxiv.org/abs/1606.01540)
