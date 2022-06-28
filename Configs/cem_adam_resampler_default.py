import tensorflow as tf

ENV_NAME = "CustomEnvironments/Pendulum"
NUM_ITERATIONS = 300

CONTROLLER_NAME = "ControllerAdamResampler"

CONTROLLER_CONFIG = {
    "SEED": 12345,
    "mpc_horizon": 3,
    "dt": 0.02,
    "cem_outer_it": 5,
    "cem_rollouts": 100,
    "cem_predictor_type": "EulerTF",
    "cem_stdev_min": 0.1,
    "cem_R": 1,
    "cem_ccrc_weight": 1,
    "cem_best_k": 5,
    "cem_LR": 0.1,
    "max_grad": 1000,
    "grad_alpha": 0.05,
    "grad_beta_1": 0.9,
    "grad_beta_2": 0.999,
    "grad_epsilon": 1e-7,
    "cem_initial_action_variance": tf.constant(0.5, dtype=tf.float32),
    "resamp_every": 1,
    "do_warmup": True,
}