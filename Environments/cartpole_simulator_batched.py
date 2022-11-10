from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from CartPoleSimulation.CartPole.cartpole_model_tf import (
    _cartpole_ode, cartpole_integration_tf)
from CartPoleSimulation.CartPole.state_utilities import (ANGLE_COS_IDX,
                                                         POSITION_IDX)
from CartPoleSimulation.GymlikeCartPole.CartPoleEnv_LTC import CartPoleEnv_LTC
from Control_Toolkit.others.environment import EnvironmentBatched
from SI_Toolkit.computation_library import ComputationLibrary, NumpyLibrary, TensorType
from gym.spaces import Box


class cartpole_simulator_batched(EnvironmentBatched, CartPoleEnv_LTC):
    num_actions = 1
    num_states = 6

    def __init__(
        self,
        batch_size=1,
        computation_lib=NumpyLibrary,
        render_mode="human",
        **kwargs,
    ):
        self._batch_size = batch_size
        self._actuator_noise = np.array(kwargs["actuator_noise"], dtype=np.float32)
        self.render_mode = render_mode

        self.shuffle_target_every = kwargs["shuffle_target_every"]
        self.config = {
            **kwargs,
            **{"render_mode": self.render_mode},
        }

        self.set_computation_library(computation_lib)
        self._set_up_rng(kwargs["seed"])
        self.dt = self.lib.to_tensor(kwargs["dt"], self.lib.float32)

        # self.CartPoleInstance = CartPole()
        # self.CartPoleInstance.dt_simulation = self.dt
        self.mode = kwargs["mode"]

        if self.mode != "stabilization":
            raise NotImplementedError("Only stabilization mode defined for now.")

        self.min_action = -1.0
        self.max_action = 1.0

        cart_length = kwargs["cart_length"]
        usable_track_length = kwargs["usable_track_length"]
        track_half_length = np.array(usable_track_length - cart_length / 2.0)
        self.u_max = kwargs["u_max"]

        self.target_position = tf.Variable(0.0, dtype=tf.float32)
        self.environment_attributes = {
            "target_position": self.target_position,
        }
        self.initial_state = kwargs['initial_state']

        self.x_threshold = (
            0.9 * track_half_length
        )  # Takes care that the cart is not going beyond the boundary

        observation_space_boundary = np.array(
            [
                np.float32(np.pi),
                np.finfo(np.float32).max,
                1.0,
                1.0,
                np.float32(track_half_length),
                np.finfo(np.float32).max,
            ]
        )

        self.observation_space = Box(
            -observation_space_boundary, observation_space_boundary
        )
        self.action_space = Box(
            low=np.float32(self.min_action),
            high=np.float32(self.max_action),
            shape=(1,),
        )

        self.viewer = None
        self.screen = None
        self.isopen = False
        self.state = None
        self.action = None
        self.reward = None
        self.done = False
        self.count = 0

        self.steps_beyond_done = None

    def reset(
        self,
        seed: "Optional[int]" = None,
        options: "Optional[dict]" = None,
    ) -> "Tuple[np.ndarray, dict]":
        if seed is not None:
            self._set_up_rng(seed)

        state = options.get("state", None) if isinstance(options, dict) else None
        self.count = 0

        if state is None and self.initial_state is None:
            low = np.array([-self.lib.pi / 4, 1.0e-1, 1.0e-1, 1.0e-1])
            high = np.array([self.lib.pi / 4, 1.0e-1, 1.0e-1, 1.0e-1])
            angle, angleD, position, positionD = self.lib.unstack(
                self.lib.uniform(
                    self.rng, [self._batch_size, 4], low, high, self.lib.float32
                ),
                4,
                1,
            )
            angle = self.lib.to_numpy(angle)
            mask = angle >= 0
            angle[mask] -= np.pi
            angle[~mask] += np.pi
            self.state = self.lib.stack(
                [
                    angle,
                    angleD,
                    self.lib.cos(angle),
                    self.lib.sin(angle),
                    position,
                    positionD,
                ],
                1,
            )
            self.steps_beyond_done = None
        elif state is None and self.initial_state is not None:

            state = [0]*6
            state[0] = self.initial_state[0]
            state[1] = self.initial_state[1]
            state[2] = self.lib.cos(self.initial_state[0])
            state[3] = self.lib.sin(self.initial_state[0])
            state[4] = self.initial_state[2]
            state[5] = self.initial_state[3]

            state = self.lib.unsqueeze(
                self.lib.to_tensor(state, self.lib.float32), 0
            )

            self.state = state
            pass
        else:
            if self.lib.ndim(state) < 2:
                state = self.lib.unsqueeze(
                    self.lib.to_tensor(state, self.lib.float32), 0
                )
            if self.lib.shape(state)[0] == 1:
                self.state = self.lib.tile(state, (self._batch_size, 1))
            else:
                self.state = state

        if self._batch_size == 1:
            self.state = self.lib.to_numpy(self.lib.squeeze(self.state))

        return self.state, {}

    def step_dynamics(
        self,
        state: TensorType,
        action: TensorType,
        dt: float,
    ) -> TensorType:
        # Convert dimensionless motor power to a physical force acting on the Cart
        u = self.u_max * action[:, 0]

        angle, angleD, angle_cos, angle_sin, position, positionD = self.lib.unstack(
            state, 6, 1
        )

        # Compute next state
        angleDD, positionDD = _cartpole_ode(angle_cos, angle_sin, angleD, positionD, u)

        angle, angleD, position, positionD = cartpole_integration_tf(
            angle, angleD, angleDD, position, positionD, positionDD, dt
        )
        angle_cos = self.lib.cos(angle)
        angle_sin = self.lib.sin(angle)

        angle = self.lib.atan2(angle_sin, angle_cos)

        next_state = self.lib.stack(
            [angle, angleD, angle_cos, angle_sin, position, positionD], 1
        )

        return next_state

    def step(
        self, action: TensorType
    ) -> Tuple[
        TensorType,
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        Union[np.ndarray, bool],
        dict,
    ]:
        self.state, action = self._expand_arrays(self.state, action)

        # Perturb action if not in planning mode
        assert self._batch_size == 1
        action = self._apply_actuator_noise(action)

        self.state = self.lib.to_numpy(self.step_dynamics(self.state, action, self.dt))

        # Update the total time of the simulation
        # self.CartPoleInstance.step_time()
        if self.count % self.shuffle_target_every == 0:
            new_target = self.lib.uniform(
                self.rng, [], -self.x_threshold, self.x_threshold, self.lib.float32
            )
            self.target_position.assign(new_target)
        self.count += 1

        reward = 0.0
        terminated = self.is_done(self.lib, self.state)
        truncated = False

        self.state = self.lib.to_numpy(self.lib.squeeze(self.state))
        reward = float(reward)

        return (
            self.state,
            reward,
            terminated,
            truncated,
            {"target": self.lib.to_numpy(self.target_position)},
        )

    @staticmethod
    def is_done(lib: "type[ComputationLibrary]", state: TensorType):
        return False