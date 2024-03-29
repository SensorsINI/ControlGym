from typing import Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import tensorflow as tf
import torch
from Control_Toolkit.others.environment import EnvironmentBatched
from SI_Toolkit.computation_library import NumpyLibrary, TensorType
from gymnasium.envs.box2d.bipedal_walker import *


class bipedal_walker_batched(EnvironmentBatched, BipedalWalker):
    """Accepts batches of data to environment"""

    num_actions = 4
    num_states = 24

    def __init__(
        self,
        batch_size=1,
        computation_lib=NumpyLibrary,
        render_mode="human",
        **kwargs,
    ):
        super().__init__(render_mode=render_mode, hardcore=False)

        self.config = {
            **kwargs,
            **{"render_mode": self.render_mode},
        }
        self.dt = kwargs["dt"]

        self._batch_size = batch_size
        self._actuator_noise = np.array(kwargs["actuator_noise"], dtype=np.float32)

        self.set_computation_library(computation_lib)
        self._set_up_rng(kwargs["seed"])
        
        self.envs = gym.vector.make("BipedalWalker-v3", num_envs=self._batch_size)

    def step_dynamics(
        self,
        state: Union[np.ndarray, tf.Tensor, torch.Tensor],
        action: Union[np.ndarray, tf.Tensor, torch.Tensor],
        dt: float,
    ) -> Union[np.ndarray, tf.Tensor, torch.Tensor]:
        self.envs.set_attr("state", state)
        observations, rewards, dones, infos = self.envs.step(action)
        return observations

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
        
        self.state = self.step_dynamics(self.state, action, self.dt)

        terminated = bool(self.is_done(self.state))
        truncated = False
        reward = self.get_reward(self.state, action)

        self.state = self.lib.squeeze(self.state)

        return (
            self.lib.to_numpy(self.lib.squeeze(self.state)),
            float(reward),
            terminated,
            truncated,
            {},
        )

    def reset(
        self,
        seed: "Optional[int]" = None,
        options: "Optional[dict]" = None,
    ) -> "Tuple[np.ndarray, dict]":
        if seed is not None:
            self._set_up_rng(seed)
        state = options.get("state", None) if isinstance(options, dict) else None

        if state is None:
            if self._batch_size == 1:
                self.state = self.lib.to_tensor(
                    [self.lib.uniform(self.rng, (), -0.6, -0.4, self.lib.float32), 0],
                    self.lib.float32,
                )
            else:
                self.state = self.lib.stack(
                    [
                        self.lib.uniform(
                            self.rng, (self._batch_size,), -0.6, -0.4, self.lib.float32
                        ),
                        self.lib.zeros((self._batch_size,)),
                    ],
                    1,
                )
        else:
            if self.lib.ndim(state) < 2:
                state = self.lib.unsqueeze(
                    self.lib.to_tensor(state, self.lib.float32), 0
                )
            if self.lib.shape(state)[0] == 1:
                self.state = self.lib.tile(state, (self._batch_size, 1))
            else:
                self.state = state

        return self._get_reset_return_val()

    def render(self):
        if self._batch_size == 1:
            return super().render()
        else:
            raise NotImplementedError("Rendering not implemented for batched mode")

    def is_done(self, state):
        return False

    def get_reward(self, state, action):
        shaping = (
            130 * pos[:, 0] / SCALE
        )  # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 5.0 * self.lib.abs(
            state[0]
        )  # keep head straight, other than that and falling, any behavior is unpunished

        reward = self.lib.zeros((self._batch_size,))
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * self.lib.clip(self.lib.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        reward[self.game_over or pos[:, 0] < 0] = -100

        return reward
