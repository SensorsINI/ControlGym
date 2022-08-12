from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from gym.envs.box2d.bipedal_walker import *

from Control_Toolkit.others.environment import EnvironmentBatched, NumpyLibrary
from Utilities.utils import CurrentRunMemory


class bipedal_walker_batched(EnvironmentBatched, BipedalWalker):
    """Accepts batches of data to environment"""
    num_actions = 4
    num_states = 24

    def __init__(self) -> None:
        super().__init__()

    def __init__(
        self, batch_size=1, computation_lib=NumpyLibrary, render_mode="human", **kwargs
    ):
        super().__init__(render_mode=render_mode, hardcore=False)

        self.config = {
            **kwargs,
            **{"render_mode": self.render_mode},
        }
        CurrentRunMemory.controller_specific_params = self.config

        self._batch_size = batch_size
        self._actuator_noise = np.array(kwargs["actuator_noise"], dtype=np.float32)

        self.set_computation_library(computation_lib)
        self._set_up_rng(kwargs["seed"])
        self.cost_functions = self.cost_functions_wrapper(self)
        
    def step_tf(self, state: tf.Tensor, action: tf.Tensor):
        state, action = self._expand_arrays(state, action)
        
        if self._batch_size == 1:
            action += self._generate_actuator_noise()
        
        position, velocity = self.lib.unstack(state, 2, 1)

        # ...
        
        state = self.lib.stack([position, velocity], 1)
        state = self.lib.squeeze(state)

        return state
    
    def step(
        self, action: Union[np.ndarray, tf.Tensor, torch.Tensor]
    ) -> Tuple[
        Union[np.ndarray, tf.Tensor, torch.Tensor],
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        dict,
    ]:
        self.state, action = self._expand_arrays(self.state, action)

        # Perturb action if not in planning mode
        if self._batch_size == 1:
            action += self._generate_actuator_noise()

        position, velocity = self.lib.unstack(self.state, 2, 1)

        assert self.hull is not None

        # self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        control_speed = False  # Should be easier as well
        if control_speed:
            self.joints[0].motorSpeed = float(SPEED_HIP) * self.lib.clip(action[:, 0], -1, 1)
            self.joints[1].motorSpeed = float(SPEED_KNEE) * self.lib.clip(action[:, 1], -1, 1)
            self.joints[2].motorSpeed = float(SPEED_HIP) * self.lib.clip(action[:, 2], -1, 1)
            self.joints[3].motorSpeed = float(SPEED_KNEE) * self.lib.clip(action[:, 3], -1, 1)
        else:
            self.joints[0].motorSpeed = float(SPEED_HIP) * self.lib.sign(action[:, 0])
            self.joints[0].maxMotorTorque = float(MOTORS_TORQUE) * self.lib.clip(self.lib.abs(action[:, 0]), 0, 1)
            self.joints[1].motorSpeed = float(SPEED_KNEE) * self.lib.sign(action[:, 1])
            self.joints[1].maxMotorTorque = float(MOTORS_TORQUE) * self.lib.clip(self.lib.abs(action[:, 1]), 0, 1)
            self.joints[2].motorSpeed = float(SPEED_HIP) * self.lib.sign(action[:, 2])
            self.joints[2].maxMotorTorque = float(MOTORS_TORQUE) * self.lib.clip(self.lib.abs(action[:, 2]), 0, 1)
            self.joints[3].motorSpeed = float(SPEED_KNEE) * self.lib.sign(action[:, 3])
            self.joints[3].maxMotorTorque = float(MOTORS_TORQUE) * self.lib.clip(self.lib.abs(action[:, 3]), 0, 1)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + self.lib.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                pos[1] - self.lib.cos(1.5 * i / 10.0) * LIDAR_RANGE,
            )
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = self.lib.stack([
            self.hull.angle,  # Normal angles up to 0.5 here, but sure more is possible.
            2.0 * self.hull.angularVelocity / FPS,
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
            self.joints[0].angle,
            # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            self.lib.cast(self.legs[1].ground_contact, self.lib.float32),
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            self.lib.cast(self.legs[3].ground_contact, self.lib.float32),
        ], 1)
        state = self.lib.concat([
            state,
            self.lib.stack([l.fraction for l in self.lidar], 1)
        ], 1)
        assert self.lib.shape(state)[1] == 24

        if self._batch_size == 1:
            self.scroll = pos.x - VIEWPORT_W / SCALE / 5

        shaping = (
            130 * pos[0] / SCALE
        )  # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 5.0 * abs(
            state[0]
        )  # keep head straight, other than that and falling, any behavior is unpunished

        reward = self.lib.zeros((self._batch_size))
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * self.lib.clip(self.lib.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        terminated = False
        if self.game_over or pos[0] < 0:
            reward = -100
            terminated = True
        if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            terminated = True



        done = self.is_done(self.state)
        reward = self.get_reward(self.state, action)

        self.state = self.lib.squeeze(self.state)

        if self._batch_size == 1:
            self.renderer.render_step()
            return (
                self.lib.to_numpy(self.lib.squeeze(self.state)),
                float(reward),
                bool(done),
                {},
            )

        return self.state, reward, done, {}

    def reset(
        self,
        state: np.ndarray = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        if seed is not None:
            self._set_up_rng(seed)

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

    def render(self, mode="human"):
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
