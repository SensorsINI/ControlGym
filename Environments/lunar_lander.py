from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from gym.envs.box2d.lunar_lander import LunarLander

from Control_Toolkit.others.environment import EnvironmentBatched
from SI_Toolkit.computation_library import NumpyLibrary, TensorType


class lunar_lander_batched(EnvironmentBatched, LunarLander):
    """Accepts batches of data to environment
    
    Uses the continuous version of LunarLander as base class
    """

    num_actions = 2  # throttle of the engines
    # The state is an 8-dimensional vector:
    # - the coordinates of the lander in x & y
    # - its linear velocities in x & y
    # - its angle
    # - its angular velocity
    # - two booleans that represent whether each leg is in contact with the ground or not
    num_states = 8
    
    def __init__(
        self,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
        batch_size=1,
        computation_lib=NumpyLibrary,
        render_mode="human",
        **kwargs,
    ):
        super().__init__(render_mode=render_mode, continuous=True, gravity=gravity, enable_wind=enable_wind, wind_power=wind_power, turbulence_power=turbulence_power)
        
        self.config = {
            **kwargs,
            **{"render_mode": self.render_mode, "gravity": self.gravity, "enable_wind": self.enable_wind, "wind_power": self.wind_power, "turbulence_power": self.turbulence_power},
        }
        self.dt = kwargs["dt"]
        
        self._batch_size = batch_size
        self._actuator_noise = np.array(kwargs["actuator_noise"], dtype=np.float32)

        self.set_computation_library(computation_lib)
        self._set_up_rng(kwargs["seed"])
    
    def step_dynamics(self, state: TensorType, action: TensorType, dt: float) -> TensorType:
        pos_x, pos_y, vel_x, vel_y, angle, vel_angle, contact_1, contact_2 = self.lib.unstack(state, 8, 1)
        if self.enable_wind and not (contact_1 or contact_2):
            wind_mag = (
                self.lib.tanh(
                    self.lib.sin(0.02 * self.wind_idx)
                    + (self.lib.sin(self.lib.pi * 0.01 * self.wind_idx))
                )
                * self.wind_power
            )
            self.wind_idx += 1  # TODO: This might not work in compile mode
            self.lander.ApplyForceToCenter(
                (wind_mag, 0.0),
                True,
            )