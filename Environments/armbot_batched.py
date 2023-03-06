# multiple DOF robot arm kinecmatics control env
# latest revision: Feb 27 2023 by dxiang@ini.ethz.ch
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.envs.classic_control.acrobot import AcrobotEnv

from Control_Toolkit.others.environment import EnvironmentBatched
from SI_Toolkit.computation_library import ComputationLibrary, NumpyLibrary, TensorType
import yaml
import os
class armbot_batched(EnvironmentBatched, AcrobotEnv):
    anglemax=np.pi
    rendercnt=0
    saveimgs=0
    num_states = 20 #reconfigurable number of joints here
    num_actions = num_states
    th1_0 = np.pi / 4
    th2_0 = np.pi / 5
    xtarget = tf.cos(th1_0) + tf.cos(th1_0 + th2_0) + (num_states - 2) * tf.cos(th1_0 + th2_0)
    ytarget = tf.sin(th1_0) + tf.sin(th1_0 + th2_0) + (num_states - 2) * tf.sin(th1_0 + th2_0)
    robs = 4
    xobs=xtarget+robs+0.5
    yobs=ytarget-2.5

    useobs=1
    config = yaml.load(
        open(os.path.join("Control_Toolkit_ASF", "config_controllers.yml"), "r"),
        Loader=yaml.FullLoader,
    )
    whichcontroller=config["mpc"]["optimizer"]
    def __init__(
            self,
            batch_size=1,
            computation_lib=NumpyLibrary,
            render_mode="human",
            **kwargs,
    ):
        super().__init__(render_mode=render_mode)

        self.config = {
            **kwargs,
            **{"render_mode": self.render_mode},
        }
        self.dt = kwargs["dt"]

        high = np.pi * np.ones(self.num_states, dtype=np.float32)
        umax = 0.2
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(-umax * np.ones(self.num_states), umax * np.ones(self.num_states),
                                       dtype=np.float32)
        dtype = np.float32,
        self._batch_size = batch_size
        self._actuator_noise = np.array(kwargs["actuator_noise"], dtype=np.float32)

        self.set_computation_library(computation_lib)
        self._set_up_rng(kwargs["seed"])

    def step_dynamics(
            self,
            state: TensorType,
            action: TensorType,
            dt: float,
    ) -> TensorType:
        tuple2 = self.lib.unstack(state, self.num_states, 1)
        for i in range(len(tuple2)):
            tuple2[i] += action[:, i] * dt
            tuple2[i]=self.lib.floormod(tuple2[i] + self.lib.pi, 2 * self.lib.pi) - self.lib.pi
            tuple2[i]=self.lib.clip(tuple2[i], -self.anglemax,self.anglemax)
        state = self.lib.stack(tuple2, 1)

        return state

    def step(
            self, action: TensorType
    ) -> Tuple[
        TensorType,
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        Union[np.ndarray, bool],
        dict,
    ]:
        reward = 0.0

        truncated = False
        self.state, action = self._expand_arrays(self.state, action)
        
        self.state = self.step_dynamics(self.state, action, self.dt)

        terminated = bool(self.is_done(self.lib, self.state, self.xtarget, self.ytarget))
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
            low, high = utils.maybe_parse_reset_bounds(options, -0.0, 0.0)
            ns = self.lib.uniform(
                self.rng, (self._batch_size, self.num_states), low, high, self.lib.float32
            )
        else:
            if self.lib.ndim(state) < 2:
                state = self.lib.unsqueeze(
                    self.lib.to_tensor(state, self.lib.float32), 0
                )
            if self.lib.shape(state)[0] == 1:
                ns = self.lib.tile(state, (self._batch_size, 1))
            else:
                ns = state


        self.state = ns

        return self._get_reset_return_val()

    @staticmethod
    def is_done(lib: "type[ComputationLibrary]", state: TensorType, xtarget: float, ytarget: float):
        tuple2 = lib.unstack(state, armbot_batched.num_states, -1)
        theta = tuple2[0]
        xee = tf.cos(theta)
        yee = tf.cos(theta)
        for i in range(1, armbot_batched.num_states):
            theta += tuple2[i]
            xee += tf.cos(theta)
            yee += tf.sin(theta)
        return (
                np.abs(xee - xtarget) + np.abs(yee - ytarget)
        ) < 0.2

    def _convert_to_state(self, state):
        if self.lib.shape(state)[-1] == self.num_states:
            return state
        return self.lib.concat(
            [
                self.lib.cos(self.lib.unsqueeze(state[..., 0], 1)),
                self.lib.sin(self.lib.unsqueeze(state[..., 0], 1)),
                self.lib.cos(self.lib.unsqueeze(state[..., 1], 1)),
                self.lib.sin(self.lib.unsqueeze(state[..., 1], 1)),
                self.lib.unsqueeze(state[..., 2], 1),
                self.lib.unsqueeze(state[..., 3], 1),
            ],
            1,
        )

    def bound(self, x, m, M=None):
        """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
        have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

        Args:
            x: scalar
            m: The lower bound
            M: The upper bound

        Returns:
            x: scalar, bound between min (m) and Max (M)
        """
        if M is None:
            M = m[1]
            m = m[0]
        # bound x between min (m) and Max (M)
        return self.lib.min(self.lib.max(x, m), M)

    def render(self):# the following rendering is adapted from gym-acrobat render implementation
        NL = self.num_states
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.SCREEN_DIM, self.SCREEN_DIM)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        surf.fill((255, 255, 255))
        s = self.state
        bound = NL + 0.2  # 2.2 for default
        scale = self.SCREEN_DIM / (bound * 2)
        offset = self.SCREEN_DIM / 2

        if s is None:
            return None

        p1 = [
            -self.LINK_LENGTH_1 * tf.cos(s[0]) * scale,
            self.LINK_LENGTH_1 * tf.sin(s[0]) * scale,
        ]

        p2 = [
            p1[0] - self.LINK_LENGTH_2 * tf.cos(s[0] + s[1]) * scale,
            p1[1] + self.LINK_LENGTH_2 * tf.sin(s[0] + s[1]) * scale,
        ]

        xys0 = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas0 = np.array([s[0] - np.pi / 2, s[0] + s[1] - np.pi / 2])
        thetas = np.array(self.state[:NL])
        thetas = np.cumsum(thetas)

        xys = [[0, 0]]
        for i in range(NL):
            p0 = xys[-1]
            p1 = [p0[0] - np.cos(thetas[i]) * scale, p0[1] + np.sin(thetas[i]) * scale]
            xys.append(p1)
        xys = np.array(xys)[:, ::-1]
        thetas -= np.pi / 2
        link_lengths = np.ones(NL) * scale

        pygame.draw.line(
            surf,
            start_pos=(-2.2 * scale + offset, 1 * scale + offset),
            end_pos=(2.2 * scale + offset, 1 * scale + offset),
            color=(0, 0, 0),
        )

        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            x = x + offset
            y = y + offset
            l, r, t, b = 0, llen, 0.2 * scale, -0.2 * scale
            coords = [(l, b), (l, t), (r, t), (r, b)]
            transformed_coords = []
            for coord in coords:
                coord = pygame.math.Vector2(coord).rotate_rad(th)
                coord = (coord[0] + x, coord[1] + y)
                transformed_coords.append(coord)
            gfxdraw.aapolygon(surf, transformed_coords, (0, 204, 204))
            gfxdraw.filled_polygon(surf, transformed_coords, (0, 204, 204))

            gfxdraw.aacircle(surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))
            gfxdraw.filled_circle(surf, int(x), int(y), int(0.15 * scale), (204, 204, 0))
        #draw target
        gfxdraw.filled_circle(surf, int(self.ytarget * scale + offset), int(-self.xtarget * scale + offset),
                              int(NL*0.05 * scale), (204, 0, 0))
        #draw obstacle(s)
        if self.useobs>0:
            gfxdraw.filled_circle(surf, int(self.yobs * scale + offset), int(-self.xobs * scale + offset),
                                  int(self.robs * scale), (0, 200, 200))
        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))
        if self.saveimgs>0:
            if self.rendercnt ==0:
                filename='./imgs/'+self.whichcontroller+f'/{self.rendercnt:04d}.png'
                pygame.image.save(self.screen, filename)
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()


        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        self.rendercnt+=1