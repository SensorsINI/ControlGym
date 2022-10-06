"""
This file has been modified from source https://github.com/gargivaidya/dubin_model_gymenv/blob/main/dubin_gymenv.py

This was in turn adapted from https://github.com/AtsushiSakai/PythonRobotics under...

The MIT License (MIT)

Copyright (c) 2016 - 2022 Atsushi Sakai and other contributors:
https://github.com/AtsushiSakai/PythonRobotics/contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
from typing import Optional, Tuple, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from Control_Toolkit.others.environment import (EnvironmentBatched,
                                                NumpyLibrary, TensorType)
from gym import spaces
from matplotlib.patches import Circle
from SI_Toolkit.Functions.TF.Compile import Compile

# Training constants
MAX_STEER = np.pi / 3
MAX_SPEED = 10.0
MIN_SPEED = 0.0
THRESHOLD_DISTANCE_2_GOAL = 0.01
MAX_X = 10.0
MAX_Y = 10.0
max_ep_length = 800

# Vehicle parameters
LENGTH = 0.45  # [m]
WIDTH = 0.2  # [m]
BACKTOWHEEL = 0.1  # [m]
WHEEL_LEN = 0.03  # [m]
WHEEL_WIDTH = 0.02  # [m]
TREAD = 0.07  # [m]
WB = 0.25  # [m]

show_animation = True


class dubins_car_batched(EnvironmentBatched, gym.Env):
    num_states = 4  # [x, y, yaw, steering_rate]
    num_actions = 2
    metadata = {"render_modes": ["console", "single_rgb_array", "rgb_array", "human"]}

    def __init__(
        self,
        target_point,
        shuffle_target_every: int,
        obstacle_positions: "list[list[float]]",
        initial_state: "list[float]",
        batch_size=1,
        computation_lib=NumpyLibrary,
        render_mode: str = None,
        **kwargs
    ):
        super().__init__()

        self.set_computation_library(computation_lib)
        self._set_up_rng(kwargs["seed"])

        self._batch_size = batch_size
        self._actuator_noise = np.array(kwargs["actuator_noise"], dtype=np.float32)
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            np.array([MIN_SPEED, -MAX_STEER]),
            np.array([MAX_SPEED, MAX_STEER]),
            dtype=np.float32,
        )  # Action space for [throttle, steer]
        low = np.array([-1.0, -1.0, -4.0, -0.5])  # low range of observation space
        high = np.array([1.0, 1.0, 4.0, 0.5])  # high range of observation space
        self.observation_space = spaces.Box(
            low, high, dtype=np.float32
        )  # Observation space for [x, y, theta]

        self.target_point = tf.Variable(target_point)
        self.shuffle_target_every = shuffle_target_every
        self.num_obstacles = 8 + math.floor(
            float(self.lib.uniform(self.rng, (), 0, 8, self.lib.float32))
        )
        self.initial_state = initial_state
        self.dt = kwargs["dt"]

        if obstacle_positions is None or obstacle_positions == []:
            self.obstacle_positions = []
            # TODO: Assign obstacles here
            # Ensure the planning env takes the same obstacles, does not generate new ones
            # Will have to check this with debugging mode on
            for _ in range(self.num_obstacles):
                self.obstacle_positions.append(
                    list(
                        self.lib.to_numpy(
                            self.lib.uniform(
                                self.rng,
                                (3,),
                                [-0.7, -0.8, 0.05],
                                [0.7, 0.8, 0.3],
                                self.lib.float32,
                            )
                        )
                    )
                )
        else:
            self.obstacle_positions = (
                obstacle_positions  # List of lists [[x_pos, y_pos, radius], ...]
            )

        self.action = [0.0, 0.0]  # Action

        self.config = {
            **kwargs,
            **dict(
                render_mode=self.render_mode,
                initial_state=self.initial_state,
                target_point=self.target_point,
                obstacle_positions=self.obstacle_positions,
                shuffle_target_every=self.shuffle_target_every,
            ),
        }

        self.fig: plt.Figure = None
        self.ax: plt.Axes = None

    def reset(
        self,
        seed: "Optional[int]" = None,
        options: "Optional[dict]" = None,
    ) -> "Tuple[np.ndarray, dict]":
        if seed is not None:
            self._set_up_rng(seed)
        state = options.get("state", None) if isinstance(options, dict) else None
        self.count = 1

        if state is None:
            if self.initial_state is None:
                x = self.lib.uniform(self.rng, (1, 1), -1.0, -0.9, self.lib.float32)
                y = self.lib.uniform(self.rng, (1, 1), -1.0, 1.0, self.lib.float32)
                theta = self.lib.unsqueeze(
                    self.get_heading(
                        self.lib.concat([x, y], 1),
                        self.lib.unsqueeze(
                            self.lib.to_tensor(
                                self.target_point, self.lib.float32
                            ),
                            0,
                        ),
                    ),
                    0,
                )
                yaw = self.lib.uniform(
                    self.rng,
                    (1, 1),
                    theta - MAX_STEER,
                    theta + MAX_STEER,
                    self.lib.float32,
                )
                rate = self.lib.to_tensor([[0.0]], self.lib.float32)

                self.state = self.lib.concat([x, y, yaw, rate], 1)
            else:
                self.state = self.lib.unsqueeze(
                    self.lib.to_numpy(self.initial_state), 0
                )
                x, y, theta, yaw = self.lib.unstack(self.state, 4, 1)
            self.traj_x = [float(x * MAX_X)]
            self.traj_y = [float(y * MAX_Y)]
            self.traj_yaw = [float(yaw)]
            if self._batch_size > 1:
                self.state = np.tile(self.state, (self._batch_size, 1))
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

    def get_reward(self, state, action):
        x, y, yaw_car, steering_rate = self.lib.unstack(state, 4, -1)
        target = self.lib.to_tensor(self.target_point, self.lib.float32)
        x_target, y_target, yaw_target = self.lib.unstack(target, 3, 0)

        head_to_target = self.get_heading(state, self.lib.unsqueeze(target, 0))
        alpha = head_to_target - yaw_car
        ld = self.get_distance(state, self.lib.unsqueeze(target, 0))
        crossTrackError = self.lib.sin(alpha) * ld

        car_in_bounds = self._car_in_bounds(x, y)
        car_at_target = self._car_at_target(x, y, x_target, y_target)

        reward = (
            self.lib.cast(car_in_bounds & car_at_target, self.lib.float32) * 10.0
            + self.lib.cast(car_in_bounds & (~car_at_target), self.lib.float32)
            * (
                -1.0
                * (
                    # 3 * crossTrackError**2
                    0.1 * (x - x_target) ** 2
                    + 0.1 * (y - y_target) ** 2
                    # + 3 * (head_to_target - yaw_car)**2 / MAX_STEER
                    + 5 * self._distance_to_obstacle_cost(x, y)
                )
                / 8.0
            )
            + self.lib.cast(~car_in_bounds, self.lib.float32) * (-1.0)
        )

        return reward

    def is_done(self, state):
        x, y, yaw_car, steering_rate = self.lib.unstack(state, 4, -1)
        target = self.lib.to_tensor(self.target_point, self.lib.float32)
        x_target, y_target, yaw_target = self.lib.unstack(target, 3, 0)

        car_in_bounds = self._car_in_bounds(x, y)
        car_at_target = self._car_at_target(x, y, x_target, y_target)
        done = ~(car_in_bounds & (~car_at_target))
        return done

    def _car_in_bounds(self, x: TensorType, y: TensorType) -> TensorType:
        return (self.lib.abs(x) < 1.0) & (self.lib.abs(y) < 1.0)

    def _car_at_target(
        self, x: TensorType, y: TensorType, x_target: float, y_target: float
    ) -> TensorType:
        return (self.lib.abs(x - x_target) < THRESHOLD_DISTANCE_2_GOAL) & (
            self.lib.abs(y - y_target) < THRESHOLD_DISTANCE_2_GOAL
        )

    def _distance_to_obstacle_cost(self, x: TensorType, y: TensorType) -> TensorType:
        costs = self.lib.unsqueeze(tf.zeros_like(x), -1)
        for obstacle_position in self.obstacle_positions:
            x_obs, y_obs, radius = obstacle_position
            _d = self.lib.sqrt((x - x_obs) ** 2 + (y - y_obs) ** 2)
            _c = 1.0 - (self.lib.min(1.0, _d / radius)) ** 2
            _c = self.lib.unsqueeze(_c, -1)
            costs = self.lib.concat([costs, _c], -1)
        return self.lib.reduce_max(costs[..., 1:], -1)

    def get_distance(self, x1, x2):
        # Distance between points x1 and x2
        return self.lib.sqrt(
            (x1[..., 0] - x2[..., 0]) ** 2 + (x1[..., 1] - x2[..., 1]) ** 2
        )

    def get_heading(self, x1, x2):
        # Heading between points x1,x2 with +X axis
        return self.lib.atan2((x2[..., 1] - x1[..., 1]), (x2[..., 0] - x1[..., 0]))

    @Compile
    def step_dynamics(
        self,
        state: Union[np.ndarray, tf.Tensor, torch.Tensor],
        action: Union[np.ndarray, tf.Tensor, torch.Tensor],
        dt: float,
    ) -> Union[np.ndarray, tf.Tensor, torch.Tensor]:
        return self.update_state(state, action, dt)

    def step(
        self, action: TensorType
    ) -> Tuple[
        TensorType,
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        Union[np.ndarray, bool],
        dict,
    ]:
        if self.count % self.shuffle_target_every == 0:
            target_new = tf.convert_to_tensor(
                [
                    self.target_point[0],
                    self.lib.uniform(self.rng, [], -1.0, 1.0, self.lib.float32),
                    self.target_point[2],
                ]
            )
            self.target_point.assign(target_new)
        self.count += 1
        self.state, action = self._expand_arrays(self.state, action)

        assert self._batch_size == 1
        action = self._apply_actuator_noise(action)

        self.state = self.lib.to_numpy(self.step_dynamics(self.state, action, self.dt))

        terminated = self.is_done(self.state)
        truncated = False
        reward = self.get_reward(self.state, action)

        self.state = self.lib.squeeze(self.state)

        return self.lib.to_numpy(self.state), float(reward), terminated, truncated, {}

    def render(self):
        assert self.render_mode in self.metadata["render_modes"]
        if self.render_mode in {"rgb_array", "single_rgb_array"}:
            # Turn interactive plotting off
            plt.ioff()
        else:
            plt.ion()

        # Storing tracked trajectory
        self.traj_x.append(self.state[0] * MAX_X)
        self.traj_y.append(self.state[1] * MAX_Y)
        self.traj_yaw.append(self.state[2])

        # for stopping simulation with the esc key.
        if self.fig is None:
            self.fig, self.ax = plt.subplots(
                nrows=1, ncols=1, figsize=(6, 6), dpi=100.0
            )
        self.ax.cla()
        self.fig.canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )
        # self.ax: plt.Axes = self.fig.axes[0]
        self.ax.plot(self.traj_x, self.traj_y, "ob", markersize=2, label="trajectory")
        # # Rendering waypoint sequence
        # for i in range(len(self.waypoints)):
        #     self.ax.plot(
        #         self.waypoints[i][0] * MAX_X,
        #         self.waypoints[i][1] * MAX_Y,
        #         "^r",
        #         label="waypoint",
        #     )
        target = self.lib.to_tensor(self.target_point, self.lib.float32)
        self.ax.plot(target[0] * MAX_X, target[1] * MAX_Y, "xg", label="target")
        self.plot_obstacles()
        self.plot_trajectory_plans()
        self.plot_car()
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True)
        self.ax.set_xlim(-MAX_X, MAX_X)
        self.ax.set_ylim(-MAX_Y, MAX_Y)
        # self.ax.set_title("Simulation")
        plt.pause(0.0001)

        if self.render_mode in {"rgb_array", "single_rgb_array"}:
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(
                tuple(
                    (self.fig.get_size_inches() * self.fig.dpi).astype(np.int32)[::-1]
                )
                + (3,)
            )
            return data
        elif self.render_mode in {"human"}:
            self.fig.show()

    def close(self):
        # For Gym AI compatibility
        plt.close(self.fig)

    def update_state(self, state, action, DT):
        x, y, yaw_car, steering_rate = self.lib.unstack(state, 4, 1)
        throttle, steer = self.lib.unstack(action, 2, 1)
        # Update the pose as per Dubin's equations

        steer = self.lib.clip(steer, -MAX_STEER, MAX_STEER)
        throttle = self.lib.clip(throttle, MIN_SPEED, MAX_SPEED)

        x = x + throttle * self.lib.cos(yaw_car) * DT
        y = y + throttle * self.lib.sin(yaw_car) * DT
        steering_rate += steer
        yaw_car = yaw_car + throttle / WB * self.lib.tan(steer) * DT
        return self.lib.stack([x, y, yaw_car, steering_rate], 1)

    def plot_car(self, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
        # print("Plotting Car")
        # Scale up the car pose to MAX_X, MAX_Y grid
        x = self.state[0] * MAX_X
        y = self.state[1] * MAX_Y
        yaw = self.state[2]
        steer = self.action[1] * MAX_STEER

        outline = np.array(
            [
                [
                    -BACKTOWHEEL,
                    (LENGTH - BACKTOWHEEL),
                    (LENGTH - BACKTOWHEEL),
                    -BACKTOWHEEL,
                    -BACKTOWHEEL,
                ],
                [WIDTH / 2, WIDTH / 2, -WIDTH / 2, -WIDTH / 2, WIDTH / 2],
            ]
        )

        fr_wheel = np.array(
            [
                [WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                [
                    -WHEEL_WIDTH - TREAD,
                    -WHEEL_WIDTH - TREAD,
                    WHEEL_WIDTH - TREAD,
                    WHEEL_WIDTH - TREAD,
                    -WHEEL_WIDTH - TREAD,
                ],
            ]
        )

        rr_wheel = np.copy(fr_wheel)

        fl_wheel = np.copy(fr_wheel)
        fl_wheel[1, :] *= -1
        rl_wheel = np.copy(rr_wheel)
        rl_wheel[1, :] *= -1

        Rot1 = np.array(
            [[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]]
        )
        Rot2 = np.array(
            [[math.cos(steer), math.sin(steer)], [-math.sin(steer), math.cos(steer)]]
        )

        fr_wheel = (fr_wheel.T.dot(Rot2)).T
        fl_wheel = (fl_wheel.T.dot(Rot2)).T
        fr_wheel[0, :] += WB
        fl_wheel[0, :] += WB

        fr_wheel = (fr_wheel.T.dot(Rot1)).T
        fl_wheel = (fl_wheel.T.dot(Rot1)).T

        outline = (outline.T.dot(Rot1)).T
        rr_wheel = (rr_wheel.T.dot(Rot1)).T
        rl_wheel = (rl_wheel.T.dot(Rot1)).T

        outline[0, :] += x
        outline[1, :] += y
        fr_wheel[0, :] += x
        fr_wheel[1, :] += y
        rr_wheel[0, :] += x
        rr_wheel[1, :] += y
        fl_wheel[0, :] += x
        fl_wheel[1, :] += y
        rl_wheel[0, :] += x
        rl_wheel[1, :] += y

        self.ax.plot(
            np.array(outline[0, :]).flatten(),
            np.array(outline[1, :]).flatten(),
            truckcolor,
        )
        self.ax.plot(
            np.array(fr_wheel[0, :]).flatten(),
            np.array(fr_wheel[1, :]).flatten(),
            truckcolor,
        )
        self.ax.plot(
            np.array(rr_wheel[0, :]).flatten(),
            np.array(rr_wheel[1, :]).flatten(),
            truckcolor,
        )
        self.ax.plot(
            np.array(fl_wheel[0, :]).flatten(),
            np.array(fl_wheel[1, :]).flatten(),
            truckcolor,
        )
        self.ax.plot(
            np.array(rl_wheel[0, :]).flatten(),
            np.array(rl_wheel[1, :]).flatten(),
            truckcolor,
        )
        self.ax.plot(x, y, "*")

    def plot_obstacles(self):
        for obstacle_position in self.obstacle_positions:
            pos_x, pos_y, radius = obstacle_position
            self.ax.add_patch(
                Circle(
                    (pos_x * MAX_X, pos_y * MAX_Y),
                    radius * np.sqrt(MAX_X * MAX_Y),
                    fill=True,
                    facecolor="dimgray",
                    edgecolor="dimgray",
                    zorder=5,
                )
            )

    def plot_trajectory_plans(self):
        trajectories = self._logs.get("rollout_trajectories_logged", None)[-1]
        costs = self._logs.get("J_logged")[-1]
        if trajectories is not None:
            for i, trajectory in enumerate(trajectories):
                if i == np.argmin(costs):
                    color = "r"
                    alpha = 1.0
                    zorder = 5
                else:
                    color = "g"
                    alpha = min(5.0 / trajectories.shape[0], 1.0)
                    zorder = 4
                self.ax.plot(
                    trajectory[:, 0] * MAX_X,
                    trajectory[:, 1] * MAX_Y,
                    linewidth=0.5,
                    alpha=alpha,
                    color=color,
                    zorder=zorder,
                )
