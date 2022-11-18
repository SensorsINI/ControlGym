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

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from Control_Toolkit.others.environment import EnvironmentBatched
from gymnasium import spaces
from matplotlib.patches import Circle
from SI_Toolkit.computation_library import (ComputationLibrary, NumpyLibrary,
                                            TensorType)
from SI_Toolkit.Functions.TF.Compile import CompileTF
from skspatial.objects import Sphere

from Control_Toolkit.others.globals_and_utils import get_logger

logger = get_logger(__name__)

# Training constants
MAX_ACCELERATION = 5.0
MAX_POSITION = 1.0
MAX_VELOCITY = 1.0

THRESHOLD_DISTANCE_2_GOAL = 0.01
max_ep_length = 800

# Vehicle parameters
LENGTH = 0.45  # [m]
WIDTH = 0.2  # [m]
BACKTOWHEEL = 0.1  # [m]
WHEEL_LEN = 0.03  # [m]
WHEEL_WIDTH = 0.02  # [m]
TREAD = 0.07  # [m]
WB = 0.25  # [m]


class obstacle_avoidance_batched(EnvironmentBatched, gym.Env):
    num_states = None
    num_actions = None
    metadata = {"render_modes": ["console", "single_rgb_array", "rgb_array", "human"], "video.frames_per_second": 30, "render_fps": 30}

    def __init__(
        self,
        target_point,
        shuffle_target_every: int,
        obstacle_positions: "list[list[float]]",
        initial_state: "list[float]",
        num_dimensions: int,
        batch_size=1,
        computation_lib=NumpyLibrary,
        render_mode: str = None,
        **kwargs
    ):
        super().__init__()

        self.set_computation_library(computation_lib)
        self._set_up_rng(kwargs["seed"])
        self.num_dimensions = num_dimensions
        self.__class__.num_states = 2 * self.num_dimensions  # One position and velocity per dimension
        self.__class__.num_actions = self.num_dimensions  # One acceleration per dimension

        self._batch_size = batch_size
        self._actuator_noise = np.array(kwargs["actuator_noise"], dtype=np.float32)
        self.render_mode = render_mode
        
        action_high = np.repeat(MAX_ACCELERATION, self.num_dimensions)
        observation_high = np.concatenate([
            np.repeat(MAX_POSITION, self.num_dimensions),
            np.repeat(MAX_VELOCITY, self.num_dimensions)
        ], dtype=np.float32)
        
        self.action_space = spaces.Box(
            -action_high, action_high, dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            -observation_high, observation_high, dtype=np.float32
        )

        if target_point is None:
            target_point = self.lib.uniform(self.rng, (self.num_dimensions,), -1.0, 1.0, self.lib.float32)
        self.target_point = self.lib.to_variable(target_point, self.lib.float32)
        self.shuffle_target_every = shuffle_target_every
        self.num_obstacles = math.floor(
            float(self.lib.uniform(self.rng, (), 2**self.num_dimensions, 3**self.num_dimensions, self.lib.float32))
        )
        self.initial_state = initial_state
        self.dt = kwargs["dt"]

        if obstacle_positions is None or obstacle_positions == []:
            self.obstacle_positions = []
            range_max = np.repeat(0.9, self.num_dimensions)
            for _ in range(self.num_obstacles):
                self.obstacle_positions.append(
                    list(
                        self.lib.to_numpy(
                            self.lib.uniform(
                                self.rng,
                                (self.num_dimensions + 1,),
                                list(-range_max) + [0.05],
                                list(range_max) + [0.3],
                                self.lib.float32,
                            )
                        )
                    )
                )
        else:
            self.obstacle_positions = (
                obstacle_positions  # List of lists [[x_pos, y_pos, radius], ...]
            )
        self.obstacle_positions = self.lib.to_variable(self.obstacle_positions, self.lib.float32)

        self.config = {
            **kwargs,
            **dict(
                render_mode=self.render_mode,
                initial_state=self.initial_state,
                target_point=self.target_point,
                obstacle_positions=self.obstacle_positions,
                shuffle_target_every=self.shuffle_target_every,
                num_dimensions=self.num_dimensions,
            ),
        }
        self.environment_attributes = {
            "target_point": self.target_point,
            "obstacle_positions": self.obstacle_positions,
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
                positions = self.lib.uniform(self.rng, (1, self.num_dimensions), -MAX_POSITION, MAX_POSITION, self.lib.float32)
                velocities = self.lib.uniform(self.rng, (1, self.num_dimensions), -MAX_VELOCITY, MAX_VELOCITY, self.lib.float32)
                self.state = self.lib.to_numpy(self.lib.concat([positions, velocities], 1))
            else:
                self.state = self.lib.unsqueeze(self.lib.to_numpy(self.initial_state), 0)
            self.traj_x = [float(self.state[..., 0])]
            self.traj_y = [float(self.state[..., 1])]
            if self.num_dimensions >= 3:
                self.traj_z = [float(self.state[..., 2])]
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
    
    @staticmethod
    def _in_bounds(lib: "type[ComputationLibrary]", x: TensorType) -> TensorType:
        return lib.reduce_all(lib.abs(x) < 1.0, -1)

    @staticmethod
    def _at_target(
        lib: "type[ComputationLibrary]",
        x: TensorType,
        target: TensorType,
    ) -> TensorType:
        return lib.reduce_all(
            lib.abs(x - target) < THRESHOLD_DISTANCE_2_GOAL, -1
        )
    
    @staticmethod
    def is_done(lib: "type[ComputationLibrary]", state: TensorType, target_point: TensorType):
        target = lib.to_tensor(target_point, lib.float32)
        num_dimensions = int(lib.shape(state)[-1] / 2)
        position = state[..., :num_dimensions]

        car_in_bounds = obstacle_avoidance_batched._in_bounds(lib, position)
        car_at_target = obstacle_avoidance_batched._at_target(lib, position, target)
        done = ~(car_in_bounds & (~car_at_target))
        return done

    @CompileTF
    def step_dynamics(
        self,
        state: TensorType,
        action: TensorType,
        dt: float,
    ) -> TensorType:
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
            target_new = self.lib.uniform(self.rng, [self.num_dimensions,], -MAX_POSITION, MAX_POSITION, self.lib.float32)
            self.target_point.assign(target_new)
        self.count += 1
        self.state, action = self._expand_arrays(self.state, action)

        assert self._batch_size == 1
        action = self._apply_actuator_noise(action)

        self.state = self.step_dynamics(self.state, action, self.dt)
        self.state = self.lib.to_numpy(self.lib.squeeze(self.state))

        terminated = bool(self.is_done(NumpyLibrary, self.state, self.target_point))
        truncated = False
        reward = 0.0

        return self.state, float(reward), terminated, truncated, {}

    def render(self):
        if self.num_dimensions == 2:
            return self._render2d()
        elif self.num_dimensions == 3:
            return self._render3d()
        else:
            return self._render2d()
    
    def _render2d(self):
        assert self.render_mode in self.metadata["render_modes"]
        if self.render_mode in {"rgb_array", "single_rgb_array"}:
            # Turn interactive plotting off
            plt.ioff()
        else:
            plt.ion()

        # Storing tracked trajectory
        self.traj_x.append(float(self.state[0]))
        self.traj_y.append(float(self.state[1]))
        target = self.lib.to_tensor(self.target_point, self.lib.float32)

        if self.fig is None:
            self.fig, self.ax = plt.subplots(
                nrows=1, ncols=1, figsize=(6, 6), dpi=300.0
            )
            self.ax.cla()
            self.ax.set_aspect("equal", adjustable="datalim")
            self.ax.grid(True)
            self.ax.set_xlim(-1.0, 1.0)
            self.ax.set_ylim(-1.0, 1.0)
            self.fig.canvas.mpl_connect( # for stopping simulation with the esc key.
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
            (self.ln_traj,) = self.ax.plot(self.traj_x, self.traj_y, "ob", markersize=2, label="trajectory", zorder=0, animated=True)
            (self.ln_target,) = self.ax.plot(*target, "xg", label="target", zorder=1, animated=True)
            self.obstacle_patches = self.plot_obstacles()
            self.trajectory_lines = self.plot_trajectory_plans()
            self.point_mass = self.plot_point_mass()
            self.bm = BlitManager(self.fig.canvas, [self.ln_traj, self.ln_target, *self.trajectory_lines, self.point_mass])
            # make sure our window is on the screen and drawn
            if self.render_mode in {"human"}:
                plt.show(block=False)
            plt.pause(1e-6)
            
            self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
            self.ax.draw_artist(self.ln_traj)
            self.ax.draw_artist(self.ln_target)
            self.fig.canvas.blit(self.fig.bbox)
        else:
            self.ln_traj.set_data(self.traj_x, self.traj_y)
            self.ln_target.set_data(*target)
            self.plot_trajectory_plans(self.trajectory_lines)
            self.point_mass.set_center(list(self.state))
            self.bm.update()

        if self.render_mode in {"rgb_array", "single_rgb_array"}:
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(
                tuple(
                    (self.fig.get_size_inches() * self.fig.dpi).astype(np.int32)[::-1]
                )
                + (3,)
            )
            return data
            
    def _render3d(self):
        assert self.render_mode in self.metadata["render_modes"]
        if self.render_mode in {"rgb_array", "single_rgb_array"}:
            # Turn interactive plotting off
            plt.ioff()
        else:
            plt.ion()
        
        

        # Storing tracked trajectory
        self.traj_x.append(float(self.state[0]))
        self.traj_y.append(float(self.state[1]))
        self.traj_z.append(float(self.state[2]))
        target = [[float(k)] for k in self.target_point]
        position = [[float(k)] for k in self.state[:3]]

        # for stopping simulation with the esc key.
        if self.fig is None:
            self.fig = plt.figure(figsize=(6, 6), dpi=300.0)
            self.ax = plt.axes(projection='3d')

            self.ax.cla()
            self.ax.set_aspect("equal", adjustable="datalim")
            self.ax.grid(True)
            self.ax.set_xlim(-1.0, 1.0)
            self.ax.set_ylim(-1.0, 1.0)
            self.ax.set_zlim(-1.0, 1.0)
            self.fig.canvas.mpl_connect( # for stopping simulation with the esc key.
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
            (self.ln_traj,) = self.ax.plot3D(self.traj_x, self.traj_y, self.traj_z, "ob", markersize=0.5, label="trajectory", animated=True)
            (self.ln_target,) = self.ax.plot3D(*target, "xg", label="target", animated=True)
            self.point_mass = self.plot_point_mass()
            self.trajectory_lines = self.plot_trajectory_plans()
            self.obstacle_patches = self.plot_obstacles()
            self.bm = BlitManager(self.fig.canvas, [self.ln_traj, self.ln_target, self.point_mass, *self.trajectory_lines])
            # make sure our window is on the screen and drawn
            if self.render_mode in {"human"}:
                plt.show(block=False)
            plt.pause(1e-6)
            
            self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
            self.ax.draw_artist(self.ln_traj)
            self.ax.draw_artist(self.ln_target)
            self.ax.draw_artist(self.point_mass)
            self.fig.canvas.blit(self.fig.bbox)
        else:
            # self.ax.view_init(elev=10., azim=self.ax.get_view().azim)
            self.ln_traj.set_data(self.traj_x, self.traj_y)
            self.ln_traj.set_3d_properties(self.traj_z)
            self.ln_target.set_data(target[0], target[1])
            self.ln_target.set_3d_properties(target[2])
            self.point_mass.set_data(position[0], position[1])
            self.point_mass.set_3d_properties(position[2])
            self.plot_trajectory_plans(self.trajectory_lines)
            self.bm.update()

        if self.render_mode in {"rgb_array", "single_rgb_array"}:
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(
                tuple(
                    (self.fig.get_size_inches() * self.fig.dpi).astype(np.int32)[::-1]
                )
                + (3,)
            )
            return data

    def close(self):
        # For Gym AI compatibility
        plt.close(self.fig)

    def update_state(self, state, action, DT):
        position, velocity = state[:, :self.num_dimensions], state[:, self.num_dimensions:]

        position += DT * velocity
        velocity += DT * action
        
        return self.lib.concat([position, velocity], 1)

    def plot_point_mass(self):
        x = self.state[0]
        y = self.state[1]
        r = 0.05
        if self.num_dimensions == 3:
            z = self.state[2]
            (patch,) = self.ax.plot3D([x], [y], [z], "or", markersize=4, label="point_mass", animated=True)
        else:
            patch = Circle(
                (x, y),
                r,
                fill=True,
                facecolor="red",
                edgecolor="red",
                zorder=5,
            )
            self.ax.add_patch(patch)
        return patch

    def plot_obstacles(self):
        patches = []
        if self.num_dimensions == 3:
            for obstacle_position in self.obstacle_positions:
                pos_x, pos_y, pos_z, radius = obstacle_position
                sphere = Sphere([pos_x, pos_y, pos_z], radius)
                sphere.plot_3d(self.ax, n_angles=30, alpha=0.5, color="dimgray")
                patches.append(sphere)
        else:
            for obstacle_position in self.obstacle_positions:
                pos_x, pos_y, radius = obstacle_position
                circle = Circle(
                    (pos_x, pos_y),
                    radius,
                    fill=True,
                    facecolor="dimgray",
                    edgecolor="dimgray",
                    zorder=5,
                )
                self.ax.add_patch(circle)
                patches.append(circle)
        return patches

    def plot_trajectory_plans(self, lines=None):
        trajectories = self._logs.get("rollout_trajectories_logged", None)
        costs = self._logs.get("J_logged")
        create_new_lines = lines == None
        lines = [] if lines == None else lines
        
        if len(trajectories) and len(costs):
            trajectories = trajectories[-1]
            costs = costs[-1]
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
                    if create_new_lines:
                        if self.num_dimensions == 3:
                            (ln,) = self.ax.plot3D(
                                trajectory[:, 0],
                                trajectory[:, 1],
                                trajectory[:, 2],
                                linewidth=0.5,
                                alpha=alpha,
                                color=color,
                                zorder=zorder,
                            )
                        else:
                            (ln,) = self.ax.plot(
                                trajectory[:, 0],
                                trajectory[:, 1],
                                linewidth=0.5,
                                alpha=alpha,
                                color=color,
                                zorder=zorder,
                            )
                        lines.append(ln)
                    else:
                        if self.num_dimensions == 3:
                            lines[i].set_data(trajectory[:, 0], trajectory[:, 1])
                            lines[i].set_3d_properties(trajectory[:, 2])
                        else:
                            lines[i].set_data(trajectory[:, 0], trajectory[:, 1])
        return lines


class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for sub-classes of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()
