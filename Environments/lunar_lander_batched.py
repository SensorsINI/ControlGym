
from typing import Optional, Tuple, Union
import numpy as np
from gymnasium.envs.box2d.lunar_lander import LunarLander
from gymnasium import spaces
import gymnasium as gym

from Control_Toolkit.others.environment import EnvironmentBatched
from SI_Toolkit.computation_library import ComputationLibrary, NumpyLibrary, TensorType, RandomGeneratorType

try:
    import pygame
    from pygame import gfxdraw
except ImportError:
    raise gym.error.DependencyNotInstalled(
        "pygame is not installed, run `pip install gymnasium[box2d]`"
    )


SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 10 * 13.0
SIDE_ENGINE_POWER = 200 * 0.6

LANDER_MASS = 5.0
LANDER_INERTIA = 5.0

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400


class GroundContactDetector:
    def __init__(self, lib: "type[ComputationLibrary]", sky_polys: list) -> None:
        self.ground_height = 1.0
        self.lib = lib
        self.lander_points = np.array(LANDER_POLY, np.float32)
        self.sky_polys_full = self.lib.to_variable(sky_polys, self.lib.float32)
        self.sky_polys = self.lib.to_variable(self.sky_polys_full[:, :2, :], self.lib.float32)
    
    def set_sky_polys(self, sky_polys: list):
        self.lib.assign(self.sky_polys_full, sky_polys)
        sky_polys_full = self.lib.to_numpy(self.sky_polys_full)
        sky_polys_full[:, :, 0] = sky_polys_full[:, :, 0] * 2 * SCALE / VIEWPORT_W - 1.0
        sky_polys_full[:, :, 1] = sky_polys_full[:, :, 1] * 2 * SCALE / VIEWPORT_H - 1.0
        self.lib.assign(self.sky_polys, sky_polys_full[:, :2, :])  # Select only the coordinates describing the boundary between sky and ground
    
    def surface_y_at_point(self, x: TensorType):
        # Determine which sky segment is needed
        sky_idcs = self.lib.cast(
            ((self.lib.clip(x, -0.9999, 0.9999) + 1.0) * 5),
            self.lib.int32
        )
        sky_line_segments = self.lib.gather(self.sky_polys, sky_idcs, 0)
        
        # Determine linear interpolation between line segment end points
        fraction_between_points = (x - sky_line_segments[..., 0, 0]) / (sky_line_segments[..., 1, 0] - sky_line_segments[..., 0, 0])
        segment_height_at_point = (1.0 - fraction_between_points) * sky_line_segments[..., 0, 1] + fraction_between_points * sky_line_segments[..., 1, 1]
        
        return segment_height_at_point
    
    def touched(self, pos_x: TensorType, pos_y: TensorType, angle: TensorType):
        # Dimensions: [batch, points_of_lunar_lander, x/y]
        lander_outline_point = self.lib.repeat(
            self.lib.stack(
                [self.lander_points[:, 0] * 2 / VIEWPORT_W, -self.lander_points[:, 1] * 2 / VIEWPORT_H], 1
            )[self.lib.newaxis, :, :, self.lib.newaxis],
            self.lib.shape(pos_x),
            0
        )
            
        # Rotate point around origin
        rot_matrix = self.lib.repeat(
            self.lib.permute(
                self.lib.to_tensor(
                    [[self.lib.cos(angle), -self.lib.sin(angle)], [self.lib.sin(angle), self.lib.cos(angle)]],
                    self.lib.float32
                ),
                (2, 0, 1)
            )[:, self.lib.newaxis, :, :],
            self.lib.shape(lander_outline_point)[1],
            1
        )
        lander_outline_point = self.lib.matmul(rot_matrix, lander_outline_point)[:, :, :, 0]
        lander_outline_point += self.lib.repeat(
            self.lib.stack([pos_x, pos_y], axis=1)[:, self.lib.newaxis, :],
            self.lib.shape(lander_outline_point)[1],
            1
        )
        
        segment_height_at_point = self.surface_y_at_point(lander_outline_point[:, :, 0])  
        
        # Or-connection for all lander outline points
        touched = self.lib.sum(self.lib.cast(lander_outline_point[:, :, 1] <= segment_height_at_point, self.lib.int32), 1)
            
        return self.lib.cast(self.lib.clip(touched, 0, 1), self.lib.float32)


class lunar_lander_batched(EnvironmentBatched, LunarLander):
    """Accepts batches of data to environment
    
    Uses the continuous version of LunarLander as base class
    """

    num_actions = 2  # throttle of the main and left/right engines
    # The state is a 7-dimensional vector:
    # - the coordinates of the lander in x & y
    # - its linear velocities in x & y
    # - its angle
    # - its angular velocity
    # - a booleans that represents whether the lander is in contact with ground
    num_states = 7
    
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
        self.FPS = 1.0 / self.dt
        
        self.continuous = True
        self._batch_size = batch_size
        self._actuator_noise = np.array(kwargs["actuator_noise"], dtype=np.float32)
        
        low = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                -1.5,
                -1.5,
                # velocity bounds is 5x rated speed
                -5.0,
                -5.0,
                -np.pi,
                -5.0,
                -0.0,
            ]
        ).astype(np.float32)
        high = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                1.5,
                1.5,
                # velocity bounds is 5x rated speed
                5.0,
                5.0,
                np.pi,
                5.0,
                1.0,
            ]
        ).astype(np.float32)
        self.observation_space = spaces.Box(low, high)

        self.set_computation_library(computation_lib)
        self._set_up_rng(kwargs["seed"])
        
        self.target_point = self.lib.to_variable([[0.0, 0.0]], self.lib.float32)
        self.sky_polys = self.init_sky_polys()
        self.ground_contact_detector = GroundContactDetector(self.lib, self.sky_polys)
        self.environment_attributes = {
            "target_point": self.target_point,
            "ground_contact_detector": self.ground_contact_detector,
        }
    
    def step_dynamics(self, state: TensorType, action: TensorType, dt: float) -> TensorType:
        pos_x, pos_y, vel_x, vel_y, angle, vel_angle, contact = self.lib.unstack(state, 7, 1)
        
        # Define variables for state derivatives
        acc_x = self.lib.zeros_like(vel_x)
        acc_y = self.gravity * self.lib.ones_like(vel_y)
        angleDD = self.lib.zeros_like(angle)
        
        # Disturbances
        contact_mask = self.enable_wind * (1.0 - self.lib.cast(contact, self.lib.float32))  # not contact
        # Define horizontal wind disturbance
        wind_mag = (
            self.lib.tanh(
                self.lib.sin(0.02 * self.wind_idx)
                + (self.lib.sin(self.lib.pi * 0.01 * self.wind_idx))
            )
            * self.wind_power
        )
        self.wind_idx += 1  # TODO: This might not work in compile mode
        acc_x += contact_mask * wind_mag / LANDER_MASS
        
        # Define rotational turbulence disturbance
        torque_mag = self.lib.tanh(
            self.lib.sin(0.02 * self.torque_idx)
            + (self.lib.sin(self.lib.pi * 0.01 * self.torque_idx))
        ) * (self.turbulence_power)
        self.torque_idx += 1
        angleDD += contact_mask * torque_mag / LANDER_INERTIA
        
        # Prepare action
        action = self.lib.clip(action, -1.0, 1.0)
        throttle_main, throttle_lr = self.lib.unstack(action, 2, 1)
        
        # Engines
        tip = (self.lib.sin(angle), self.lib.cos(angle))
        side = (-tip[1], tip[0])
        dispersion = [self.lib.uniform(self.rng, self.lib.shape(angle), -1.0, +1.0, self.lib.float32) / SCALE for _ in range(2)]

        m_power = 0.0
        
        # Main engine power, if <=0.0 then engine is off
        throttle_main_mask = self.lib.cast(throttle_main > 0.0, self.lib.float32)
        m_power = throttle_main_mask * (self.lib.clip(throttle_main, 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
        acc_x += tip[0] * MAIN_ENGINE_POWER * m_power / LANDER_MASS
        acc_y += tip[1] * MAIN_ENGINE_POWER * m_power / LANDER_MASS
        
        # Orientation engines
        throttle_lr_mask = self.lib.cast(self.lib.abs(throttle_lr) > 0.5, self.lib.float32)
        direction = self.lib.sign(throttle_lr)
        s_power = throttle_lr_mask * self.lib.clip(self.lib.abs(throttle_lr), 0.5, 1.0)
        
        # acc_x += tip[0] * SIDE_ENGINE_POWER * s_power / LANDER_MASS
        # acc_y += tip[1] * SIDE_ENGINE_POWER * s_power / LANDER_MASS
        angleDD += SIDE_ENGINE_POWER * direction * s_power / LANDER_INERTIA
        
        # Euler integration
        pos_x_updated = pos_x + self.dt * vel_x
        pos_y_updated = pos_y + self.dt * vel_y
        vel_x_updated = vel_x + self.dt * acc_x
        vel_y_updated = vel_y + self.dt * acc_y
        angle_updated = angle + self.dt * vel_angle
        vel_angle_updated = vel_angle + self.dt * angleDD
        
        contact_updated = self.ground_contact_detector.touched(pos_x, pos_y, angle)
        
        state_updated = self.lib.stack([pos_x_updated, pos_y_updated, vel_x_updated, vel_y_updated, angle_updated, vel_angle_updated, contact_updated], 1)
        
        return state_updated

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
        assert self._batch_size == 1
        action = self._apply_actuator_noise(action)

        state_updated: TensorType = self.step_dynamics(self.state, action, self.dt)
        self.state = self.lib.to_numpy(state_updated)

        terminated = bool(self.is_done(self.lib, self.state, self.target_point))
        truncated = bool(self.is_truncated(self.state, self.target_point))
        reward = 0.0

        self.state = self.lib.squeeze(self.state)

        return (
            self.lib.to_numpy(self.lib.squeeze(self.state)),
            float(reward),
            terminated,
            truncated,
            {},
        )
        
    def init_sky_polys(self):
        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE
        
        # terrain
        CHUNKS = 11
        height = self.lib.uniform(self.rng, [CHUNKS + 1,], 0, H / 2, self.lib.float32).numpy()
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]
        
        sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
            
        return sky_polys
    
    def reset(self, seed: "Optional[int]" = None, options: "Optional[dict]" = None) -> "Tuple[np.ndarray, dict]":
        self.game_over = False
        self.prev_shaping = None

        self.sky_polys = self.init_sky_polys()
        self.ground_contact_detector.set_sky_polys(self.sky_polys)
        
        target_x = self.lib.uniform(self.rng, (1, 1), -0.8, 0.8, self.lib.float32)
        self.lib.assign(self.target_point, self.lib.concat([target_x, self.ground_contact_detector.surface_y_at_point(target_x)], 1))
    
        if seed is not None:
            self._set_up_rng(seed)
        state = options.get("state", None) if isinstance(options, dict) else None
        self.count = 1

        if state is None:
            pos_x = self.lib.uniform(self.rng, (self._batch_size, 1), -0.6, 0.6, self.lib.float32)
            pos_y = self.lib.uniform(self.rng, (self._batch_size, 1), 0.6, 1.0, self.lib.float32)
            vel_x = self.lib.uniform(self.rng, (self._batch_size, 1), -0.6, 0.6, self.lib.float32)
            vel_y = self.lib.uniform(self.rng, (self._batch_size, 1), -0.6, 0.6, self.lib.float32)
            angle = self.lib.uniform(self.rng, (self._batch_size, 1), -0.2, 0.2, self.lib.float32)
            vel_angle = self.lib.uniform(self.rng, (self._batch_size, 1), -1.0, 1.0, self.lib.float32)
            
            self.state = self.lib.concat([pos_x, pos_y, vel_x, vel_y, angle, vel_angle, self.lib.zeros((self._batch_size, 1))], 1)
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
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())  # Draw white background

        for p in self.sky_polys:
            scaled_poly = []
            for coord in p:
                scaled_poly.append((coord[0] * SCALE, VIEWPORT_H - coord[1] * SCALE))
            pygame.draw.polygon(self.surf, (0, 0, 0), scaled_poly)
            # gfxdraw.aapolygon(self.surf, scaled_poly, (0, 0, 0))
        
        coords = []
        for c in LANDER_POLY:
            lander_outline_point = pygame.math.Vector2((c[0], -c[1])).rotate_rad(self.state[4])
            c = pygame.math.Vector2((
                lander_outline_point[0] + float(self.state[0]) * (VIEWPORT_W / 2) + (VIEWPORT_W / 2),
                lander_outline_point[1] + float(self.state[1]) * (-VIEWPORT_H / 2) + (VIEWPORT_H / 2)
            ))
            coords.append((c[0], c[1]))
        pygame.draw.polygon(self.surf, (220, 10, 10), coords)
        
        # Draw target
        target = list(self.target_point[0, :].numpy())
        target_x_scaled = target[0] * (VIEWPORT_W / 2) + (VIEWPORT_W / 2)
        target_y_scaled = target[1] * (-VIEWPORT_H / 2) + (VIEWPORT_H / 2)
        pygame.draw.line(
            self.surf,
            color=(255, 255, 255),
            start_pos=[target_x_scaled, target_y_scaled],
            end_pos=[target_x_scaled, target_y_scaled - 0.1 * VIEWPORT_H],
            width=1,
        )
        pygame.draw.polygon(
            self.surf,
            color=(204, 204, 0),
            points=[
                (target_x_scaled, target_y_scaled - 0.10 * VIEWPORT_H),
                (target_x_scaled, target_y_scaled - 0.05 * VIEWPORT_H),
                (target_x_scaled + 0.05 * VIEWPORT_W, target_y_scaled - 0.075 * VIEWPORT_H),
            ],
        )
        
        # Render rollouts
        trajectories = self.logs.get("rollout_trajectories_logged", [])
        costs = self.logs.get("J_logged", [])
        
        if len(trajectories) and len(costs):
            trajectories = trajectories[-1]
            costs = costs[-1]
            if trajectories is not None:
                trajectories[:, :, 0] = trajectories[:, :, 0] * (VIEWPORT_W / 2) + (VIEWPORT_W / 2)
                trajectories[:, :, 1] = trajectories[:, :, 1] * (-VIEWPORT_H / 2) + (VIEWPORT_H / 2)
                for i, trajectory in enumerate(trajectories):
                    if i == np.argmin(costs):
                        alpha = 1.0
                        color = (255, 0, 0, alpha)
                    else:
                        alpha = min(2.0 / trajectories.shape[0], 1.0)
                        color = (0, 255, 0, alpha)
                    pygame.draw.lines(self.surf, color, False, trajectory[:, :2], width=1)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )
    
    @staticmethod
    def is_done(lib: "type[ComputationLibrary]", state: TensorType, target_point: TensorType):
        pos_x, pos_y, vel_x, vel_y, angle, vel_angle, contact = lib.unstack(state, 7, -1)
        target_point = lib.to_tensor(target_point, lib.float32)
        
        return (
            lib.cast(contact, lib.bool)
            & ((pos_x - target_point[0, 0]) ** 2 < 0.02)
            & ((pos_y - target_point[0, 1]) ** 2 < 0.02)
            & (vel_y ** 2 < 0.1)
        )
    
    def is_truncated(self, state: TensorType, target_point: TensorType):
        pos_x, pos_y, vel_x, vel_y, angle, vel_angle, contact = self.lib.unstack(state, 7, -1)
        target_point = self.lib.to_tensor(target_point, self.lib.float32)
        
        return (
            self.lib.cast(contact, self.lib.bool) & ~lunar_lander_batched.is_done(self.lib, state, target_point)  # lander touched ground but not at target and not soft enough
            | (self.lib.abs(pos_x) > 1.0)  # Out of bounds
            | (self.lib.abs(pos_y) > 1.0)  # Out of bounds
        )
        
        