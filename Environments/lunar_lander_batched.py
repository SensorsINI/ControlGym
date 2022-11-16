
from typing import Optional, Tuple, Union
import numpy as np
from gymnasium.envs.box2d.lunar_lander import LunarLander

from Control_Toolkit.others.environment import EnvironmentBatched
from SI_Toolkit.computation_library import ComputationLibrary, NumpyLibrary, TensorType, RandomGeneratorType


SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

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
    def __init__(self, lib: "type[ComputationLibrary]", rng: RandomGeneratorType) -> None:
        self.ground_height = 1.0
        self.lib = lib
        self.rng = rng
    
    def touched(self, pos_x, pos_y, angle):
        tip = (self.lib.sin(angle), self.lib.cos(angle))
        side = (-tip[1], tip[0])
        dispersion = [self.lib.uniform(self.rng, self.lib.shape(angle), -1.0, +1.0, self.lib.float32) / SCALE for _ in range(2)]
        ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
        oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
        
        return self.lib.cast(pos_y + oy < self.ground_height, self.lib.float32)


class lunar_lander_batched(EnvironmentBatched, LunarLander):
    """Accepts batches of data to environment
    
    Uses the continuous version of LunarLander as base class
    """

    num_actions = 2  # throttle of the main and left/right engines
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
        self.FPS = 1.0 / self.dt
        
        self.continuous = True
        self._batch_size = batch_size
        self._actuator_noise = np.array(kwargs["actuator_noise"], dtype=np.float32)

        self.set_computation_library(computation_lib)
        self._set_up_rng(kwargs["seed"])
        
        self.ground_contact_detector = GroundContactDetector(self.lib, self.rng)
    
    def step_dynamics(self, state: TensorType, action: TensorType, dt: float) -> TensorType:
        pos_x, pos_y, vel_x, vel_y, angle, vel_angle, contact_1, contact_2 = self.lib.unstack(state, 8, 1)
        
        # Convert state from view-related variables to actual variables
        pos_x = pos_x * (VIEWPORT_W / SCALE / 2) + (VIEWPORT_W / SCALE / 2)
        pos_y = pos_y * (VIEWPORT_H / SCALE / 2) + (self.helipad_y + LEG_DOWN / SCALE)
        vel_x = vel_x * self.FPS / (VIEWPORT_W / SCALE / 2)
        vel_y = vel_y * self.FPS / (VIEWPORT_H / SCALE / 2)
        angle = angle
        vel_angle = vel_angle * self.FPS / 20.0
        contact_1 = contact_1
        contact_2 = contact_2
        
        # Define variables for state derivatives
        acc_x = self.lib.zeros_like(vel_x)
        acc_y = self.lib.zeros_like(vel_y)
        angleDD = self.lib.zeros_like(angle)
        
        if self.enable_wind and not (contact_1 or contact_2):
            # Define horizontal wind disturbance
            wind_mag = (
                self.lib.tanh(
                    self.lib.sin(0.02 * self.wind_idx)
                    + (self.lib.sin(self.lib.pi * 0.01 * self.wind_idx))
                )
                * self.wind_power
            )
            self.wind_idx += 1  # TODO: This might not work in compile mode
            acc_x += wind_mag / LANDER_MASS
            
            # Define rotational turbulence disturbance
            torque_mag = self.lib.tanh(
                self.lib.sin(0.02 * self.torque_idx)
                + (self.lib.sin(self.lib.pi * 0.01 * self.torque_idx))
            ) * (self.turbulence_power)
            self.torque_idx += 1
            angleDD += torque_mag / LANDER_INERTIA
        
        # Prepare action
        action = self.lib.clip(action, -1.0, 1.0)
        throttle_main, throttle_lr = self.lib.unstack(action, 2, 1)
        
        # Engines
        tip = (self.lib.sin(angle), self.lib.cos(angle))
        side = (-tip[1], tip[0])
        dispersion = [self.lib.uniform(self.rng, self.lib.shape(angle), -1.0, +1.0, self.lib.float32) / SCALE for _ in range(2)]

        m_power = 0.0
        
        throttle_main_mask = self.lib.cast(throttle_main > 0.0, self.lib.float32)
        # Main engine power, if <=0.0 then engine is off
        m_power = throttle_main_mask * (self.lib.clip(throttle_main, 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
        
        # 4 is move a bit downwards, +-2 for randomness
        ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
        oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
        impulse_pos = (pos_x + ox, pos_y + oy)
        # p = self._create_particle(
        #     3.5,  # 3.5 is here to make particle speed adequate
        #     impulse_pos[0],
        #     impulse_pos[1],
        #     m_power,
        # )  # particles are just a decoration
        # p.ApplyLinearImpulse(
        #     (ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
        #     impulse_pos,
        #     True,
        # )
        acc_x += -ox * MAIN_ENGINE_POWER * m_power / LANDER_MASS
        acc_y += -oy * MAIN_ENGINE_POWER * m_power / LANDER_MASS
        
        s_power = self.lib.zeros_like(throttle_lr)
        
        throttle_lr_mask = self.lib.cast(self.lib.abs(throttle_lr) > 0.5, self.lib.float32)
        # Orientation engines
        direction = self.lib.sign(throttle_lr)
        s_power = throttle_lr_mask * self.lib.clip(self.lib.abs(throttle_lr), 0.5, 1.0)
        
        ox = tip[0] * dispersion[0] + side[0] * (
            3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
        )
        oy = -tip[1] * dispersion[0] - side[1] * (
            3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
        )
        
        impulse_pos = (
            pos_x + ox - tip[0] * 17 / SCALE,
            pos_y + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
        )
        # p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
        # p.ApplyLinearImpulse(
        #     (ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
        #     impulse_pos,
        #     True,
        # )
        acc_x += -ox * SIDE_ENGINE_POWER * s_power / LANDER_MASS
        acc_y += -oy * SIDE_ENGINE_POWER * s_power / LANDER_MASS
        angleDD += (tip[0] * 17 / SCALE) * s_power / LANDER_INERTIA
        
        # Euler integration
        pos_x += self.dt * vel_x
        pos_y += self.dt * vel_y
        vel_x += self.dt * acc_x
        vel_y += self.dt * acc_y
        angle += self.dt * vel_angle
        vel_angle += self.dt * angleDD
        
        contact_1 = self.ground_contact_detector.touched(pos_x, pos_y, angle)
        contact_2 = contact_1
        
        # Recover state info
        state_updated = self.lib.stack([
            (pos_x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos_y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel_x * (VIEWPORT_W / SCALE / 2) / self.FPS,
            vel_y * (VIEWPORT_H / SCALE / 2) / self.FPS,
            angle,
            20.0 * vel_angle / self.FPS,
            contact_1,
            contact_2,
        ], 1)
        
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

        terminated = bool(self.is_done(self.lib, self.state))
        truncated = False
        reward = 0.0

        self.state = self.lib.squeeze(self.state)

        return (
            self.lib.to_numpy(self.lib.squeeze(self.state)),
            float(reward),
            terminated,
            truncated,
            {},
        )
    
    def reset(self, seed: "Optional[int]" = None, options: "Optional[dict]" = None) -> "Tuple[np.ndarray, dict]":
        self.ground_contact_detector = GroundContactDetector(self.lib, self.rng)
        self.game_over = False
        self.prev_shaping = None

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
        # smooth_y = [
        #     0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
        #     for i in range(CHUNKS)
        # ]

        # self.moon = self.world.CreateStaticBody(
        #     shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        # )
        # self.sky_polys = []
        # for i in range(CHUNKS - 1):
        #     p1 = (chunk_x[i], smooth_y[i])
        #     p2 = (chunk_x[i + 1], smooth_y[i + 1])
        #     self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
        #     self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        # self.moon.color1 = (0.0, 0.0, 0.0)
        # self.moon.color2 = (0.0, 0.0, 0.0)

        initial_y = VIEWPORT_H / SCALE
        # self.lander: Box2D.b2Body = self.world.CreateDynamicBody(
        #     position=(VIEWPORT_W / SCALE / 2, initial_y),
        #     angle=0.0,
        #     fixtures=fixtureDef(
        #         shape=polygonShape(
        #             vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
        #         ),
        #         density=5.0,
        #         friction=0.1,
        #         categoryBits=0x0010,
        #         maskBits=0x001,  # collide only with ground
        #         restitution=0.0,
        #     ),  # 0.99 bouncy
        # )
        # self.lander.color1 = (128, 102, 230)
        # self.lander.color2 = (77, 77, 128)
        # self.lander.ApplyForceToCenter(
        #     (
        #         self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
        #         self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
        #     ),
        #     True,
        # )
        

        # self.legs = []
        # for i in [-1, +1]:
        #     leg = self.world.CreateDynamicBody(
        #         position=(VIEWPORT_W / SCALE / 2 - i * LEG_AWAY / SCALE, initial_y),
        #         angle=(i * 0.05),
        #         fixtures=fixtureDef(
        #             shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
        #             density=1.0,
        #             restitution=0.0,
        #             categoryBits=0x0020,
        #             maskBits=0x001,
        #         ),
        #     )
        #     leg.ground_contact = False
        #     leg.color1 = (128, 102, 230)
        #     leg.color2 = (77, 77, 128)
        #     rjd = revoluteJointDef(
        #         bodyA=self.lander,
        #         bodyB=leg,
        #         localAnchorA=(0, 0),
        #         localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
        #         enableMotor=True,
        #         enableLimit=True,
        #         maxMotorTorque=LEG_SPRING_TORQUE,
        #         motorSpeed=+0.3 * i,  # low enough not to jump back into the sky
        #     )
        #     if i == -1:
        #         rjd.lowerAngle = (
        #             +0.9 - 0.5
        #         )  # The most esoteric numbers here, angled legs have freedom to travel within
        #         rjd.upperAngle = +0.9
        #     else:
        #         rjd.lowerAngle = -0.9
        #         rjd.upperAngle = -0.9 + 0.5
        #     leg.joint = self.world.CreateJoint(rjd)
        #     self.legs.append(leg)

        # self.drawlist = [self.lander] + self.legs

        # if self.render_mode == "human":
        #     self.render()
        # return self.step(np.array([0, 0]) if self.continuous else 0)[0], {}
    
    
    
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
            
            self.state = self.lib.concat([pos_x, pos_y, vel_x, vel_y, angle, vel_angle, self.lib.zeros((self._batch_size, 1)), self.lib.zeros((self._batch_size, 1))], 1)
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
    def is_done(lib: "type[ComputationLibrary]", state: TensorType, *args, **kwargs):
        return False