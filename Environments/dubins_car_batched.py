# Original Source: https://github.com/gargivaidya/dubin_model_gymenv/blob/main/dubin_gymenv.py

from typing import Optional, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import math
import gym
from gym import spaces
import time
import itertools
import datetime
import tensorflow as tf

from Environments import EnvironmentBatched, NumpyLibrary, cost_functions

# Training constants
MAX_STEER = np.pi / 2
MAX_SPEED = 10.0
MIN_SPEED = 0.0
THRESHOLD_DISTANCE_2_GOAL = 0.05
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
    num_states = 3
    num_actions = 2
    metadata = {"render_modes": ["console", "single_rgb_array", "rgb_array", "human"]}

    def __init__(
        self,
        start_point,
        waypoints,
        target_point,
        n_waypoints,
        batch_size=1,
        computation_lib=NumpyLibrary,
        **kwargs
    ):
        super(dubins_car_batched, self).__init__()

        self.config = {
            **kwargs,
            **dict(
                start_point=start_point,
                waypoints=waypoints,
                target_point=target_point,
                n_waypoints=n_waypoints,
            ),
        }
        self._batch_size = batch_size
        self._actuator_noise = np.array(kwargs["actuator_noise"], dtype=np.float32)

        self.action_space = spaces.Box(
            np.array([0.0, -1.57]), np.array([1.0, 1.57]), dtype=np.float32
        )  # Action space for [throttle, steer]
        low = np.array([-1.0, -1.0, -4.0])  # low range of observation space
        high = np.array([1.0, 1.0, 4.0])  # high range of observation space
        self.observation_space = spaces.Box(
            low, high, dtype=np.float32
        )  # Observation space for [x, y, theta]
        self.waypoints = np.divide(waypoints, MAX_X).astype(
            np.float32
        )  # List of waypoints without start and final goal point
        self.look_ahead = self.waypoints[
            0:n_waypoints
        ]  # List of look-ahead waypoints from closest waypoint
        self.target = [
            target_point[0] / MAX_X,
            target_point[1] / MAX_Y,
            target_point[2],
        ]  # Final goal point of trajectory
        self.state = [
            start_point[0] / MAX_X,
            start_point[1] / MAX_Y,
            start_point[2],
        ]  # Current pose of car
        self.action = [0.0, 0.0]  # Action
        self.traj_x = [
            self.state[0] * MAX_X
        ]  # List of tracked trajectory for rendering
        self.traj_y = [
            self.state[1] * MAX_Y
        ]  # List of tracked trajectory for rendering
        self.traj_yaw = [self.state[2]]  # List of tracked trajectory for rendering
        self.n_waypoints = n_waypoints  # Number of look-ahead waypoints
        self.d_to_waypoints = np.zeros(
            (self._batch_size, n_waypoints)
        )  # List of distance of car from current position to every waypoint
        self.closest_idx = 0  # Index of closest waypoint

        self.set_computation_library(computation_lib)
        self._set_up_rng(kwargs["seed"])
        self.cost_functions = cost_functions(self)

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
            self.state = self.lib.to_tensor(
                [0.0, 0.0, 1.57], self.lib.float32
            )  # Resets to Origin
            self.traj_x = [0.0 * MAX_X]
            self.traj_y = [0.0 * MAX_Y]
            self.traj_yaw = [1.57]
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
        state, action = self._expand_arrays(state, action)
        x, y, yaw_car = self.lib.unstack(state, 3, 1)
        x_target = self.target[0]
        y_target = self.target[1]

        head = self.lib.atan2((y_target - y), (x_target - x + 0.01))

        cond1 = (self.lib.abs(x) < 1.0) & (self.lib.abs(y) < 1.0)
        cond2 = (self.lib.abs(x - x_target) < THRESHOLD_DISTANCE_2_GOAL) & (
            self.lib.abs(y - y_target) < THRESHOLD_DISTANCE_2_GOAL
        )

        reward = (
            self.lib.cast(cond1 & cond2, self.lib.float32) * 10.0
            + self.lib.cast(cond1 & (~cond2), self.lib.float32)
            * (
                -1.0
                * (
                    self.lib.abs(x - x_target)
                    + self.lib.abs(y - y_target)
                    + self.lib.abs(head - yaw_car)
                )
            )
            + self.lib.cast(~cond1, self.lib.float32) * (-1.0)
        )

        return reward

    def is_done(self, state):
        x, y, yaw_car = self.lib.unstack(state, 3, 1)
        x_target = self.target[0]
        y_target = self.target[1]

        cond1 = (self.lib.abs(x) < 1.0) & (self.lib.abs(y) < 1.0)
        cond2 = (self.lib.abs(x - x_target) < THRESHOLD_DISTANCE_2_GOAL) & (
            self.lib.abs(y - y_target) < THRESHOLD_DISTANCE_2_GOAL
        )
        done = ~(cond1 & (~cond2))
        return done

    def get_distance(self, x1, x2):
        # Distance between points x1 and x2
        return self.lib.sqrt((x1[:, 0] - x2[:, 0]) ** 2 + (x1[:, 1] - x2[:, 1]) ** 2)

    def get_heading(self, x1, x2):
        # Heading between points x1,x2 with +X axis
        return self.lib.atan2((x2[:, 1] - x1[:, 1]), (x2[:, 0] - x1[:, 0]))

    def get_closest_idx(self):
        # Get closest waypoint index from current position of car
        self.d_to_waypoints = []

        for i in range(len(self.waypoints)):
            self.d_to_waypoints.append(
                self.get_distance(
                    self.lib.unsqueeze(self.waypoints[i, :], 0), self.state
                )
            )  # Calculate distance from each of the waypoints

        self.d_to_waypoints = self.lib.stack(self.d_to_waypoints, 1)

        # Find the index to two least distance waypoints
        prev_ind, next_ind = self.lib.unstack(
            self.lib.argpartition(self.d_to_waypoints, 2), 2, 1
        )
        self.closest_idx = self.lib.max(
            prev_ind, next_ind
        )  # Next waypoint to track is higher of the two indices in the sequence of waypoints

    def step_tf(self, state: tf.Tensor, action: tf.Tensor):
        state, action = self._expand_arrays(state, action)

        # Perturb action if not in planning mode
        if self._batch_size == 1:
            action += self._generate_actuator_noise()

        state = self.update_state(
            state, action, 0.005
        )  # 0.005 Modify time discretization for getting pose from Dubin's simulation
        self.get_closest_idx()  # Get closest index

        return state

    def step(
        self, action: Union[np.ndarray, tf.Tensor]
    ) -> Tuple[
        Union[np.ndarray, tf.Tensor],
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        dict,
    ]:
        self.state, action = self._expand_arrays(self.state, action)

        # Perturb action if not in planning mode
        if self._batch_size == 1:
            action += self._generate_actuator_noise()

        info = {}

        self.state = self.update_state(
            self.state, action, 0.005
        )  # 0.005 Modify time discretization for getting pose from Dubin's simulation

        done = self.is_done(self.state)
        reward = self.get_reward(self.state, action)

        self.state = self.lib.squeeze(self.state)

        if self._batch_size == 1:
            return self.lib.to_numpy(self.state), float(reward), bool(done), {}

        return self.state, reward, done, {}

    def render(self):
        # Storing tracked trajectory
        self.traj_x.append(self.state[0] * MAX_X)
        self.traj_y.append(self.state[1] * MAX_Y)
        self.traj_yaw.append(self.state[2])

        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )
        plt.plot(
            self.traj_x * 10, self.traj_y * 10, "ob", markersize=2, label="trajectory"
        )
        # Rendering waypoint sequence
        for i in range(len(self.waypoints)):
            plt.plot(
                self.waypoints[i][0] * MAX_X,
                self.waypoints[i][1] * MAX_Y,
                "^r",
                label="waypoint",
            )
        plt.plot(self.target[0] * MAX_X, self.target[1] * MAX_Y, "xg", label="target")
        # Rendering the car and action taken
        self.plot_car()
        plt.axis("equal")
        plt.grid(True)
        plt.title("Simulation")
        plt.pause(0.0001)

    def close(self):
        # For Gym AI compatibility
        pass

    def update_state(self, state, action, DT):
        x, y, yaw_car = self.lib.unstack(state, 3, 1)
        throttle, steer = self.lib.unstack(action, 2, 1)
        # Update the pose as per Dubin's equations

        steer = self.lib.clip(steer, -MAX_STEER, MAX_STEER)
        throttle = self.lib.clip(throttle, MIN_SPEED, MAX_SPEED)

        x = x + throttle * self.lib.cos(yaw_car) * DT
        y = y + throttle * self.lib.sin(yaw_car) * DT
        yaw_car = yaw_car + throttle / WB * self.lib.tan(steer) * DT
        return self.lib.stack([x, y, yaw_car], 1)

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

        plt.plot(
            np.array(outline[0, :]).flatten(),
            np.array(outline[1, :]).flatten(),
            truckcolor,
        )
        plt.plot(
            np.array(fr_wheel[0, :]).flatten(),
            np.array(fr_wheel[1, :]).flatten(),
            truckcolor,
        )
        plt.plot(
            np.array(rr_wheel[0, :]).flatten(),
            np.array(rr_wheel[1, :]).flatten(),
            truckcolor,
        )
        plt.plot(
            np.array(fl_wheel[0, :]).flatten(),
            np.array(fl_wheel[1, :]).flatten(),
            truckcolor,
        )
        plt.plot(
            np.array(rl_wheel[0, :]).flatten(),
            np.array(rl_wheel[1, :]).flatten(),
            truckcolor,
        )
        plt.plot(x, y, "*")


# def main():

#     ### Declare variables for environment
#     start_point = [0.0, 0.0, 1.57]
#     target_point = [4.0, 8.0, 1.57]
#     waypoints = [
#         [0.0, 1.0, 1.57],
#         [0.0, 2.0, 1.57],
#         [1.0, 3.0, 1.57],
#         [2.0, 4.0, 1.57],
#         [3.0, 5.0, 1.57],
#         [4.0, 6.0, 1.57],
#         [4.0, 7.0, 1.57],
#     ]
#     n_waypoints = 1  # look ahead waypoints

#     # Instantiate Gym object
#     env = dubins_car_batched(start_point, waypoints, target_point, n_waypoints)
#     max_steps = int(1e6)

#     ## Model Training
#     agent = SAC(env.observation_space.shape[0], env.action_space, args)
#     # Memory
#     memory = ReplayMemory(args.replay_size, args.seed)
#     # Tensorboard
#     writer = SummaryWriter(
#         "runs/{}_SAC_{}_{}_{}".format(
#             datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
#             "DeepracerGym",
#             args.policy,
#             "autotune" if args.automatic_entropy_tuning else "",
#         )
#     )

#     # Training parameters
#     total_numsteps = 0
#     updates = 0
#     num_goal_reached = 0
#     for i_episode in itertools.count(1):
#         # Training Loop
#         episode_reward = 0
#         episode_steps = 0
#         done = False
#         state = env.reset()

#         while not done:
#             env.render()
#             start_time = time.time()
#             if args.start_steps > total_numsteps:
#                 action = env.action_space.sample()  # Sample random action
#             else:
#                 action = agent.select_action(state)  # Sample action from policy

#             next_state, reward, done, _ = env.step(action)  # Step
#             if (reward > 9) and (
#                 episode_steps > 1
#             ):  # Count the number of times the goal is reached
#                 num_goal_reached += 1

#             episode_steps += 1
#             total_numsteps += 1
#             episode_reward += reward
#             if episode_steps > args.max_episode_length:
#                 done = True

#             # Ignore the "done" signal if it comes from hitting the time horizon.
#             # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
#             mask = 1 if episode_steps == args.max_episode_length else float(not done)
#             # mask = float(not done)
#             memory.push(
#                 state, action, reward, next_state, mask
#             )  # Append transition to memory

#             state = next_state
#             # print(done)

#         # if i_episode % UPDATE_EVERY == 0:
#         if len(memory) > args.batch_size:
#             # Number of updates per step in environment
#             for i in range(args.updates_per_step * args.max_episode_length):
#                 # Update parameters of all the networks
#                 (
#                     critic_1_loss,
#                     critic_2_loss,
#                     policy_loss,
#                     ent_loss,
#                     alpha,
#                 ) = agent.update_parameters(memory, args.batch_size, updates)

#                 writer.add_scalar("loss/critic_1", critic_1_loss, updates)
#                 writer.add_scalar("loss/critic_2", critic_2_loss, updates)
#                 writer.add_scalar("loss/policy", policy_loss, updates)
#                 writer.add_scalar("loss/entropy_loss", ent_loss, updates)
#                 writer.add_scalar("entropy_temprature/alpha", alpha, updates)
#                 updates += 1

#         if total_numsteps > args.num_steps:
#             break

#         if episode_steps > 1:
#             writer.add_scalar("reward/train", episode_reward, i_episode)
#             writer.add_scalar("reward/episode_length", episode_steps, i_episode)
#             writer.add_scalar("reward/num_goal_reached", num_goal_reached, i_episode)

#         print(
#             "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
#                 i_episode, total_numsteps, episode_steps, round(episode_reward, 2)
#             )
#         )
#         print("Number of Goals Reached: ", num_goal_reached)

#     print("----------------------Training Ending----------------------")

#     agent.save_model("right_curve", suffix="2")  # Rename it as per training scenario
#     return True

#     ## Environment Quick Test
#     # state = env.reset()
#     # env.render()
#     # for ep in range(5):
#     # 	state = env.reset()
#     # 	env.render()
#     # 	for i in range(max_steps):
#     # 		action = [1.0, 0.]
#     # 		n_state,reward,done,info = env.step(action)
#     # 		env.render()
#     # 		if done:
#     # 			state = env.reset()
#     # 			done = False
#     # 			break


# if __name__ == "__main__":
#     main()
