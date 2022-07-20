from typing import Callable, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
from gym.envs.registration import register
from numpy.random import SFC64, Generator
from Utilities.utils import get_logger

log = get_logger(__name__)

ENV_REGISTRY = {
    "CustomEnvironments/CartPoleContinuous": "Environments.continuous_cartpole_batched:continuous_cartpole_batched",
    "CustomEnvironments/MountainCarContinuous": "Environments.continuous_mountaincar_batched:continuous_mountaincar_batched",
    "CustomEnvironments/Pendulum": "Environments.pendulum_batched:pendulum_batched",
}


def register_envs():
    for identifier, entry_point in ENV_REGISTRY.items():
        register(
            id=identifier,
            entry_point=entry_point,
            max_episode_steps=None,
        )


TensorType = Union[np.ndarray, tf.Tensor, torch.Tensor]
RandomGeneratorType = Union[Generator, tf.random.Generator, torch.Generator]


class ComputationLibrary:
    reshape: Callable[[TensorType, "tuple[int]"], TensorType] = None
    shape: Callable[[TensorType], "list[int]"] = None
    to_numpy: Callable[[TensorType], np.ndarray] = None
    to_tensor: Callable[[TensorType, float], TensorType] = None
    unstack: Callable[[TensorType, int, int], "list[TensorType]"] = None
    ndim: Callable[[TensorType], int] = None
    clip: Callable[[TensorType, float, float], TensorType] = None
    sin: Callable[[TensorType], TensorType] = None
    cos: Callable[[TensorType], TensorType] = None
    squeeze: Callable[[TensorType], TensorType] = None
    unsqueeze: Callable[[TensorType, int], TensorType] = None
    stack: Callable[["list[TensorType]", int], TensorType] = None
    cast: Callable[[TensorType, float], TensorType] = None
    float32: Callable[[], float] = None
    tile: Callable[[TensorType, "tuple[int]"], TensorType] = None
    zeros: Callable[["tuple[int]"], TensorType] = None
    create_rng: Callable[[int], RandomGeneratorType] = None
    standard_normal: Callable[[RandomGeneratorType, "tuple[int]"], TensorType] = None
    uniform: Callable[
        [RandomGeneratorType, "tuple[int]", TensorType, TensorType, float], TensorType
    ] = None
    sum: Callable[[TensorType, int], TensorType] = None
    set_shape: Callable[[TensorType, "list[int]"], None] = None
    concat: Callable[["list[TensorType]", int], TensorType]


class NumpyLibrary(ComputationLibrary):
    reshape = lambda x, shape: np.reshape(x, shape)
    shape = np.shape
    to_numpy = lambda x: np.array(x)
    to_tensor = lambda x, t: np.array(x, dtype=t)
    unstack = lambda x, num, axis: list(np.moveaxis(x, axis, 0))
    ndim = np.ndim
    clip = np.clip
    sin = np.sin
    cos = np.cos
    squeeze = np.squeeze
    unsqueeze = np.expand_dims
    stack = np.stack
    cast = lambda x, t: x.astype(t)
    float32 = np.float32
    tile = np.tile
    zeros = np.zeros
    create_rng = lambda seed: Generator(SFC64(seed))
    standard_normal = lambda generator, shape: generator.standard_normal(size=shape)
    uniform = lambda generator, shape, low, high, dtype: generator.uniform(
        low=low, high=high, size=shape
    ).astype(dtype)
    sum = lambda x, a: np.sum(x, axis=a, keepdims=False)
    set_shape = lambda x, shape: x
    concat = lambda x, a: np.concatenate(x, axis=a)


class TensorFlowLibrary(ComputationLibrary):
    reshape = tf.reshape
    shape = lambda x: x.get_shape()#.as_list()
    to_numpy = lambda x: x.numpy()
    to_tensor = lambda x, t: tf.convert_to_tensor(x, dtype=t)
    unstack = lambda x, num, axis: tf.unstack(x, num=num, axis=axis)
    ndim = tf.rank
    clip = tf.clip_by_value
    sin = tf.sin
    cos = tf.cos
    squeeze = tf.squeeze
    unsqueeze = tf.expand_dims
    stack = tf.stack
    cast = lambda x, t: tf.cast(x, dtype=t)
    float32 = tf.float32
    tile = tf.tile
    zeros = tf.zeros
    create_rng = lambda seed: tf.random.Generator.from_seed(seed)
    standard_normal = lambda generator, shape: generator.normal(shape)
    uniform = lambda generator, shape, low, high, dtype: generator.uniform(
        shape, minval=low, maxval=high, dtype=dtype
    )
    sum = lambda x, a: tf.reduce_sum(x, axis=a, keepdims=False)
    set_shape = lambda x, shape: x.set_shape(shape)
    concat = lambda x, a: tf.concat(x, a)


class PyTorchLibrary(ComputationLibrary):
    reshape = torch.reshape
    shape = lambda x: list(x.size())
    to_numpy = lambda x: x.cpu().detach().numpy()
    to_tensor = lambda x, t: torch.as_tensor(x, dtype=t)
    unstack = lambda x, num, dim: torch.unbind(x, dim=dim)
    ndim = lambda x: x.ndim
    clip = torch.clamp
    sin = torch.sin
    cos = torch.cos
    squeeze = torch.squeeze
    unsqueeze = torch.unsqueeze
    stack = torch.stack
    cast = lambda x, t: x.type(t)
    float32 = torch.float32
    tile = torch.tile
    zeros = torch.zeros
    create_rng = lambda seed: torch.Generator().manual_seed(seed)
    standard_normal = lambda generator, shape: torch.normal(
        torch.zeros(shape), 1.0, generator=generator
    )
    uniform = (
        lambda generator, shape, low, high, dtype: (high - low)
        * torch.rand(*shape, generator=generator, dtype=dtype)
        + low
    )
    sum = lambda x, a: torch.sum(x, a, keepdim=False)
    set_shape = lambda x, shape: x
    concat = lambda x, a: torch.concat(x, dim=a)


class EnvironmentBatched:
    def step(
        self, action: Union[np.ndarray, tf.Tensor, torch.Tensor]
    ) -> Tuple[
        Union[np.ndarray, tf.Tensor, torch.Tensor],
        Union[np.ndarray, float],
        Union[np.ndarray, bool],
        dict,
    ]:
        return NotImplementedError()

    def reset(
        self,
        state: np.ndarray = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        return NotImplementedError()

    def _set_up_rng(self, seed: int = None) -> None:
        if seed is None:
            seed = 0
            log.warn(f"Environment set up with no seed specified. Setting to {seed}.")

        self.rng = self.lib.create_rng(seed)

    def is_done(self, state):
        return NotImplementedError()

    def get_reward(self, state, action):
        return NotImplementedError()

    def _generate_actuator_noise(self):
        return (
            self._actuator_noise
            * (self.action_space.high - self.action_space.low)
            * self.lib.standard_normal(
                self.rng, (self._batch_size, len(self._actuator_noise))
            )
        )

    def _expand_arrays(
        self,
        state: Union[np.ndarray, tf.Tensor, torch.Tensor],
        action: Union[np.ndarray, tf.Tensor, torch.Tensor],
    ):
        if self.lib.ndim(action) < 2:
            action = self.lib.reshape(
                action, (self._batch_size, sum(self.action_space.shape))
            )
        if self.lib.ndim(state) < 2:
            state = self.lib.reshape(
                state, (self._batch_size, sum(self.observation_space.shape))
            )
        return state, action

    def _get_reset_return_val(self, return_info: bool = False):
        if self._batch_size == 1:
            self.state = self.lib.to_numpy(self.lib.squeeze(self.state))

        if return_info:
            return tuple((self.state, {}))
        return self.state

    def set_computation_library(self, computation_lib: ComputationLibrary):
        try:
            self.lib = computation_lib
        except KeyError as error:
            log.exception(error)

    # Overloading properties/methods for Bharadhwaj implementation
    @property
    def a_size(self):
        return self.action_space.shape[0]

    def reset_state(self, batch_size):
        self.reset()

    @property
    def B(self):
        return self._batch_size

    def rollout(self, actions, return_traj=False):
        # Uncoditional action sequence rollout
        # actions: shape: TxBxA (time, batch, action)
        assert actions.dim() == 3
        assert actions.size(1) == self.B, "{}, {}".format(actions.size(1), self.B)
        assert actions.size(2) == self.a_size
        T = actions.size(0)
        rs = []
        ss = []

        total_r = torch.zeros(self.B, requires_grad=True, device=actions.device)
        for i in range(T):
            # Reshape for step function: BxTxA
            s, r, _, _ = self.step(actions[i])
            rs.append(r)
            ss.append(s)
            total_r = total_r + r
            # if(done):
            #     break
        if return_traj:
            return rs, ss
        else:
            return total_r


class cost_functions:
    def __init__(self, env: EnvironmentBatched) -> None:
        self.env = env
    
    def get_terminal_cost(self, s_hor):
        return 0.0
    
    def get_stage_cost(self, s, u, u_prev):
        return -self.env.get_reward(s, u)

    def get_trajectory_cost(self, s_hor, u, u_prev=None):
        total_cost = 0
        for horizon_step in range(self.env.lib.shape(u)[1]):
            total_cost += self.get_stage_cost(s_hor[:, horizon_step, :], u[:, horizon_step, :], None)
        total_cost = total_cost + self.get_terminal_cost(s_hor)
        return total_cost
