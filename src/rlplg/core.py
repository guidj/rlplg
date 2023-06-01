"""
This module defines core abstractions.
"""
import abc
import dataclasses
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame, SupportsFloat
from numpy.typing import ArrayLike

NestedArray = Union[Mapping, np.ndarray]
TimeStep = Tuple[ObsType, SupportsFloat, bool, bool, Mapping[str, Any]]
InitState = Tuple[ObsType, Mapping[str, Any]]
RenderType = Optional[Union[RenderFrame, Sequence[RenderFrame]]]


@dataclasses.dataclass(frozen=True)
class PolicyStep:
    """
    Output of a policy's action function.
    Encapsulates the chosen action and policy state.
    """

    action: ArrayLike
    state: Any
    info: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class Trajectory:
    """
    A trajectory an example for training RL agents.
    """

    observation: ObsType
    action: ActType
    policy_info: Mapping[str, Any]
    terminated: bool
    truncated: bool
    reward: float

    @staticmethod
    def from_transition(
        time_step: TimeStep,
        action_step: PolicyStep,
        next_time_step: TimeStep,
    ) -> "Trajectory":
        """
        Builds a trajectory given a state and action.
        """
        obs, _, terminated, truncated, _ = time_step
        _, next_reward, _, _, _ = next_time_step

        return Trajectory(
            observation=obs,
            action=action_step.action,
            policy_info=action_step.info,
            terminated=terminated,
            truncated=truncated,
            reward=next_reward,
        )


class PyPolicy(abc.ABC):
    """
    Base class for python policies.
    """

    def __init__(
        self,
        emit_log_probability: bool = False,
    ):
        self.emit_log_probability = emit_log_probability

    def get_initial_state(self, batch_size: Optional[int] = None) -> Any:
        """Returns an initial state usable by the policy.

        Args:
          batch_size: An optional batch size.

        Returns:
          An initial policy state.
        """
        return self._get_initial_state(batch_size)

    @abc.abstractmethod
    def _get_initial_state(self, batch_size: Optional[int]) -> Any:
        """Default implementation of `get_initial_state`.

        Args:
          batch_size: The batch shape.

        Returns:
          An object of type `policy_state` containing properly
          initialized Arrays.
        """

    def action(
        self,
        observation: ObsType,
        policy_state: Any = (),
        seed: Optional[int] = None,
    ) -> PolicyStep:
        """Generates next action given the time_step and policy_state.


        Args:
          observation: An observation.
          policy_state: An optional previous policy state.
          seed: Seed to use when choosing action. Impl specific.

        Returns:
          A PolicyStep containing:
            `action`: The policy's chosen action.
            `state`: A policy state to be fed into the next call to action.
            `info`: Optional side information such as action log probabilities.
        """
        return self._action(observation, policy_state, seed=seed)

    @abc.abstractmethod
    def _action(
        self,
        observation: ObsType,
        policy_state: Any,
        seed: Optional[int] = None,
    ) -> PolicyStep:
        """Implementation of `action`.

        Args:
          observation: An observation.
          policy_state: An Array, or a nested dict, list or tuple of Arrays
            representing the previous policy state.
          seed: Seed to use when choosing action. Impl specific.

        Returns:
          A `PolicyStep` named tuple containing:
            `action`: The policy's chosen action.
            `state`: A policy state to be fed into the next call to action.
            `info`: Optional side information such as action log probabilities.
        """


class PyEnvironment(gym.Env):
    """
    A generic environment class.
    It exists to introduce pattern of overriding
    using private methods.
    """

    def step(self, action: ActType) -> TimeStep:
        """
        Takes an action and advances the environment
        to the next state.
        """
        return self._step(action)

    @abc.abstractmethod
    def _step(self, action: ActType) -> TimeStep:
        """
        Override this method to define `step`.
        """
        raise NotImplementedError

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None
    ) -> InitState:
        """
        Resets the environment to a starting state.
        TODO: support seed and options.
        """
        del seed
        del options
        return self._reset()

    @abc.abstractmethod
    def _reset(self) -> InitState:
        """
        Override this method to define `reset`.
        """
        raise NotImplementedError

    def render(self) -> RenderType:
        """
        Returns frames to render.
        These can be basic data types, e.g. string.
        """
        return self._render()

    @abc.abstractmethod
    def _render(self) -> RenderType:
        """
        Override this method to define `render`.
        """
        raise NotImplementedError
