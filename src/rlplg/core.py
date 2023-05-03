"""
This module defines core abstractions.
"""
import abc
import dataclasses
from typing import Any, Mapping, Optional, Union

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike

NestedArray = Union[Mapping, np.ndarray]


class StepType:
    """
    Indicates the type of step.
    """

    FIRST = 1
    MID = 2
    LAST = 3


@dataclasses.dataclass(frozen=True)
class TimeStep:
    """
    Encapsulates an observation at time t - 1,
    and the reward, discount and step type
    from time step t.
    """

    step_type: ArrayLike
    reward: float
    discount: float
    observation: NestedArray

    @classmethod
    def restart(cls, observation: NestedArray) -> "TimeStep":
        """
        Creates a time step with step type `FIRST`.
        """
        return cls(
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=observation,
        )

    @classmethod
    def transition(
        cls, observation: NestedArray, reward: float, discount: float = 1.0
    ) -> "TimeStep":
        return cls(
            step_type=StepType.MID,
            reward=reward,
            discount=discount,
            observation=observation,
        )

    @classmethod
    def termination(cls, observation: NestedArray, reward: float) -> "TimeStep":
        return cls(
            step_type=StepType.LAST,
            reward=reward,
            discount=0.0,
            observation=observation,
        )


@dataclasses.dataclass(frozen=True)
class PolicyStep:
    """
    Output of a policy's action function.
    Encapsulates the chosen action and policy state.
    """

    action: ArrayLike
    state: ArrayLike
    info: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class Trajectory:
    """
    A trajectory an example for training RL agents.
    """

    step_type: ArrayLike
    observation: NestedArray
    action: ArrayLike
    policy_info: Mapping[str, Any]
    next_step_type: ArrayLike
    reward: float
    discount: float

    @staticmethod
    def from_transition(
        time_step: TimeStep,
        action_step: PolicyStep,
        next_time_step: TimeStep,
    ) -> "Trajectory":
        """
        Builds a trajectory given a state action and transition to new
        state.
        """
        return Trajectory(
            step_type=time_step.step_type,
            observation=time_step.observation,
            action=action_step.action,
            policy_info=action_step.info,
            next_step_type=next_time_step.step_type,
            reward=next_time_step.reward,
            discount=next_time_step.discount,
        )


class PyPolicy(abc.ABC):
    """
    Base class for python policies.
    """

    def __init__(
        self,
        time_step_spec: Any,
        action_spec: Any,
        emit_log_probability: bool = False,
    ):
        self.time_step_spec = time_step_spec
        self.action_spec = action_spec
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
        time_step: TimeStep,
        policy_state: Any = (),
        seed: Optional[int] = None,
    ) -> PolicyStep:
        """Generates next action given the time_step and policy_state.


        Args:
          time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
          policy_state: An optional previous policy_state.
          seed: Seed to use when choosing action. Impl specific.

        Returns:
          A PolicyStep containing:
            `action`: A nest of action Arrays matching the `action_spec()`.
            `state`: A nest of policy states to be fed into the next call to action.
            `info`: Optional side information such as action log probabilities.
        """
        return self._action(time_step, policy_state, seed=seed)

    @abc.abstractmethod
    def _action(
        self,
        time_step: TimeStep,
        policy_state: Any,
        seed: Optional[int] = None,
    ) -> PolicyStep:
        """Implementation of `action`.

        Args:
          time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
          policy_state: An Array, or a nested dict, list or tuple of Arrays
            representing the previous policy_state.
          seed: Seed to use when choosing action. Impl specific.

        Returns:
          A `PolicyStep` named tuple containing:
            `action`: A nest of action Arrays matching the `action_spec()`.
            `state`: A nest of policy states to be fed into the next call to action.
            `info`: Optional side information such as action log probabilities.
        """


class PyEnvironment(gym.Env):
    """
    A generic environment class.
    """

    def step(self, action: Any) -> TimeStep:
        return self._step(action)

    @abc.abstractmethod
    def _step(self, action: Any) -> TimeStep:
        """ """

    def reset(self) -> TimeStep:
        return self._reset()

    @abc.abstractmethod
    def _reset(self) -> TimeStep:
        """ """

    def time_step_spec(self):
        del self
        return ()

    def action_spec(self):
        return ()
