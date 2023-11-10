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
class TrajectoryStep:
    """
    A trajectory step for training RL agents.
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
    ) -> "TrajectoryStep":
        """
        Builds a trajectory step given a state and action.
        """
        obs, _, terminated, truncated, _ = time_step
        _, next_reward, _, _, _ = next_time_step

        return TrajectoryStep(
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

    @abc.abstractmethod
    def get_initial_state(self, batch_size: Optional[int] = None) -> Any:
        """Returns an initial state usable by the policy.

        Args:
          batch_size: An optional batch size.

        Returns:
          An initial policy state.
        """

    @abc.abstractmethod
    def action(
        self,
        observation: ObsType,
        policy_state: Any = (),
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


@dataclasses.dataclass(frozen=True)
class EnvDesc:
    """
    Class contains properties of the environment.
    """

    num_states: int
    num_actions: int


class Mdp:
    """
    Markov Decision Process.
    """

    @abc.abstractmethod
    def transition_probability(self, state: int, action: int, next_state: int) -> float:
        """
        Given a state s, action a, and next state s' returns a transition probability.
        Args:
            state: starting state
            action: agent's action
            next_state: state transition into after taking the action.

        Returns:
            A transition probability.
        """

    @abc.abstractmethod
    def reward(self, state: int, action: int, next_state: int) -> float:
        """
        Given a state s, action a, and next state s' returns the expected reward.

        Args:
            state: starting state
            action: agent's action
            next_state: state transition into after taking the action.
        Returns
            A transition probability.
        """

    @property
    @abc.abstractmethod
    def env_desc(self) -> EnvDesc:
        """
        Returns:
            An instance of EnvDesc with properties of the environment.
        """


class MdpDiscretizer:
    """
    Class contains functions to map state and action spaces
    to discrete range values.
    """

    @abc.abstractmethod
    def state(self, observation: Any) -> int:
        """
        Maps an observation to a state ID.
        """

    @abc.abstractmethod
    def action(self, action: Any) -> int:
        """
        Maps an agent action to an action ID.
        """


@dataclasses.dataclass(frozen=True)
class EnvSpec:
    """
    Class holds environment variables.
    """

    name: str
    level: str
    environment: gym.Env
    discretizer: MdpDiscretizer
    env_desc: EnvDesc
