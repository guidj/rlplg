"""
This module has classes that pertain to markov decision processes.
"""
import abc
from typing import Any

from tf_agents.typing.types import NestedArray

from rlplg import envdesc


class MDP(abc.ABC):
    """
    Markov Decision Process.
    """

    @abc.abstractmethod
    def transition_probability(
        self, state: NestedArray, action: NestedArray, next_state: NestedArray
    ) -> float:
        """
        Given a state s, action a, and next state s' returns a transition probability.
        Args:
            state: starting state
            action: agent's action
            next_state: state transition into after taking the action.

        Returns:
            A transition probability.
        """
        pass

    @abc.abstractmethod
    def reward(
        self, state: NestedArray, action: NestedArray, next_state: NestedArray
    ) -> float:
        """
        Given a state s, action a, and next state s' returns the expected reward.

        Args:
            state: starting state
            action: agent's action
            next_state: state transition into after taking the action.
        Returns
            A transition probability.
        """
        pass

    @abc.abstractmethod
    def env_desc(self) -> envdesc.EnvDesc:
        """
        Returns:
            An instance of EnvDesc with properties of the environment.
        """
        pass


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
        pass

    @abc.abstractmethod
    def action(self, action: Any) -> int:
        """
        Maps an agent action to an action ID.
        """
        pass
