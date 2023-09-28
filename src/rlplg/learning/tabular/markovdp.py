"""
This module has classes that pertain to markov decision processes.
"""
import abc
from typing import Any

from rlplg import envdesc


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

    @abc.abstractmethod
    def env_desc(self) -> envdesc.EnvDesc:
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
