import abc
import copy
import dataclasses
from typing import Any, Callable, Optional

import numpy as np

from rlplg import core
from rlplg.core import ObsType


class PyRandomPolicy(core.PyPolicy):
    """
    A policy that chooses actions with equal probability.
    """

    def __init__(
        self,
        num_actions: int,
        emit_log_probability: bool = False,
    ):
        super().__init__(
            emit_log_probability=emit_log_probability,
        )
        self._num_actions = num_actions
        self._arms = np.arange(start=0, stop=num_actions)
        self._uniform_chance = np.array(1.0) / np.array(num_actions, dtype=np.float32)

    def get_initial_state(self, batch_size: Optional[int] = None) -> Any:
        del batch_size
        return ()

    def action(
        self,
        observation: ObsType,
        policy_state: Any = (),
        seed: Optional[int] = None,
    ) -> core.PolicyStep:
        del observation
        if seed is not None:
            raise NotImplementedError(f"Seed is not supported; but got seed: {seed}")
        action = np.random.choice(self._arms)
        if self.emit_log_probability:
            policy_info = {
                "log_probability": np.array(
                    np.log(self._uniform_chance), dtype=np.float32
                )
            }
        else:
            policy_info = {}

        return core.PolicyStep(
            action=action,
            state=policy_state,
            info=policy_info,
        )


class PyQGreedyPolicy(core.PyPolicy):
    """
    A Q policy for tabular problems, i.e. finite states and finite actions.
    """

    def __init__(
        self,
        state_id_fn: Callable[[Any], int],
        action_values: np.ndarray,
        emit_log_probability: bool = False,
    ):
        """
        The following initializes to base class defaults:
            - policy_state_spec: Any = (),
            - info_spec: Any = (),
            - observation_and_action_constraint_splitter: Optional[types.Splitter] = None
        """

        super().__init__(
            emit_log_probability=emit_log_probability,
        )

        self._state_id_fn = state_id_fn
        self._state_action_value_table = copy.deepcopy(action_values)

    def get_initial_state(self, batch_size: Optional[int] = None) -> Any:
        del batch_size
        return ()

    def action(
        self,
        observation: ObsType,
        policy_state: Any = (),
        seed: Optional[int] = None,
    ) -> core.PolicyStep:
        if seed is not None:
            raise NotImplementedError(f"Seed is not supported; but got seed: {seed}")

        state_id = self._state_id_fn(observation)
        action = np.argmax(self._state_action_value_table[state_id])
        if self.emit_log_probability:
            # the best arm has 1.0 probability of being chosen
            policy_info = {"log_probability": np.array(np.log(1.0), dtype=np.float32)}
        else:
            policy_info = {}

        return core.PolicyStep(
            action=action,
            state=policy_state,
            info=policy_info,
        )


class PyEpsilonGreedyPolicy(core.PyPolicy):
    """
    A e-greedy, which randomly chooses actions with e probability,
    and the chooses teh best action otherwise.
    """

    def __init__(
        self,
        policy: PyQGreedyPolicy,
        num_actions: int,
        epsilon: float,
        emit_log_probability: bool = False,
    ):
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError(f"Epsilon must be between [0, 1]: {epsilon}")
        if emit_log_probability and not hasattr(policy, "emit_log_probability"):
            raise ValueError("Policy has no property `emit_log_probability`")
        if emit_log_probability != getattr(policy, "emit_log_probability"):
            raise ValueError(
                f"""emit_log_probability differs between given policy and constructor argument:
                policy.emit_log_probability={getattr(policy, 'emit_log_probability')},
                emit_log_probability={emit_log_probability}""",
            )

        super().__init__(
            emit_log_probability=emit_log_probability,
        )

        self._num_actions = num_actions
        self.exploit_policy = policy
        self.explore_policy = PyRandomPolicy(
            num_actions=num_actions,
            emit_log_probability=emit_log_probability,
        )
        self.epsilon = epsilon

    def get_initial_state(self, batch_size: Optional[int] = None) -> Any:
        del batch_size
        return ()

    def action(
        self,
        observation: ObsType,
        policy_state: Any = (),
        seed: Optional[int] = None,
    ) -> core.PolicyStep:
        if seed is not None:
            raise NotImplementedError(f"Seed is not supported; but got seed: {seed}")
        # greedy move, find out the greedy arm
        explore = np.random.rand() <= self.epsilon
        policy_: core.PyPolicy = self.explore_policy if explore else self.exploit_policy
        prob = (
            self.epsilon / self._num_actions
            if explore
            else self.epsilon / self._num_actions + (1.0 - self.epsilon)
        )
        policy_step_ = policy_.action(observation, policy_state)
        # Update log-prob in _policy_step
        if self.emit_log_probability:
            policy_info = {
                "log_probability": np.array(
                    np.log(prob),
                    np.float32,
                )
            }
            return dataclasses.replace(policy_step_, info=policy_info)

        return policy_step_


class ObservablePolicy(abc.ABC):
    """
    An interface for policies that can emit the probability for state action pair.
    """

    @abc.abstractmethod
    def action_probability(self, state: Any, action: Any):
        """
        Given a state and action, it returns a probability
        choosing the action in that state.
        """


class PyObservableRandomPolicy(ObservablePolicy):
    """
    Implements an observable random policy.
    """

    def __init__(
        self,
        num_actions: int,
        emit_log_probability: bool = False,
    ):
        self._policy = PyRandomPolicy(
            num_actions=num_actions,
            emit_log_probability=emit_log_probability,
        )
        prob = 1.0 / num_actions
        self._probs = np.ones(shape=num_actions) * prob

    def action(
        self,
        observation: ObsType,
        policy_state: Any = (),
        seed: Optional[int] = None,
    ) -> core.PolicyStep:
        return self._policy.action(
            observation=observation, policy_state=policy_state, seed=seed
        )

    def action_probability(self, state, action):
        del state
        return self._probs[action]
