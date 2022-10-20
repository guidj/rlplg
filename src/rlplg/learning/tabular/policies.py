import abc
import copy
from typing import Callable, Optional

import numpy as np
from tf_agents.policies import py_policy
from tf_agents.specs import array_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing.types import NestedArray, NestedArraySpec, Seed


class PyPolicy(py_policy.PyPolicy):
    """
    Base class for python policies.
    Adds support for `emit_log_probability` to `py_policy.PyPolicy`.
    """

    def __init__(
        self,
        time_step_spec: ts.TimeStep,
        action_spec: NestedArraySpec,
        emit_log_probability: bool = False,
    ):
        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            info_spec=policy_step.PolicyInfo(
                log_probability=array_spec.BoundedArraySpec(
                    shape=(),
                    dtype=np.float32,
                    maximum=0,
                    minimum=-float("inf"),
                    name="log_probability",
                )
            )
            if emit_log_probability
            else (),
        )
        # Unsupported by parent class
        self.emit_log_probability = emit_log_probability


class PyRandomPolicy(PyPolicy):
    """
    A policy that chooses actions with equal probability.
    """

    def __init__(
        self,
        time_step_spec: ts.TimeStep,
        action_spec: NestedArraySpec,
        num_actions: int,
        emit_log_probability: bool = False,
    ):
        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            emit_log_probability=emit_log_probability,
        )
        self._num_actions = num_actions
        self._arms = np.arange(start=0, stop=num_actions, dtype=self.action_spec.dtype)
        self._uniform_chance = np.array(1.0) / np.array(num_actions, dtype=np.float32)

    def _action(
        self,
        time_step: ts.TimeStep,
        policy_state: NestedArray = (),
        seed: Optional[Seed] = None,
    ) -> policy_step.PolicyStep:
        del time_step
        if seed is not None:
            raise NotImplementedError(f"Seed is not supported; but got seed: {seed}")
        action = np.random.choice(self._arms)
        if self.emit_log_probability:
            policy_info = policy_step.set_log_probability(
                info=(),
                log_probability=np.array(
                    np.math.log(self._uniform_chance), dtype=np.float32
                ),
            )
        else:
            policy_info = ()

        return policy_step.PolicyStep(
            action=action.astype(self.action_spec.dtype),
            state=policy_state,
            info=policy_info,
        )


class PyQGreedyPolicy(PyPolicy):
    """
    A Q policy for tabular problems, i.e. finite states and finite actions.
    """

    def __init__(
        self,
        time_step_spec: ts.TimeStep,
        action_spec: NestedArraySpec,
        state_id_fn: Callable[[NestedArray], int],
        action_values: np.ndarray,
        emit_log_probability: bool = False,
    ):
        """
        The following initializes to base class defaults:
            - policy_state_spec: types.NestedArraySpec = (),
            - info_spec: types.NestedArraySpec = (),
            - observation_and_action_constraint_splitter: Optional[types.Splitter] = None
        """

        if action_spec.shape != ():
            raise ValueError(
                f"Action spec must have shape (). Received {action_spec.shape } instead."
            )
        super().__init__(
            time_step_spec,
            action_spec,
            emit_log_probability=emit_log_probability,
        )

        self._state_id_fn = state_id_fn
        self._state_action_value_table = copy.deepcopy(action_values)

    def _action(
        self,
        time_step: ts.TimeStep,
        policy_state: NestedArray,
        seed: Optional[Seed] = None,
    ) -> policy_step.PolicyStep:
        if seed is not None:
            raise NotImplementedError(f"Seed is not supported; but got seed: {seed}")

        state_id = self._state_id_fn(time_step.observation)
        action = np.argmax(self._state_action_value_table[state_id])
        if self.emit_log_probability:
            # the best arm has 1.0 probability of being chosen
            policy_info = policy_step.set_log_probability(
                info=(),
                log_probability=np.array(np.math.log(1.0), dtype=np.float32),
            )
        else:
            policy_info = ()

        return policy_step.PolicyStep(
            action=action.astype(self.action_spec.dtype),
            state=policy_state,
            info=policy_info,
        )


class PyEpsilonGreedyPolicy(PyPolicy):
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
            time_step_spec=policy.time_step_spec,
            action_spec=policy.action_spec,
            emit_log_probability=emit_log_probability,
        )

        self._num_actions = num_actions
        self.exploit_policy = policy
        self.explore_policy = PyRandomPolicy(
            time_step_spec=policy.time_step_spec,
            action_spec=policy.action_spec,
            num_actions=num_actions,
            emit_log_probability=emit_log_probability,
        )
        self.epsilon = epsilon

    def _action(
        self,
        time_step: ts.TimeStep,
        policy_state: NestedArray = (),
        seed: Optional[Seed] = None,
    ) -> policy_step.PolicyStep:
        if seed is not None:
            raise NotImplementedError(f"Seed is not supported; but got seed: {seed}")
        # greedy move, find out the greedy arm
        if np.random.rand() <= self.epsilon:
            _policy = self.explore_policy
            prob = self.epsilon / self._num_actions
        else:
            _policy = self.exploit_policy
            prob = self.epsilon / self._num_actions + (1.0 - self.epsilon)

        _policy_step = _policy.action(time_step, policy_state)

        # Update log-prob in _policy_step
        if self.emit_log_probability:
            policy_info = policy_step.set_log_probability(
                info=(),
                log_probability=np.array(
                    np.math.log(prob),
                    np.float32,
                ),
            )
            return _policy_step.replace(info=policy_info)

        return _policy_step


class ObservablePolicy(abc.ABC):
    """
    An interface for policies that can emit the probability for state action pair.
    """

    @abc.abstractmethod
    def action_probability(self, state: NestedArray, action: NestedArray):
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
        time_step_spec: ts.TimeStep,
        action_spec: NestedArraySpec,
        num_actions: int,
        emit_log_probability: bool = False,
    ):
        self._policy = PyRandomPolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            num_actions=num_actions,
            emit_log_probability=emit_log_probability,
        )
        prob = 1.0 / num_actions
        self._probs = np.ones(shape=num_actions) * prob

    def _action(
        self,
        time_step: ts.TimeStep,
        policy_state: NestedArray = (),
        seed: Optional[Seed] = None,
    ) -> policy_step.PolicyStep:
        return self._policy.action(
            time_step=time_step, policy_state=policy_state, seed=seed
        )

    def action_probability(self, state, action):
        del state
        return self._probs[action]
