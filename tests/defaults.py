import copy
from typing import Any, Optional, Sequence

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.policies import py_policy
from tf_agents.specs import array_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing.types import NestedArray, NestedArraySpec, Seed

from rlplg import envdesc
from rlplg.learning.tabular import markovdp

GRID_WIDTH = 5
GRID_HEIGHT = 5
NUM_ACTIONS = 4
NUM_STATES = GRID_HEIGHT * GRID_WIDTH

CLIFF_COLOR = (25, 50, 75)
PATH_COLOR = (50, 75, 25)
ACTOR_COLOR = (75, 25, 50)
EXIT_COLOR = (255, 204, 0)


class BasePyEnv(py_environment.PyEnvironment):
    """
    A base class for test environments.
    """

    def __init__(self, action_spec: NestedArraySpec, observation_spec: NestedArraySpec):
        self._action_spec = action_spec
        self._observation_spec = observation_spec

    def observation_spec(self) -> NestedArraySpec:
        return self._observation_spec

    def action_spec(self) -> NestedArraySpec:
        return self._action_spec


class CountEnv(BasePyEnv):
    """
    Choose between moving forward or stopping, until we reach 3, starting from zero.
        - States: 0, 1, 2, 3 (terminal)
        - Actions: do nothing, next

    If none: value = value, R = -10
    If next: value + 1, R = -1
    If value == 3, action = next - game over, R = 0.

    Q-Table:
      none next
    0:  -10  -1
    1:  -10  -1
    2:  -10  -1
    3:   0   0
    """

    MAX_VALUE = 3
    ACTION_NOTHING = 0
    ACTION_NEXT = 1
    WRONG_MOVE_REWARD = -10.0
    RIGHT_MOVE_REWARD = -1.0

    def __init__(self):
        super().__init__(
            action_spec=array_spec.BoundedArraySpec(
                shape=(),
                dtype=np.int64,
                minimum=0,
                maximum=1,
                name="action",
            ),
            observation_spec=array_spec.BoundedArraySpec(
                shape=(),
                dtype=np.int64,
                minimum=0,
                name="observation",
            ),
        )
        # env specific
        self._observation: Optional[np.ndarray] = None
        self._seed = None

    def _step(self, action: NestedArray) -> ts.TimeStep:
        """Updates the environment according to action and returns a `TimeStep`.

        See `step(self, action)` docstring for more details.

        Args:
        action: A NumPy array, or a nested dict, list or tuple of arrays
            corresponding to `action_spec()`.
        """

        if self._observation is None:
            raise RuntimeError(
                f"{type(self).__name__} environment needs to be reset. Call the `reset` method."
            )

        if action == self.ACTION_NOTHING:
            new_obs = copy.deepcopy(self._observation)
            reward = self.WRONG_MOVE_REWARD
        elif action == self.ACTION_NEXT:
            new_obs = np.array(
                np.min([self._observation + 1, self.MAX_VALUE]), np.int64
            )
            reward = self.RIGHT_MOVE_REWARD
        else:
            raise ValueError(f"Unknown action {action}")

        # terminal state reward override
        if self._observation == self.MAX_VALUE:
            reward = 0.0

        self._observation = new_obs
        finished = np.array_equal(new_obs, self.MAX_VALUE)
        if finished:
            return ts.termination(
                observation=copy.deepcopy(self._observation), reward=reward
            )
        return ts.transition(
            observation=copy.deepcopy(self._observation), reward=reward
        )

    def _reset(self) -> ts.TimeStep:
        """Starts a new sequence, returns the first `TimeStep` of this sequence.

        See `reset(self)` docstring for more details
        """
        self._observation = np.array(0, np.int64)
        return ts.restart(observation=self._observation)


class CountEnvMDP(markovdp.MDP):
    """
    Markov decision process definition for CountEnv.
    """

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
        # terminal state
        if (
            (state == CountEnv.MAX_VALUE and np.array_equal(next_state, state))
            or (
                action == CountEnv.ACTION_NEXT and np.array_equal(next_state, state + 1)
            )
            or (action == CountEnv.ACTION_NOTHING and np.array_equal(next_state, state))
        ):
            return 1.0
        return 0.0

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
        if state == CountEnv.MAX_VALUE:
            return 0.0
        elif action == CountEnv.ACTION_NOTHING:
            return CountEnv.WRONG_MOVE_REWARD
        return CountEnv.RIGHT_MOVE_REWARD

    def env_desc(self) -> envdesc.EnvDesc:
        """
        Returns:
            An instance of EnvDesc with properties of the environment.
        """
        return envdesc.EnvDesc(num_states=4, num_actions=2)


class SingleStateEnv(BasePyEnv):
    """
    An environment that remains in a perpetual state.
    """

    def __init__(self, num_actions: int):
        assert num_actions > 0, "`num_actios` must be positive."
        self.num_actions = num_actions
        super().__init__(
            action_spec=array_spec.BoundedArraySpec(
                shape=(),
                dtype=np.int64,
                minimum=0,
                maximum=num_actions,
                name="action",
            ),
            observation_spec=array_spec.BoundedArraySpec(
                shape=(),
                dtype=np.int64,
                minimum=0,
                maximum=0,
                name="observation",
            ),
        )

        # env specific
        self._observation: Optional[np.ndarray] = None
        self._seed = None

    def _step(self, action: NestedArray) -> ts.TimeStep:
        """Updates the environment according to action and returns a `TimeStep`.

        See `step(self, action)` docstring for more details.

        Args:
        action: A NumPy array, or a nested dict, list or tuple of arrays
            corresponding to `action_spec()`.
        """

        # none
        if not (0 <= action < self.num_actions):
            raise ValueError(f"Unknown action {action}")
        return ts.transition(observation=copy.deepcopy(self._observation), reward=0.0)

    def _reset(self) -> ts.TimeStep:
        """Starts a new sequence, returns the first `TimeStep` of this sequence.

        See `reset(self)` docstring for more details
        """
        self._observation = np.array(0, np.int64)
        return ts.restart(observation=self._observation)


class RoundRobinActionsPolicy(py_policy.PyPolicy):
    """
    Chooses a sequence of actions provided in the constructor, forever.
    """

    def __init__(
        self,
        time_step_spec: ts.TimeStep,
        action_spec: NestedArraySpec,
        actions: Sequence[Any],
    ):
        super().__init__(time_step_spec, action_spec)
        self._counter = 0
        self._actions = actions
        self._iterator = iter(actions)
        self.emit_log_probability = True

    def _action(
        self,
        time_step: ts.TimeStep,
        policy_state: NestedArray,
        seed: Optional[Seed] = None,
    ) -> policy_step.PolicyStep:
        """
        Takes the current time step (which includes the environment feedback)
        """
        del time_step, policy_state, seed
        state, info = (), policy_step.PolicyInfo(log_probability=np.math.log(0.5))

        try:
            action = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._actions)
            action = next(self._iterator)

        return policy_step.PolicyStep(np.array(action, dtype=np.int64), state, info)


def identity(value: Any) -> Any:
    """
    Returns the input.
    """
    return value


def item(value: np.ndarray) -> Any:
    """
    Returns the basic value in an array.
    """
    try:
        return value.item()
    except AttributeError:
        pass
    return value


def batch(*args: Any):
    """
    Collects a sequence of values into an np.ndarray.
    """
    # We use int32 and float32 for all examples/tests
    sample = next(iter(args))
    if isinstance(sample, float):
        return np.array(args, dtype=np.float32)
    if isinstance(sample, int):
        return np.array(args, dtype=np.int64)
    return np.array(args)


def policy_info(log_probability: float):
    """
    Creates a policy_step.PolicyInfo instance from a given log_probability.
    """
    return policy_step.PolicyInfo(log_probability=log_probability)


def log_prob_policy_info_spec() -> array_spec.BoundedArraySpec:
    """
    Creates policy_step.PolicyInfo spec.
    """
    return policy_step.PolicyInfo(
        log_probability=array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.float32,
            maximum=0,
            minimum=-float("inf"),
            name="log_probability",
        )
    )
