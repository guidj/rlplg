"""
This module contains the definition of the State-Random-Walk problem.

From Barto, Sutton, p.125

"...all episodes start in the center state, C,then proceed eitherleft or right by one state on each step, with equal probability.
Episodes terminate either on the extreme left or the extreme right.
When an episode terminates on the right, a reward of +1 occurs; all other rewards are zero."

A distiction is that we introduce agent actions - going left of right.
Combined with a random policy, it should produce the same effect.
"""

import base64
import copy
import hashlib
import math
from typing import Any, Mapping, Optional, Tuple

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing.types import NestedArray, NestedArraySpec, Seed

from rlplg import envdesc, envspec, npsci
from rlplg.environments.randomwalk import constants
from rlplg.learning.tabular import markovdp


class StateRandomWalk(py_environment.PyEnvironment):
    """
    An environment where an agent is meant to right, until the end.
    The environment can terminate on the last left or right states.
    By default, only the right direction yields a positive reward.

    Example:
    Steps: 5
    The starting state is the middle state, i.e. ceil(steps / 2) (without shifting by zero-index, since zero is a terminal state)
    Starting state = ceil(5 / 2) = 3
    Terminal states: 0 and 6.
    """

    metadata = {"render.modes": ["raw"]}

    def __init__(
        self,
        steps: int,
        left_end_reward: float = constants.LEFT_REWARD,
        right_end_reward: float = constants.RIGHT_REWARD,
        step_reward: float = constants.STEP_REWARD,
    ):
        """
        Args:
            steps: number of states. The left and right ends are terminal states.
            left_reward: reward for terminating on the left.
            right_reward: reward for terminating on the right.
            step_reward: reward for any other move.
        """
        assert steps > 2, f"Steps must be greater than 2. Got {steps}"
        super().__init__()

        self.steps = steps
        self.left_end_reward = left_end_reward
        self.right_end_reward = right_end_reward
        self.step_reward = step_reward
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int64,
            minimum=0,
            maximum=len(constants.ACTIONS) - 1,
            name="action",
        )
        self._observation_spec = {
            constants.OBS_KEY_POSITION: array_spec.BoundedArraySpec(
                shape=(),
                dtype=np.int64,
                minimum=0,
                maximum=steps - 1,
                name=constants.OBS_KEY_POSITION,
            ),
            constants.OBS_KEY_STEPS: array_spec.BoundedArraySpec(
                shape=(),
                dtype=np.int64,
                minimum=steps,
                maximum=steps,
                name=constants.OBS_KEY_STEPS,
            ),
            constants.OBS_KEY_RIGHT_END_REWARD: array_spec.BoundedArraySpec(
                shape=(),
                dtype=np.float32,
                minimum=np.finfo(np.float32).min,
                maximum=np.finfo(np.float32).max,
                name=constants.OBS_KEY_RIGHT_END_REWARD,
            ),
            constants.OBS_KEY_LEFT_END_REWARD: array_spec.BoundedArraySpec(
                shape=(),
                dtype=np.float32,
                minimum=np.finfo(np.float32).min,
                maximum=np.finfo(np.float32).max,
                name=constants.OBS_KEY_LEFT_END_REWARD,
            ),
            constants.OBS_KEY_STEP_REWARD: array_spec.BoundedArraySpec(
                shape=(),
                dtype=np.float32,
                minimum=np.finfo(np.float32).min,
                maximum=np.finfo(np.float32).max,
                name=constants.OBS_KEY_STEP_REWARD,
            ),
        }

        # env specific
        self._observation: Optional[NestedArray] = None
        self._seed = None

    def observation_spec(self) -> NestedArraySpec:
        """Defines the observations provided by the environment.

        May use a subclass of `ArraySpec` that specifies additional properties such
        as min and max bounds on the values.

        Returns:
          An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.
        """
        return self._observation_spec

    def action_spec(self) -> NestedArraySpec:
        """Defines the actions that should be provided to `step()`.

        May use a subclass of `ArraySpec` that specifies additional properties such
        as min and max bounds on the values.

        Returns:
          An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.
        """
        return self._action_spec

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
        new_observation, reward = apply_action(self._observation, action)
        finished = is_finished(new_observation)
        self._observation = new_observation
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
        self._observation = beginning_state(
            steps=self.steps,
            left_end_reward=self.left_end_reward,
            right_end_reward=self.right_end_reward,
            step_reward=self.step_reward,
        )
        return ts.restart(observation=copy.deepcopy(self._observation))

    def render(self, mode="rgb_array") -> Optional[NestedArray]:
        if self._observation is None:
            raise RuntimeError(
                f"{type(self).__name__} environment needs to be reset. Call the `reset` method."
            )
        if mode == "rgb_array":
            return state_representation(self._observation)
        return super().render(mode)

    def seed(self, seed: Seed = None) -> Any:
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)
        return self._seed


class StateRandomWalkMdpDiscretizer(markovdp.MdpDiscretizer):
    """
    Creates an environment discrete maps for states and actions.
    """

    def state(self, observation: Any) -> int:
        """
        Maps an observation to a state ID.
        """
        del self
        return get_state_id(observation)

    def action(self, action: Any) -> int:
        """
        Maps an agent action to an action ID.
        """
        del self
        action_: int = npsci.item(action)
        return action_


def apply_action(
    observation: NestedArray, action: NestedArray
) -> Tuple[NestedArray, float]:
    """
    Computes a new observation and reward given the current state and action.

    Example:
        steps: 5
        starting_state: 2
        possible_states: [0, 4]
        (0 is terminal, 1 is a left step, 3 is a right steps, 6 is a terminal state)
    """
    right_terminal_state = observation[constants.OBS_KEY_STEPS] - 1
    terminal_states = set((0, right_terminal_state))
    new_observation = copy.deepcopy(observation)
    # default reward + for terminal states
    reward = 0.0
    if npsci.item(observation[constants.OBS_KEY_POSITION]) in terminal_states:
        # override step reward in terminal states
        step_reward = 0.0
    else:
        if action == constants.GO_LEFT:
            new_observation[constants.OBS_KEY_POSITION] -= 1
        elif action == constants.GO_RIGHT:
            new_observation[constants.OBS_KEY_POSITION] += 1

        step_reward = new_observation[constants.OBS_KEY_STEP_REWARD]
        if new_observation[constants.OBS_KEY_POSITION] == right_terminal_state:
            reward = new_observation[constants.OBS_KEY_RIGHT_END_REWARD]
        elif new_observation[constants.OBS_KEY_POSITION] == 0:  # left end
            reward = new_observation[constants.OBS_KEY_LEFT_END_REWARD]
    return new_observation, reward + step_reward


def beginning_state(
    steps: int, left_end_reward: float, right_end_reward: float, step_reward: float
) -> Mapping[str, Any]:
    """
    Generates the starting state.
    """
    return {
        constants.OBS_KEY_POSITION: np.array(math.floor(steps / 2), dtype=np.int64),
        constants.OBS_KEY_STEPS: np.array(steps, dtype=np.int64),
        constants.OBS_KEY_RIGHT_END_REWARD: np.array(
            right_end_reward, dtype=np.float32
        ),
        constants.OBS_KEY_LEFT_END_REWARD: np.array(left_end_reward, dtype=np.float32),
        constants.OBS_KEY_STEP_REWARD: np.array(step_reward, dtype=np.float32),
    }


def is_finished(observation: Mapping[str, Any]) -> bool:
    """
    This function is called after the action is applied - i.e.
    observation is a new state from taking the `action` passed in.
    """
    return npsci.item(observation[constants.OBS_KEY_POSITION]) in set(
        (0, observation[constants.OBS_KEY_STEPS] - 1)
    )


def create_env_spec(
    steps: int,
    left_end_reward: float = constants.LEFT_REWARD,
    right_end_reward: float = constants.RIGHT_REWARD,
    step_reward: float = constants.STEP_REWARD,
) -> envspec.EnvSpec:
    """
    Creates an env spec from a config.
    """
    environment = StateRandomWalk(
        steps=steps,
        left_end_reward=left_end_reward,
        right_end_reward=right_end_reward,
        step_reward=step_reward,
    )
    discretizer = StateRandomWalkMdpDiscretizer()
    num_states = steps
    num_actions = len(constants.ACTIONS)
    env_desc = envdesc.EnvDesc(num_states=num_states, num_actions=num_actions)
    return envspec.EnvSpec(
        name=constants.ENV_NAME,
        level=__encode_env(
            steps=steps,
            left_end_reward=left_end_reward,
            right_end_reward=right_end_reward,
            step_reward=step_reward,
        ),
        environment=environment,
        discretizer=discretizer,
        env_desc=env_desc,
    )


def __encode_env(
    steps: int,
    left_end_reward: float,
    right_end_reward: float,
    step_reward: float,
) -> str:
    hash_key = tuple((steps, left_end_reward, right_end_reward, step_reward))
    hashing = hashlib.sha512(str(hash_key).encode("UTF-8"))
    return base64.b32encode(hashing.digest()).decode("UTF-8")


def get_state_id(observation: Mapping[str, Any]) -> int:
    """
    Computes an integer ID that represents that state.
    """
    state_id: int = observation[constants.OBS_KEY_POSITION]
    return state_id


def state_representation(observation: Mapping[str, Any]) -> NestedArray:
    """
    An array view of the state, where the position of the
    agent is marked with an 1.

    E.g. {
        "position": 2,
        "steps": 5,
        "step_reward": 0,
        ...
    }

    output = [0, 0, 1, 0, 0]
    """
    array = np.zeros(shape=(observation[constants.OBS_KEY_STEPS],), dtype=np.int64)
    array[npsci.item(observation[constants.OBS_KEY_POSITION])] = 1
    return array