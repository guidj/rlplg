"""
#1 Taking other pills has no impact - other than the right pills
[0] - no medication
[1] - red
[2] - red, green
[3] - red, green, red...
[4] - red, green, red, green
[5] - red, green, red, green, wait
Done
[0, 1, 2, 3, 4, 5, TERMINAL_STATE]

S: integer indicating pos: from 0 to 5
A: integer 3 actions - red, green, wait
(S x A x S): (0, red, 1), (0, green/wait, 0)
R: (action penalty of -1) + (-1 if you take the wrong action); zero for terminal state

(0, green, 0) - (0, [red], [1])

(0, green, 0)
(0, red, 1)
(1, red, 1) --> (1, green, 2)
(1, green, 2)

outcome, cumulative reward: -100 /// -10
state change: red to green instead staying red
we want a change of state to represent progression of cure

What is ||x - x'||?
- x: probably the trajectory for us; we will compare the change in trajectories

problem formulation: tutoring an agent (e.g. doctor, specialist)
assumptions we're making: our policy knows better
Teacher policy: optimize to make as fewer changes on the student policy
"""

import base64
import copy
import hashlib
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing.types import NestedArray, NestedArraySpec, Seed

from rlplg import envdesc, envspec, npsci
from rlplg.environments.redgreen import constants
from rlplg.learning.tabular import markovdp


class RedGreenSeq(py_environment.PyEnvironment):
    """
    An environment where an agent is meant to follow a pre-defined
    sequence of actions.
    """

    metadata = {"render.modes": ["raw"]}

    def __init__(self, cure: Sequence[str]):
        """
        Args:
            cure_sequence: sequence of actions to take to cure a patient.
                Each value is a string, one of `constants.ACTION_NAME_MAPPING`.
        """
        super().__init__()

        if len(cure) < 1:
            raise ValueError(f"Cure sequence should be longer than one: {len(cure)}")

        self.cure_sequence = [constants.ACTION_NAME_MAPPING[action] for action in cure]
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=len(constants.ACTIONS) - 1,
            name="action",
        )
        self._observation_spec = {
            constants.OBS_KEY_CURE_SEQUENCE: array_spec.BoundedArraySpec(
                shape=(len(self.cure_sequence),),
                dtype=np.int32,
                minimum=np.array([min(constants.ACTIONS)] * len(cure)),
                maximum=np.array([max(constants.ACTIONS)] * len(cure)),
                name=constants.OBS_KEY_CURE_SEQUENCE,
            ),
            constants.OBS_KEY_POSITION: array_spec.BoundedArraySpec(
                shape=(),
                dtype=np.int32,
                minimum=0,
                maximum=len(cure),
                name=constants.OBS_KEY_POSITION,
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
        self._observation = beginning_state(self.cure_sequence)
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


class RedGreenMdpDiscretizer(markovdp.MdpDiscretizer):
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
        cure_sequence: [red, green, red, wait]
        starting_state: 0
        possible_states: [0, 1 (after red), 2 (after green), 3 (after red), 4 (after wait - terminal state)]
        len(possible states) = len(cure_sequence) + 1
        terminal state = len(cure_sequence)
        possible_actions: 0 (red), 1 (green), 2 (wait)


        S = 0 (starting pos), Action 1 (green treatment)
            new state = 0 (no change from starting pos)
            reward = -1 - 1 = -2 (one penalty for acting, one for the wrong move)
        S = 0 (starting pos), Action 0 (red treatment)
            new state = 1 (has taken the first treatment)
            reward = -1 + 0 = -1 (one penalty for acting)
        S = 1 (first treatment), Action 1 (green treatment)
            new state = 2 (has taken the second treatment)
            reward = -1 + 0 = -1 (one penalty for acting)
        S = 2 (second treatment), Action 0 (red treatment)
            new state = 3 (has taken the third treatment)
            reward = -1 + 0 = -1 (one penalty for acting)
        S = 3 (second treatment), Action 2 (wait treatment)
            new state = 4 (has taken the fourth treatment) - terminal state
            reward = -1 + 0 = -1 (one penalty for acting)

    """
    pos = observation[constants.OBS_KEY_POSITION]
    terminal_state = len(observation[constants.OBS_KEY_CURE_SEQUENCE])
    new_observation = copy.deepcopy(observation)
    if observation[constants.OBS_KEY_POSITION] == terminal_state:
        move_penalty = 0.0
        reward = 0.0
    else:
        move_penalty = -1.0
        if action == observation[constants.OBS_KEY_CURE_SEQUENCE][pos]:
            new_observation[constants.OBS_KEY_POSITION] += 1
            reward = 0.0
        else:
            # wrong move
            reward = -1.0

    return new_observation, move_penalty + reward


def beginning_state(cure_sequence: Sequence[int]):
    """
    Generates the starting state.
    """
    return {
        constants.OBS_KEY_CURE_SEQUENCE: cure_sequence,
        constants.OBS_KEY_POSITION: 0,
    }


def is_finished(observation: NestedArray) -> bool:
    """
    This function is called after the action is applied - i.e.
    observation is a new state from taking the `action` passed in.
    """
    # does that fact that we just went into the
    # terminal state matter? No
    terminal_state = len(observation[constants.OBS_KEY_CURE_SEQUENCE])
    is_finished_: bool = observation[constants.OBS_KEY_POSITION] == terminal_state
    return is_finished_


def create_env_spec(cure: Sequence[str]) -> envspec.EnvSpec:
    """
    Creates an env spec from a gridworld config.
    """
    environment = RedGreenSeq(cure=cure)
    discretizer = RedGreenMdpDiscretizer()
    num_states = len(cure) + 1
    num_actions = len(constants.ACTIONS)
    env_desc = envdesc.EnvDesc(num_states=num_states, num_actions=num_actions)
    return envspec.EnvSpec(
        name=constants.ENV_NAME,
        level=__encode_env(cure),
        environment=environment,
        discretizer=discretizer,
        env_desc=env_desc,
    )


def __encode_env(cure: Sequence[str]) -> str:
    hash_key = tuple(cure)
    hashing = hashlib.sha512(str(hash_key).encode("UTF-8"))
    return base64.b32encode(hashing.digest()).decode("UTF-8")


def get_state_id(observation: NestedArray) -> int:
    """
    Computes an integer ID that represents that state.
    """
    state_id: int = observation[constants.OBS_KEY_POSITION]
    return state_id


def state_observation(cure_sequence: Sequence[int], state_id: int) -> NestedArray:
    """
    Generates a state observation, given an interger ID and the cure sequence.

    Args:
        state_id: An interger ID of the current state.
        cure_sequence: The sequence of actions required to end the game successfully.

    Returns:
        A mapping with the cure sequence and the current state.
    """
    return {
        constants.OBS_KEY_CURE_SEQUENCE: cure_sequence,
        constants.OBS_KEY_POSITION: state_id,
    }


def state_representation(observation: NestedArray) -> Sequence[int]:
    """
    An array view of the state, where successful steps are marked
    with 1s and missing steps are marked with a 0.

    E.g.
    observation = {
        "position": 2,
        "cure_sequnece": [0, 1, 2, 0, 1]
    }

    output = [1, 1, 1, 0, 0]
    """
    pos = observation[constants.OBS_KEY_POSITION]
    return [
        1 if idx < pos else 0
        for idx in range(len(observation[constants.OBS_KEY_CURE_SEQUENCE]))
    ]
