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
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from rlplg import core, envdesc, envspec, npsci
from rlplg.learning.tabular import markovdp

ENV_NAME = "RedGreenSeq"
RED_PILL = 0
GREEN_PILL = 1
WAIT = 2
ACTIONS = [RED_PILL, GREEN_PILL, WAIT]

ACTION_NAME_MAPPING = {"red": RED_PILL, "green": GREEN_PILL, "wait": WAIT}
OBS_KEY_CURE_SEQUENCE = "cure_sequence"
OBS_KEY_POSITION = "position"


class RedGreenSeq(core.PyEnvironment):
    """
    An environment where an agent is meant to follow a pre-defined
    sequence of actions.
    """

    def __init__(self, cure: Sequence[str], render_mode: str = "rgb_array"):
        """
        Args:
            cure_sequence: sequence of actions to take to cure a patient.
                Each value is a string, one of `ACTION_NAME_MAPPING`.
        """
        super().__init__()

        if len(cure) < 1:
            raise ValueError(f"Cure sequence should be longer than one: {len(cure)}")

        self.cure_sequence = [ACTION_NAME_MAPPING[action] for action in cure]
        self.render_mode = render_mode
        self._action_spec = ()
        self._observation_spec = {
            OBS_KEY_CURE_SEQUENCE: (),
            OBS_KEY_POSITION: (),
        }

        # env specific
        self._observation: Mapping[str, Any] = {}
        self._seed: Optional[int] = None

    def observation_spec(self) -> Any:
        """Defines the observations provided by the environment.

        May use a subclass of `ArraySpec` that specifies additional properties such
        as min and max bounds on the values.

        Returns:
          An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.
        """
        return self._observation_spec

    def action_spec(self) -> Any:
        """Defines the actions that should be provided to `step()`.

        May use a subclass of `ArraySpec` that specifies additional properties such
        as min and max bounds on the values.

        Returns:
          An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.
        """
        return self._action_spec

    def _step(self, action: Any) -> core.TimeStep:
        """Updates the environment according to action and returns a `TimeStep`.

        See `step(self, action)` docstring for more details.

        Args:
            action: A NumPy array, or a nested dict, list or tuple of arrays
                corresponding to `action_spec()`.
        """
        if self._observation == {}:
            raise RuntimeError(
                f"{type(self).__name__} environment needs to be reset. Call the `reset` method."
            )
        new_observation, reward = apply_action(self._observation, action)

        finished = is_finished(new_observation)

        self._observation = new_observation
        if finished:
            return core.TimeStep.termination(
                observation=copy.deepcopy(self._observation), reward=reward
            )
        return core.TimeStep.transition(
            observation=copy.deepcopy(self._observation), reward=reward
        )

    def _reset(self) -> core.TimeStep:
        """Starts a new sequence, returns the first `TimeStep` of this sequence.

        See `reset(self)` docstring for more details
        """
        self._observation = beginning_state(self.cure_sequence)
        return core.TimeStep.restart(observation=copy.deepcopy(self._observation))

    def render(self) -> Any:
        if self._observation == {}:
            raise RuntimeError(
                f"{type(self).__name__} environment needs to be reset. Call the `reset` method."
            )
        if self.render_mode == "rgb_array":
            return state_representation(self._observation)
        return super().render()

    def seed(self, seed: Optional[int] = None) -> Any:
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


def apply_action(observation: Any, action: Any) -> Tuple[Any, float]:
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
    pos = observation[OBS_KEY_POSITION]
    terminal_state = len(observation[OBS_KEY_CURE_SEQUENCE])
    new_observation = copy.deepcopy(observation)
    if observation[OBS_KEY_POSITION] == terminal_state:
        move_penalty = 0.0
        reward = 0.0
    else:
        move_penalty = -1.0
        if action == observation[OBS_KEY_CURE_SEQUENCE][pos]:
            new_observation[OBS_KEY_POSITION] += 1
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
        OBS_KEY_CURE_SEQUENCE: cure_sequence,
        OBS_KEY_POSITION: 0,
    }


def is_finished(observation: Any) -> bool:
    """
    This function is called after the action is applied - i.e.
    observation is a new state from taking the `action` passed in.
    """
    # does that fact that we just went into the
    # terminal state matter? No
    terminal_state = len(observation[OBS_KEY_CURE_SEQUENCE])
    is_finished_: bool = observation[OBS_KEY_POSITION] == terminal_state
    return is_finished_


def create_env_spec(cure: Sequence[str]) -> envspec.EnvSpec:
    """
    Creates an env spec from a gridworld config.
    """
    environment = RedGreenSeq(cure=cure)
    discretizer = RedGreenMdpDiscretizer()
    num_states = len(cure) + 1
    num_actions = len(ACTIONS)
    env_desc = envdesc.EnvDesc(num_states=num_states, num_actions=num_actions)
    return envspec.EnvSpec(
        name=ENV_NAME,
        level=__encode_env(cure),
        environment=environment,
        discretizer=discretizer,
        env_desc=env_desc,
    )


def __encode_env(cure: Sequence[str]) -> str:
    hash_key = tuple(cure)
    hashing = hashlib.sha512(str(hash_key).encode("UTF-8"))
    return base64.b32encode(hashing.digest()).decode("UTF-8")


def get_state_id(observation: Any) -> int:
    """
    Computes an integer ID that represents that state.
    """
    state_id: int = observation[OBS_KEY_POSITION]
    return state_id


def state_observation(cure_sequence: Sequence[int], state_id: int) -> Mapping[str, Any]:
    """
    Generates a state observation, given an interger ID and the cure sequence.

    Args:
        state_id: An interger ID of the current state.
        cure_sequence: The sequence of actions required to end the game successfully.

    Returns:
        A mapping with the cure sequence and the current state.
    """
    return {
        OBS_KEY_CURE_SEQUENCE: cure_sequence,
        OBS_KEY_POSITION: state_id,
    }


def state_representation(observation: Any) -> Sequence[int]:
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
    pos = observation[OBS_KEY_POSITION]
    return [
        1 if idx < pos else 0 for idx in range(len(observation[OBS_KEY_CURE_SEQUENCE]))
    ]
