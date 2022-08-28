"""
This is an episodic task.
  - Action space ~ state space.

The objective of the environment is to learn the alphabet in sequence.
Starting from the first letter, the agent is suppose to choose the one that follows it,
and then the one that follows after.
Choosing letters out of order incurs a negative reward, while choosing letters
in order incurs the smallest penalty reward.
The player reaches the end of the game when they reach the final letter,
regardless of how many they have gone through.

Observation:
The agent can choose any letter next, or to stop playing - i.e. jump to the end.
The observation is just what the agent has already mastered.
As such, there is a final non-letter state/action, and the agent must choose it
to end the game.
Choosing this action before mastering every letter incurs a penalty
equivalent to the number of letters unmastered.
Failing to choose the action after mastering every letter incurs a penalty of -1.

[A, B, C, D] => [0, 0, 0, 0, Stop]
So the number of actions with 4 letters is 5.

Transitions:
  - The agent can only say to have learned a letter if they have go to it after
  completing the previous one. The first letter is an execption - there is no prior
  letter.
  - If an agent chooses a letter that is further ahead, they get penalized by the distance
  and the state doesn't change.
  - If they choose a letter that lies before, they also get penalized - the entire sequence
  of steps forward until the beginning, and from there to the letter they chose.
  The state doesn't change.
  - After msatering every letter, they must choose to stop playing. Any other
  action will incur a penalty of one.

The number of possible states is the number of letters plus one:
E.g. With four letters
  - starting: [0, 0, 0, 0]
  - mastered the first letter: [1, 0, 0, 0]
  - mastered the second letter: [1, 1, 0, 0]
  - mastered the third letter: [1, 1, 1, 0]
  - mastered the fourth letter: [1, 1, 1, 1] (terminal state)


Notes:
  - If an agent chooses the right action the first time in a given state,
  it won't get to explore other actions in that state.
"""

import copy
from typing import Any, Optional

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing.types import NestedArray, NestedArraySpec, Seed

from rlplg import envdesc, envspec, npsci
from rlplg.environments.alphabet import constants
from rlplg.learning.tabular import markovdp


class ABCSeq(py_environment.PyEnvironment):
    metadata = {"render.modes": ["raw"]}

    def __init__(self, length: int):
        super().__init__()
        self.length = length
        if constants.NUM_LETTERS < length < constants.MIN_SEQ_LENGTH:
            raise ValueError(
                f"Length must be between {constants.MIN_SEQ_LENGTH} and {constants.NUM_LETTERS}: {length}"
            )
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=length - 1,
            name="action",
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(length + 1,),
            dtype=np.int32,
            minimum=0,
            name="observation",
        )

        # env specific
        self._observation: Optional[np.ndarray] = None
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
        new_observation = apply_action(self._observation, action)
        reward = action_reward(self._observation, action)

        finished = is_finished(new_observation, action)

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
        self._observation = beginning_state(length=self.length)
        return ts.restart(observation=copy.deepcopy(self._observation))

    def render(self, mode="rgb_array") -> Optional[NestedArray]:
        if mode == "rgb_array":
            return copy.deepcopy(self._observation)
        return super().render(mode)

    def seed(self, seed: Seed = None) -> Any:
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)
        return self._seed


class ABCSeqMdpDiscretizer(markovdp.MdpDiscretizer):
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
        return npsci.item(action)


def apply_action(observation: np.ndarray, action: NestedArray) -> np.ndarray:
    """
    Transitions:
        - The agent can only say to have learned a letter if they go to it after
        completing the previous one. The first letter is an execption - there is no prior
        letter.
        - If an agent chooses a letter that is further ahead, they get penalized by the distance
        and the state doesn't change.
        - If they choose a letter that lies before, they also get penalized - the entire sequence
        of steps forward until the beginning, and from there to the letter they chose.
        The state doesn't change.
        - After msatering every letter, they enter a terminal state.
    """
    new_observation = copy.deepcopy(observation)
    # For four letters, obs.size == 5
    # Check action in within bounds
    if action < observation.size - 1:
        new_observation[action + 1] = (
            1 if observation[action] == 1 else observation[action + 1]
        )
    return new_observation


def action_reward(observation: np.ndarray, action: NestedArray) -> float:
    """
    One penalty per turn and one for the distance - except
    in the terminal state.
    """
    chosen_step = action + 1
    (unmastered_letters,) = np.where(observation == 0)
    # non terminal state
    if unmastered_letters.size > 0:
        next_letter = unmastered_letters[0]
        if chosen_step == next_letter:
            distance = 0.0
        elif chosen_step > next_letter:
            distance = chosen_step - next_letter
        else:
            # distance from the current position to the chosen
            # one, going forward and shifting back to the starting
            # state
            num_letters = observation.size - 1
            distance = (num_letters - next_letter + 1) + chosen_step

        turn_penalty = -1.0

    else:
        # terminal state
        distance = 0.0
        turn_penalty = 0.0

    return -distance + turn_penalty


def beginning_state(length: int):
    """
    Args:
        lenght: number of tokens in sequence.
    Returns:
        Initial observation for a starting game.
    """
    observation = np.zeros(shape=(length + 1,), dtype=np.int32)
    observation[0] = 1
    return observation


def is_finished(observation: np.ndarray, action: NestedArray) -> bool:
    """
    This function is called after the action is applied - i.e.
    observation is a new state from taking the `action` passed in.
    """
    return np.sum(observation) == observation.size and action == observation.size - 2


def create_env_spec(length: int) -> envspec.EnvSpec:
    """
    Creates an env spec from a gridworld config.
    """
    environment = ABCSeq(length=length)
    discretizer = ABCSeqMdpDiscretizer()
    env_desc = envdesc.EnvDesc(num_states=length + 1, num_actions=length)
    return envspec.EnvSpec(
        name=constants.ENV_NAME,
        level=str(length),
        environment=environment,
        discretizer=discretizer,
        env_desc=env_desc,
    )


def get_state_id(observation: np.ndarray) -> int:
    """
    Each letter can be a zero or one.
    They can only be learned in order.
    There is one dummy value of one in the observation.
    For 2 letters:
        [1, 0, 0] = 0 (starting state)
        [1, 1, 0] = 1
        [1, 1, 1] = 2 (terminal state)
    """
    return np.sum(observation) - 1


def state_observation(state_id: int, length: int) -> np.ndarray:
    """
    Given a state ID for an environment, returns an observation.
    """
    if state_id > length:
        raise ValueError(f"State id should be <= length: {state_id} > {length}")
    observation = np.zeros(shape=(length + 1,), dtype=np.int32)
    observation[: state_id + 1] = 1
    return observation
