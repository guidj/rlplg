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
The agent can choose any letter.
The observation is just what the agent has already mastered.

[A, B, C, D] => [1, 0, 0, 0, 0]
So the number of actions with 4 letters.

Transitions:
  - The agent can only say to have learned a letter if they have go to it after
  completing the previous one. The first letter is an execption - there is no prior
  letter.
  - If an agent chooses a letter that is further ahead, they get penalized by the distance
  and the state doesn't change.
  - If they choose a letter that lies before, they also get penalized - the entire sequence
  of steps forward until the beginning, and from there to the letter they chose.
  The state doesn't change.

The number of possible states is the number of letters plus one:
E.g. With four letters (first value is set to 1 to simplify calculations)
  - starting: [1, 0, 0, 0, 0]
  - mastered the first letter: [1, 1, 0, 0, 0]
  - mastered the second letter: [1, 1, 1, 0, 0]
  - mastered the third letter: [1, 1, 1, 1, 0]
  - mastered the fourth letter: [1, 1, 1, 1, 1] (terminal state)


Notes:
  - If an agent chooses the right action the first time in a given state,
  it won't get to explore other actions in that state.
"""

import copy
from typing import Any, Optional

import numpy as np

from rlplg import core, envdesc, envspec, npsci
from rlplg.learning.tabular import markovdp

ENV_NAME = "ABCSeq"
MIN_SEQ_LENGTH = 1
LETTERS = [chr(value) for value in range(ord("A"), ord("Z") + 1)]
NUM_LETTERS = len(LETTERS)


class ABCSeq(core.PyEnvironment):
    metadata = {"render.modes": ["raw"]}

    def __init__(self, length: int):
        super().__init__()
        self.length = length
        if NUM_LETTERS < length < MIN_SEQ_LENGTH:
            raise ValueError(
                f"Length must be between {MIN_SEQ_LENGTH} and {NUM_LETTERS}: {length}"
            )
        self._action_spec = ()
        self._observation_spec = ()

        # env specific
        self._observation: Optional[np.ndarray] = None
        self._seed = None

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
        new_observation = apply_action(self._observation, action)
        reward = action_reward(self._observation, action)

        finished = is_finished(new_observation, action)

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
        self._observation = beginning_state(length=self.length)
        return core.TimeStep.restart(observation=copy.deepcopy(self._observation))

    def render(self, mode="rgb_array") -> Optional[Any]:
        if mode == "rgb_array":
            return copy.deepcopy(self._observation)
        return super().render(mode)

    def seed(self, seed: int = None) -> Any:
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
        action_: int = npsci.item(action)
        return action_


def apply_action(observation: np.ndarray, action: Any) -> np.ndarray:
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
            np.array(1, dtype=np.int64)
            if observation[action] == 1
            else observation[action + 1]
        )
    return new_observation


def action_reward(observation: np.ndarray, action: Any) -> float:
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
    observation = np.zeros(shape=(length + 1,), dtype=np.int64)
    observation[0] = 1
    return observation


def is_finished(observation: np.ndarray, action: Any) -> bool:
    """
    This function is called after the action is applied - i.e.
    observation is a new state from taking the `action` passed in.
    """
    is_finished_: bool = (
        np.sum(observation) == observation.size and action == observation.size - 2
    )
    return is_finished_


def create_env_spec(length: int) -> envspec.EnvSpec:
    """
    Creates an env spec from a gridworld config.
    """
    environment = ABCSeq(length=length)
    discretizer = ABCSeqMdpDiscretizer()
    env_desc = envdesc.EnvDesc(num_states=length + 1, num_actions=length)
    return envspec.EnvSpec(
        name=ENV_NAME,
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
    state_id: int = np.sum(observation) - 1
    return state_id


def state_observation(state_id: int, length: int) -> np.ndarray:
    """
    Given a state ID for an environment, returns an observation.
    """
    if state_id > length:
        raise ValueError(f"State id should be <= length: {state_id} > {length}")
    observation = np.zeros(shape=(length + 1,), dtype=np.int64)
    observation[: state_id + 1] = 1
    return observation
