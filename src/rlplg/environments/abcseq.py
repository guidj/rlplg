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
  completing the previous one. The first letter is an exception - there is no prior
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
from typing import Any, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlplg import core, npsci
from rlplg.core import InitState, MutableEnvTransition, RenderType, TimeStep

ENV_NAME = "ABCSeq"
MIN_SEQ_LENGTH = 1
LETTERS = [chr(value) for value in range(ord("A"), ord("Z") + 1)]
NUM_LETTERS = len(LETTERS)


class ABCSeq(gym.Env[np.ndarray, int]):
    """
    A sequence of tokens in order.
    The goal is to follow then, from left to right, until the end, selecting
    the current state as an action in the current state.
    """

    metadata = {"render.modes": ["raw"]}

    def __init__(self, length: int, render_mode: str = "raw"):
        super().__init__()
        self.length = length
        self.render_mode = render_mode
        if NUM_LETTERS < length < MIN_SEQ_LENGTH:
            raise ValueError(
                f"Length must be between {MIN_SEQ_LENGTH} and {NUM_LETTERS}: {length}"
            )
        self.action_space = spaces.Discrete(self.length)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(length + 1,), dtype=np.int64
        )
        self.transition: MutableEnvTransition = {}
        for state in range(self.length + 1):
            self.transition[state] = {}
            state_obs = state_observation(state, self.length)
            for action in range(self.length):
                self.transition[state][action] = []
                next_state_obs, reward = apply_action(state_obs, action)
                next_state_id = get_state_id(next_state_obs)
                for next_state in range(self.length + 1):
                    prob = 1.0 if next_state == next_state_id else 0.0
                    actual_reward = reward if next_state == next_state_id else 0.0
                    terminated = state != next_state and next_state == self.length
                    self.transition[state][action].append(
                        (prob, next_state, actual_reward, terminated)
                    )

        # env specific
        self._observation: np.ndarray = np.empty(shape=(0,))
        self._seed: Optional[int] = None

    def step(self, action: int) -> TimeStep:
        """Updates the environment according to action and returns a `TimeStep`.

        See `step(self, action)` docstring for more details.

        Args:
            action: A policy's chosen action.

        """
        new_observation, reward = apply_action(self._observation, action)
        finished = is_finished(new_observation, action)
        self._observation = new_observation
        return copy.deepcopy(self._observation), reward, finished, False, {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None
    ) -> InitState:
        """Starts a new sequence, returns the first `TimeStep` of this sequence.

        See `reset(self)` docstring for more details
        """
        del options
        self.seed(seed)
        self._observation = beginning_state(length=self.length)
        return copy.deepcopy(self._observation), {}

    def render(self) -> RenderType:
        """
        Renders a view of the environment's current
        state.
        """
        if self.render_mode == "rgb_array":
            return copy.deepcopy(self._observation)
        return super().render()

    def seed(self, seed: Optional[int] = None) -> Any:
        """
        Sets a seed, if defined.
        """
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)
        return self._seed


class ABCSeqMdpDiscretizer(core.MdpDiscretizer):
    """
    Creates an environment discrete maps for states and actions.
    """

    def state(self, observation: np.ndarray) -> int:
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


def apply_action(observation: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
    """
    Takes an action in a given state.

    Returns:
        New observation and reward
    """
    return _step_observation(observation, action), _step_reward(observation, action)


def _step_observation(observation: np.ndarray, action: int) -> np.ndarray:
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


def _step_reward(observation: np.ndarray, action: int) -> float:
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


def beginning_state(length: int) -> np.ndarray:
    """
    Args:
        lenght: number of tokens in sequence.
    Returns:
        Initial observation for a starting game.
    """
    observation = np.zeros(shape=(length + 1,), dtype=np.int64)
    observation[0] = 1
    return observation


def is_finished(observation: np.ndarray, action: int) -> bool:
    """
    This function is called after the action is applied - i.e.
    observation is a new state from taking the `action` passed in.
    """
    del action
    return np.sum(observation) == observation.size  # type: ignore


def create_env_spec(length: int) -> core.EnvSpec:
    """
    Creates an env spec from a gridworld config.
    """
    environment = ABCSeq(length=length)
    discretizer = ABCSeqMdpDiscretizer()
    mdp = core.EnvMdp(
        env_desc=core.EnvDesc(num_states=length + 1, num_actions=length),
        transition=environment.transition,
    )
    return core.EnvSpec(
        name=ENV_NAME,
        level=str(length),
        environment=environment,
        discretizer=discretizer,
        mdp=mdp,
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
