"""
This module contains the definition of the State-Random-Walk problem.

From Barto, Sutton, p.125

"...all episodes start in the center state, C,then proceed either
left or right by one state on each step, with equal probability.
Episodes terminate either on the extreme left or the extreme right.
When an episode terminates on the right, a reward of +1 occurs; all other rewards are zero."

A distiction is that we introduce agent actions - going left or right.
Combined with a random policy, it should produce the same effect.
"""

import base64
import copy
import hashlib
import math
from typing import Any, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlplg import core, npsci
from rlplg.core import InitState, MutableEnvTransition, RenderType, TimeStep

ENV_NAME = "StateRandomWalk"
GO_LEFT = 0
GO_RIGHT = 1
ACTIONS = [GO_LEFT, GO_RIGHT]
RIGHT_REWARD = 1
LEFT_REWARD = 0
STEP_REWARD = 0
OBS_KEY_POSITION = "position"
OBS_KEY_STEPS = "steps"
OBS_KEY_LEFT_END_REWARD = "left_end_reward"
OBS_KEY_RIGHT_END_REWARD = "right_end_reward"
OBS_KEY_STEP_REWARD = "step_reward"


class StateRandomWalk(gym.Env[Mapping[str, Any], int]):
    """
    An environment where an agent is meant to go right, until the end.
    The environment can terminate on the last left or right states.
    By default, only the right direction yields a positive reward.

    Example:
    Steps: 5
    The starting state is the middle state, i.e. ceil(steps / 2) (without shifting by zero-index, since zero is a terminal state)
    Starting state = ceil(5 / 2) = 3
    Terminal states: 0 and 6.
    """

    def __init__(
        self,
        steps: int,
        left_end_reward: float = LEFT_REWARD,
        right_end_reward: float = RIGHT_REWARD,
        step_reward: float = STEP_REWARD,
        render_mode: str = "rgb_array",
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
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Dict(
            {
                OBS_KEY_POSITION: spaces.Box(low=0, high=steps - 1, dtype=np.int64),
                OBS_KEY_STEPS: spaces.Box(low=steps, high=steps, dtype=np.int64),
                OBS_KEY_RIGHT_END_REWARD: spaces.Box(
                    low=right_end_reward, high=right_end_reward, dtype=np.float32
                ),
                OBS_KEY_LEFT_END_REWARD: spaces.Box(
                    low=left_end_reward, high=left_end_reward, dtype=np.float32
                ),
                OBS_KEY_STEP_REWARD: spaces.Box(
                    low=step_reward, high=step_reward, dtype=np.float32
                ),
            }
        )
        self.transition: MutableEnvTransition = {}
        for state in range(self.steps):
            self.transition[state] = {}
            state_obs = state_observation(
                position=state,
                steps=self.steps,
                left_end_reward=self.left_end_reward,
                right_end_reward=self.right_end_reward,
                step_reward=self.step_reward,
            )
            for action in range(len(ACTIONS)):
                self.transition[state][action] = []
                next_state_obs, reward = apply_action(
                    observation=state_obs, action=action
                )
                next_state_id = get_state_id(next_state_obs)
                for next_state in range(self.steps):
                    prob = 1.0 if next_state == next_state_id else 0.0
                    actual_reward = reward if next_state == next_state_id else 0.0
                    terminated = state != next_state and next_state in set(
                        [0, self.steps - 1]
                    )
                    self.transition[state][action].append(
                        (prob, next_state, actual_reward, terminated)
                    )

        # env specific
        self._observation: Mapping[str, Any] = {}
        self._seed: Optional[int] = None

    def step(self, action: int) -> TimeStep:
        """Updates the environment according to action and returns a `TimeStep`.

        See `step(self, action)` docstring for more details.

        Args:
            action: A policy's chosen action.

        """
        if self._observation == {}:
            raise RuntimeError(
                f"{type(self).__name__} environment needs to be reset. Call the `reset` method."
            )
        new_observation, reward = apply_action(self._observation, action)
        finished = is_finished(new_observation)
        self._observation = new_observation
        return copy.deepcopy(self._observation), reward, finished, False, {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None
    ) -> InitState:
        """Starts a new sequence, returns the `InitState` for this environment.

        See `reset(self)` docstring for more details
        """
        del options
        self.seed(seed)
        self._observation = state_observation(
            position=math.floor(self.steps / 2),
            steps=self.steps,
            left_end_reward=self.left_end_reward,
            right_end_reward=self.right_end_reward,
            step_reward=self.step_reward,
        )
        return copy.deepcopy(self._observation), {}

    def render(self) -> RenderType:
        """
        Renders a view of the environment's current
        state.
        """
        if self._observation == {}:
            raise RuntimeError(
                f"{type(self).__name__} environment needs to be reset. Call the `reset` method."
            )
        if self.render_mode == "rgb_array":
            return state_representation(self._observation)
        return super().render()

    def seed(self, seed: Optional[int] = None) -> Any:
        """
        Sets a seed, if defined.
        """
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)
        return self._seed


class StateRandomWalkMdpDiscretizer(core.MdpDiscretizer):
    """
    Creates an environment discrete maps for states and actions.
    """

    def state(self, observation: Mapping[str, Any]) -> int:
        """
        Maps an observation to a state ID.
        """
        del self
        return get_state_id(observation)

    def action(self, action: int) -> int:
        """
        Maps an agent action to an action ID.
        """
        del self
        action_: int = npsci.item(action)
        return action_


def apply_action(observation: Mapping[str, Any], action: int) -> Tuple[Any, float]:
    """
    Computes a new observation and reward given the current state and action.

    Example:
        steps: 5
        starting_state: 2
        possible_states: [0, 4]
        (0 is terminal, 1 is a left step, 3 is a right steps, 4 is a terminal state)
    """
    right_terminal_state = observation[OBS_KEY_STEPS] - 1
    terminal_states = set((0, right_terminal_state))
    new_observation = dict(**copy.deepcopy(observation))
    # default reward + for terminal states
    reward = 0.0
    if npsci.item(observation[OBS_KEY_POSITION]) in terminal_states:
        # override step reward in terminal states
        step_reward = 0.0
    else:
        if action == GO_LEFT:
            new_observation[OBS_KEY_POSITION] -= 1
        elif action == GO_RIGHT:
            new_observation[OBS_KEY_POSITION] += 1

        step_reward = new_observation[OBS_KEY_STEP_REWARD]
        if new_observation[OBS_KEY_POSITION] == right_terminal_state:
            reward = new_observation[OBS_KEY_RIGHT_END_REWARD]
        elif new_observation[OBS_KEY_POSITION] == 0:  # left end
            reward = new_observation[OBS_KEY_LEFT_END_REWARD]
    return new_observation, reward + step_reward


def state_observation(
    position: int,
    steps: int,
    left_end_reward: float,
    right_end_reward: float,
    step_reward: float,
) -> Mapping[str, Any]:
    """
    Generates the starting state.
    """
    return {
        OBS_KEY_POSITION: np.array(position, dtype=np.int64),
        OBS_KEY_STEPS: np.array(steps, dtype=np.int64),
        OBS_KEY_RIGHT_END_REWARD: np.array(right_end_reward, dtype=np.float32),
        OBS_KEY_LEFT_END_REWARD: np.array(left_end_reward, dtype=np.float32),
        OBS_KEY_STEP_REWARD: np.array(step_reward, dtype=np.float32),
    }


def is_finished(observation: Mapping[str, Any]) -> bool:
    """
    This function is called after the action is applied - i.e.
    observation is a new state from taking the `action` passed in.
    """
    return npsci.item(observation[OBS_KEY_POSITION]) in set(
        (0, observation[OBS_KEY_STEPS] - 1)
    )


def create_env_spec(
    steps: int,
    left_end_reward: float = LEFT_REWARD,
    right_end_reward: float = RIGHT_REWARD,
    step_reward: float = STEP_REWARD,
) -> core.EnvSpec:
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
    num_actions = len(ACTIONS)
    return core.EnvSpec(
        name=ENV_NAME,
        level=__encode_env(
            steps=steps,
            left_end_reward=left_end_reward,
            right_end_reward=right_end_reward,
            step_reward=step_reward,
        ),
        environment=environment,
        discretizer=discretizer,
        mdp=core.EnvMdp(
            env_desc=core.EnvDesc(num_states=num_states, num_actions=num_actions),
            transition=environment.transition,
        ),
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
    state_id: int = observation[OBS_KEY_POSITION].item()
    return state_id


def state_representation(observation: Mapping[str, Any]) -> np.ndarray:
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
    array = np.zeros(shape=(observation[OBS_KEY_STEPS],), dtype=np.int64)
    array[npsci.item(observation[OBS_KEY_POSITION])] = 1
    return array
