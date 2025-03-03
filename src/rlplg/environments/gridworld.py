"""
The objective of the environment is simple: go from a starting point to a goal state.
There can be cliffs, and if the agent falls into one, the reward is -100,
and they go back to the starting position.
The agent can go up, down, left and right.
If an action takes the agent outside the grid, they stay in the same position.
The reward for every action is -1.
"""

import abc
import contextlib
import copy
import io
import os
import os.path
import sys
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image as image

from rlplg.core import InitState, MutableEnvTransition, RenderType, TimeStep

ENV_NAME = "GridWorld"
CLIFF_PENALTY = -100.0
MOVE_PENALTY = -1.0
TERMINAL_REWARD = 0.0
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
MOVES = ["L", "R", "U", "D"]
COLOR_SILVER = (192, 192, 192)
PATH_BG = "path-bg.png"
CLIFF_BG = "cliff-bg.png"
ACTOR = "actor.png"


LAYER_AGENT = 0
LAYER_CLIFF = 1
LAYER_EXIT = 2

OBS_KEY_ID = "id"
OBS_KEY_SIZE = "size"
OBS_KEY_AGENT = "agent"
OBS_KEY_CLIFFS = "cliffs"
OBS_KEY_EXITS = "exits"
OBS_KEY_START = "start"


class GridWorld(gym.Env[Mapping[str, Any], int]):
    """
    GridWorld environment.
    """

    def __init__(
        self,
        size: Tuple[int, int],
        cliffs: Sequence[Tuple[int, int]],
        exits: Sequence[Tuple[int, int]],
        start: Tuple[int, int],
        render_mode: str = "rgb_array",
    ):
        super().__init__()
        validate_dimensions(size=size, start=start, cliffs=cliffs, exits=exits)
        validate_starting_grid(start=start, cliffs=cliffs, exits=exits)
        self.render_mode = render_mode
        self._height, self._width = size
        self._size = size
        self._start = start
        self._cliffs = set(cliffs)
        self._exits = set(exits)

        states_mapping_ = states_mapping(
            size=self._size,
            cliffs=tuple(self._cliffs),
        )
        self.__get_state_id = create_obs_state_id_fn(states=states_mapping_)

        num_states = len(states_mapping_)
        num_actions = len(MOVES)
        self.__reverse_state_mapping = {
            value: key for key, value in states_mapping_.items()
        }

        # left, right, up, down
        self.action_space = spaces.Discrete(len(MOVES))
        self.observation_space = spaces.Dict(
            {
                "id": spaces.Discrete(num_states),
                "start": spaces.Tuple(
                    (spaces.Discrete(self._height), spaces.Discrete(self._width))
                ),
                "agent": spaces.Tuple(
                    (spaces.Discrete(self._height), spaces.Discrete(self._width))
                ),
                "cliffs": spaces.Sequence(
                    spaces.Tuple(
                        (spaces.Discrete(self._height), spaces.Discrete(self._width))
                    )
                ),
                "exits": spaces.Sequence(
                    spaces.Tuple(
                        (spaces.Discrete(self._height), spaces.Discrete(self._width))
                    )
                ),
                "size": spaces.Box(
                    low=np.array([self._height, self._width]),
                    high=np.array([self._height, self._width]),
                    dtype=np.int64,
                ),
            }
        )
        self.transition: MutableEnvTransition = {}
        for state in range(num_states):
            self.transition[state] = {}
            state_pos = self.__reverse_state_mapping[state]
            for action in range(num_actions):
                self.transition[state][action] = []
                next_obs, reward = apply_action(
                    create_observation(
                        size=self._size,
                        start=self._start,
                        agent=state_pos,
                        cliffs=tuple(self._cliffs),
                        exits=tuple(self._exits),
                        get_state_id=self.__get_state_id,
                    ),
                    action=action,
                    get_state_id=self.__get_state_id,
                )
                for next_state in range(num_states):
                    next_state_pos = self.__reverse_state_mapping[next_state]
                    prob = 1.0 if next_obs[OBS_KEY_AGENT] == next_state_pos else 0.0
                    actual_reward = (
                        reward if next_obs[OBS_KEY_AGENT] == next_state_pos else 0.0
                    )
                    # transition to an exit
                    terminated = state != next_state and (next_state_pos in self._exits)
                    self.transition[state][action].append(
                        (prob, next_state, actual_reward, terminated)
                    )
        # env specific
        self._observation: Mapping[str, Any] = {}
        self._seed: Optional[int] = None
        self._rng: np.random.Generator = np.random.default_rng()

    def step(self, action: int) -> TimeStep:
        """Updates the environment according to action and returns a `TimeStep`.

        See `step(self, action)` docstring for more details.

        Args:
            action: A policy's chosen action.
        """
        if not self._observation:
            raise RuntimeError(
                f"{type(self).__name__} environment needs to be reset. Call the `reset` method."
            )
        next_observation, reward = apply_action(
            self._observation, action=action, get_state_id=self.__get_state_id
        )
        self._observation = next_observation
        finished = self._observation[OBS_KEY_AGENT] in self._exits
        # Note: obs fields are immutable;
        # so shallow copies suffice to prevent tampering with
        # internal state.
        return copy.copy(self._observation), reward, finished, False, {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None
    ) -> InitState:
        """Starts a new sequence, returns the `InitState` for this environment.

        See `reset(self)` docstring for more details
        """
        del options
        self.seed(seed)
        # place agent at the start
        self._observation = create_observation(
            size=self._size,
            start=self._start,
            agent=self._start,
            cliffs=tuple(self._cliffs),
            exits=tuple(self._exits),
            get_state_id=self.__get_state_id,
        )
        return copy.copy(self._observation), {}

    def render(self) -> RenderType:
        """
        Renders a view of the environment's current
        state.
        """
        if not self._observation:
            raise RuntimeError(
                f"{type(self).__name__} environment needs to be reset. Call the `reset` method."
            )
        if self.render_mode == "rgb_array":
            return as_grid_3d(self._observation)
        return super().render()

    def seed(self, seed: Optional[int] = None) -> Any:
        """
        Sets a seed, if defined.
        """
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(self._seed)
        return self._seed


class GridWorldRenderer:
    """
    Class for handling rendering of the environment
    """

    metadata = {"render.modes": ["raw", "rgb_array", "human", "ansi"]}

    def __init__(self, sprites_dir: Optional[str]):
        self.sprites: Optional[Sprites] = (
            BlueMoonSprites(sprites_dir) if sprites_dir is not None else None
        )
        self.viewer: Optional[Any] = None
        raise NotImplementedError("Rendering not supported!")

    def render(
        self,
        mode: str,
        observation: Mapping[str, Any],
        last_move: Optional[Any],
        caption: Optional[str],
        sleep: Optional[float] = 0.05,
    ) -> Any:
        """
        Args:
            mode: the rendering mode
            observation: the grid, as a numpy array of shape (W, H, 3)
            last_move: index of the last move, [0, 4)
            caption: string write on rendered Window
            sleep: time between rendering frames
        """
        if mode not in self.metadata["render.modes"]:
            raise ValueError(f"Unsupported mode: {mode}")
        if sleep is not None and sleep > 0:
            time.sleep(sleep)
        if mode == "raw":
            return observation
        elif mode == "rgb_array":
            if self.sprites is None:
                raise RuntimeError(f"No sprites fo reder in {mode} mode.")
            return observation_as_image(
                self.sprites, as_grid_3d(observation), last_move
            )
        elif mode == "human":
            if self.viewer is None:
                raise RuntimeError(
                    "ImageViewer is undefined. Likely cause: pyglget import failure."
                )
            if self.sprites is None:
                raise RuntimeError(f"No sprites fo reder in {mode} mode.")

            self.viewer.imshow(
                observation_as_image(self.sprites, as_grid_3d(observation), last_move)
            )
            if isinstance(caption, str):
                self.viewer.window.set_caption(caption)
            return self.viewer.isopen
        elif mode == "ansi":
            return observation_as_string(as_grid_3d(observation), last_move)
        else:
            raise RuntimeError(
                f"Unknown mode: {mode}. Exepcted of one: {self.metadata['render.modes']}"
            )

    def close(self):
        """
        Closes the viewer if initialized.
        """
        if self.viewer:
            self.viewer.close()


class Sprites(abc.ABC):
    """
    Interface for sprites.
    """

    @property
    @abc.abstractmethod
    def cliff_sprite(self):
        """
        Returns sprite for cliff.
        """
        del self
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def path_sprite(self):
        """
        Returns sprite for clear path.
        """
        del self
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def actor_sprite(self):
        """
        Returns sprites for actor.
        """
        del self
        raise NotImplementedError


class BlueMoonSprites(Sprites):
    """
    Implements strips with blue moon as main character.
    """

    def __init__(self, assets_dir: str):
        self._cliff_sprite = image_as_array(
            image.open(os.path.join(assets_dir, CLIFF_BG))
        )
        self._path_sprite = image_as_array(
            image.open(os.path.join(assets_dir, PATH_BG))
        )
        self._actor_sprite = image_as_array(image.open(os.path.join(assets_dir, ACTOR)))

    @property
    def cliff_sprite(self):
        return self._cliff_sprite

    @property
    def path_sprite(self):
        return self._path_sprite

    @property
    def actor_sprite(self):
        return self._actor_sprite


def parse_grid_from_text(grid: Sequence[str]):
    """
    Parses grid from text files.
    """
    cliffs = []
    exits = []
    start = None
    height, width = 0, 0
    for pos_x, line in enumerate(grid):
        row = line.strip()
        width = max(width, len(row))
        for pos_y, elem in enumerate(row):
            if elem.lower() == "x":
                cliffs.append((pos_x, pos_y))
            elif elem.lower() == "g":
                exits.append((pos_x, pos_y))
            elif elem.lower() == "s":
                start = (pos_x, pos_y)
        height += 1
    return (height, width), cliffs, exits, start


def states_mapping(
    size: Tuple[int, int], cliffs: Sequence[Tuple[int, int]]
) -> Mapping[Tuple[int, int], int]:
    """
    Creates a mapping of state IDs to grid positions.

    Args:
       size: the grid size, width x height.
       cliffs: the position of cliffs on the grid.
    Returns:
        A mapping grid positions to state ID.
    """
    states = []
    height, width = size
    for pos_x in range(height):
        for pos_y in range(width):
            state = (pos_x, pos_y)
            if state not in set(cliffs):
                states.append(state)
    return {value: key for key, value in enumerate(sorted(states))}


def apply_action(
    observation: Mapping[str, Any],
    action: int,
    get_state_id: Callable[[Tuple[int, int]], int],
) -> Tuple[Any, float]:
    """
    One step transition of the MDP.

    Args:
        observation: the current state.
        action: the agent's decision.

    Returns:
        The reward for the action and new state.
    """

    next_pos = _step(observation, action)
    reward = _step_reward(observation, next_position=next_pos)
    if next_pos in observation[OBS_KEY_CLIFFS]:
        # send back to the beginning
        next_pos = observation[OBS_KEY_START]
    next_observation = dict(observation)
    next_observation[OBS_KEY_AGENT] = next_pos
    next_observation[OBS_KEY_ID] = get_state_id(next_pos)
    return next_observation, reward


def _step(observation: Mapping[str, Any], action: int) -> Tuple[int, int]:
    # If in exit, stay
    if observation[OBS_KEY_AGENT] in observation[OBS_KEY_EXITS]:
        pos: Tuple[int, int] = observation[OBS_KEY_AGENT]
        return pos
    pos_x, pos_y = observation[OBS_KEY_AGENT]
    height, width = observation[OBS_KEY_SIZE]
    if action == LEFT:
        pos_y = max(pos_y - 1, 0)
    elif action == RIGHT:
        pos_y = min(pos_y + 1, width - 1)
    elif action == UP:
        pos_x = max(pos_x - 1, 0)
    elif action == DOWN:
        pos_x = min(pos_x + 1, height - 1)
    return (pos_x, pos_y)


def _step_reward(
    observation: Mapping[str, Any], next_position: Tuple[int, int]
) -> float:
    # terminal state (current pos)
    if observation[OBS_KEY_AGENT] in observation[OBS_KEY_EXITS]:
        return TERMINAL_REWARD
    if next_position in observation[OBS_KEY_CLIFFS]:
        return CLIFF_PENALTY
    return MOVE_PENALTY


def create_observation(
    size: Tuple[int, int],
    start: Tuple[int, int],
    agent: Tuple[int, int],
    cliffs: Sequence[Tuple[int, int]],
    exits: Sequence[Tuple[int, int]],
    get_state_id: Callable[[Tuple[int, int]], int],
) -> Mapping[str, Any]:
    """
    Creates an observation representation - a mapping of the layers of the grid,
    and other information.
    """
    return {
        OBS_KEY_ID: get_state_id(start),
        OBS_KEY_START: start,
        OBS_KEY_AGENT: agent,
        OBS_KEY_CLIFFS: cliffs,
        OBS_KEY_EXITS: exits,
        OBS_KEY_SIZE: size,
    }


def validate_dimensions(
    size: Tuple[int, int],
    start: Tuple[int, int],
    cliffs: Sequence[Tuple[int, int]],
    exits: Sequence[Tuple[int, int]],
) -> None:
    """
    Verifies the coordinates are sensible:
      - There are no overlaps between items in different layers.
    """
    height, width = size
    start_x, start_y = start
    for pos_x, pos_y in cliffs:
        if not (0 <= pos_x <= height):
            raise ValueError(
                f"Cliff has invalid coordinates, ({pos_x}, _), limits [0, {height})"
            )
        if not (0 <= pos_y <= width):
            raise ValueError(
                f"Cliff has invalid coordinates, (_, {pos_y}), limits [0, {width})"
            )

    for pos_x, pos_y in exits:
        if not (0 <= pos_x <= height):
            raise ValueError(
                f"Exit has invalid coordinates, ({pos_x}, _), limits [0, {height})"
            )
        if not (0 <= pos_y <= width):
            raise ValueError(
                f"Exit has invalid coordinates, (_, {pos_y}), limits [0, {width})"
            )

    if not (0 <= start_x <= height):
        raise ValueError(
            f"Starting position has invalid coordinates, ({start_x}, _), limits [0, {height})"
        )
    if not (0 <= start_y <= width):
        raise ValueError(
            f"Starting position has invalid coordinates, (_, {start_y}), limits [0, {width})"
        )


def validate_starting_grid(
    start: Tuple[int, int],
    cliffs: Sequence[Tuple[int, int]],
    exits: Sequence[Tuple[int, int]],
) -> None:
    """
    Verifies starting grid is sensible - there is no overlap between elements.
    """
    for cliff in cliffs:
        if cliff == start:
            raise ValueError(f"Starting on a cliff: ({cliff})")

        for exit_ in exits:
            if cliff == exit_:
                raise ValueError(f"Exit and cliff overlap: ({cliff})")

            # this only needs to run once really
            if exit_ == start:
                raise ValueError(f"Starting on an exit: ({exit_})")


def create_obs_state_id_fn(
    states: Mapping[Tuple[int, int], int],
) -> Callable[[Tuple[int, int]], int]:
    """
    Creates a function that returns an integer state ID for a given observation.

    Args:
        states: A dictionary that maps observations to state IDs.
    Returns:
        A callable that takes an observation and returns a state ID.
    """

    def obs_to_state_id(agent_position: Tuple[int, int]) -> int:
        return states[agent_position]

    return obs_to_state_id


def create_position_from_state_id_fn(
    states: Mapping[Tuple[int, int], int],
) -> Callable[[int], Tuple[int, int]]:
    """
    Creates a function that maps a state ID to a grid (x, y) coordinates.
    """

    rev_state = {value: key for key, value in states.items()}

    def position_from_state_id(state: int) -> Tuple[int, int]:
        """
        Given a state ID, computes the grid coordinates.

        Args:
            state: an integer state ID.

        Returns:
            Grid (x, y) position.
        """
        return rev_state[state]

    return position_from_state_id


def as_grid_3d(observation: Mapping[str, Any]) -> np.ndarray:
    """
    Creates a 3D array representation of the grid, with 1s where
    there is an item of the layer and 0 otherwise.

    Args:
        observation: the state observation.
    Returns:
        A stack of 2D grids, with binary flags to indicate the presence layer elements.
    """

    agent = np.zeros(shape=observation[OBS_KEY_SIZE], dtype=np.int64)
    cliff = np.zeros(shape=observation[OBS_KEY_SIZE], dtype=np.int64)
    exit_ = np.zeros(shape=observation[OBS_KEY_SIZE], dtype=np.int64)
    # Place agent at the start.
    # There is only one (x, y) pair.
    agent[observation[OBS_KEY_AGENT]] = 1
    for pos_x, pos_y in observation[OBS_KEY_CLIFFS]:
        cliff[pos_x, pos_y] = 1
    for pos_x, pos_y in observation[OBS_KEY_EXITS]:
        exit_[pos_x, pos_y] = 1
    return np.stack([agent, cliff, exit_])


def image_as_array(img: image.Image) -> np.ndarray:
    """
    Creates an array to represent the image.
    """
    array = np.array(img)
    if len(array.shape) == 3 and array.shape[2] == 4:
        return array[:, :, :3]
    return array


def observation_as_image(
    sprites: Sprites,
    grid_3d: np.ndarray,
    last_move: Optional[Any],
) -> np.ndarray:
    """
    Converts an observation to an image (matrix).
    """
    _, height, width = grid_3d.shape
    rows = []
    for pos_x in range(height):
        row = []
        for pos_y in range(width):
            pos = position_as_string(grid_3d, pos_x, pos_y, last_move)
            if pos in set(["[X]", "[x̄]"]):
                row.append(sprites.cliff_sprite)
            elif pos in set(["[S]"] + [f"[{move}]" for move in MOVES]):
                row.append(sprites.actor_sprite)
            # TODO: add sprite for exits
            else:
                row.append(sprites.path_sprite)
            # vertical border; same size as sprite
            if pos_y < width - 1:
                row.append(_vborder(size=row[-1].shape[0]))
        rows.append(np.hstack(row))
        # horizontal border
        rows.append(_hborder(size=sum(sprite.shape[1] for sprite in row)))
    return np.vstack(rows)


def _vborder(size: int) -> np.ndarray:
    if size < 1:
        raise ValueError("vertical border size must be positve")
    return np.array([[COLOR_SILVER]] * size, dtype=np.uint8)


def _hborder(size: int) -> np.ndarray:
    if size < 1:
        raise ValueError("horizontal border size must be positive")
    return np.array([[COLOR_SILVER] * size], dtype=np.uint8)


def observation_as_string(grid_3d: np.ndarray, last_move: Optional[Any]) -> str:
    """
    Converts an observation to a string.
    """
    _, height, width = grid_3d.shape
    out = io.StringIO()
    for pos_x in range(height):
        for pos_y in range(width):
            out.write(position_as_string(grid_3d, pos_x, pos_y, last_move))
        out.write("\n")

    out.write("\n\n")

    with contextlib.closing(out):
        sys.stdout.write(out.getvalue())
        return out.getvalue()


def position_as_string(
    grid_3d: np.ndarray, pos_x: int, pos_y: int, last_move: Optional[Any]
) -> str:
    """
    Given a position:

     - If the agent is in a starting position -
        returns the representation for starting position - S
     - If the agent is on the position and they made a move to get there,
        returns the representation of the move they made: L, R, U, D
     - If there is no agent, and it's a cliff, returns the X
     - If the agent is on the cliff returns x̄ (x-bar)
     - If there is no agent and it's a safe zone, returns " "
     - If there is no agent on an exit, returns E
     - If the agent is on an exit, returns Ē

    """
    if grid_3d[LAYER_AGENT, pos_x, pos_y] and grid_3d[LAYER_CLIFF, pos_x, pos_y]:
        return "[x̄]"
    elif grid_3d[LAYER_AGENT, pos_x, pos_y] and grid_3d[LAYER_EXIT, pos_x, pos_y]:
        return "[Ē]"
    elif grid_3d[LAYER_CLIFF, pos_x, pos_y] == 1:
        return "[X]"
    elif grid_3d[LAYER_EXIT, pos_x, pos_y] == 1:
        return "[E]"
    elif grid_3d[LAYER_AGENT, pos_x, pos_y] == 1:
        if last_move is not None:
            return f"[{MOVES[last_move]}]"
        return "[S]"
    return "[ ]"
