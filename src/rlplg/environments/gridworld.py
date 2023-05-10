"""
The objective of the environment is simple: from a starting point, with a final state.
There can be cliffs, and if the player goes to a cliff, the reward is -100,
and they go back to the starting position.
The player can go up, down, left and right.
If an action takes the player outside the grid, they stay in the same position.
The reward for every action is -1.

In the current implementation, the observation provides no information about
cliffs, the starting point, or exit.
So an agent needs to learn the value of every state from scratch.
"""

import abc
import base64
import contextlib
import copy
import hashlib
import io
import os
import os.path
import sys
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from gymnasium import spaces
from PIL import Image as image

from rlplg import core, envdesc, envspec, npsci
from rlplg.core import InitState, RenderType, TimeStep
from rlplg.learning.tabular import markovdp

ENV_NAME = "GridWorld"
CLIFF_PENALTY = -100.0
MOVE_PENALTY = -1.0
TERMINAL_REWARD = 0.0
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
MOVES = ["L", "R", "U", "D"]
MOVE_SYMBOLS = ["←", "→", "↑", "↓"]
COLOR_SILVER = (192, 192, 192)
PATH_BG = "path-bg.png"
CLIFF_BG = "cliff-bg.png"
ACTOR = "actor.png"


class Layers:
    """
    Layers spec.
    """

    player = 0
    cliff = 1
    exit = 2


class Strings:
    """
    Env keys.
    """

    size = "size"
    player = "player"
    cliffs = "cliffs"
    exits = "exits"
    start = "start"


class GridWorld(core.PyEnvironment):
    """
    GridWorld environment.
    """

    _num_layers = 3

    def __init__(
        self,
        size: Tuple[int, int],
        cliffs: Sequence[Tuple[int, int]],
        exits: Sequence[Tuple[int, int]],
        start: Tuple[int, int],
        render_mode: str = "rgb_array",
    ):
        super().__init__()
        assert_dimensions(size=size, start=start, cliffs=cliffs, exits=exits)
        assert_starting_grid(start=start, cliffs=cliffs, exits=exits)
        self.render_mode = render_mode
        self._height, self._width = size
        self._size = size
        self._start = start
        self._cliffs = set(cliffs)
        self._exits = set(exits)

        # left, right, up, down
        self.action_space = spaces.Box(low=0, high=3, dtype=np.int64)
        self.observation_space = spaces.Dict(
            {
                "start": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self._height - 1, self._width - 1]),
                    dtype=np.int64,
                ),
                "player": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self._height - 1, self._width - 1]),
                    dtype=np.int64,
                ),
                "cliffs": spaces.Box(
                    low=np.array([0, 0, 0]),
                    high=np.array([0, self._height - 1, self._width - 1]),
                    dtype=np.int64,
                ),
                "exits": spaces.Box(
                    low=np.array([0, 0, 0]),
                    high=np.array([0, self._height - 1, self._width - 1]),
                    dtype=np.int64,
                ),
                "size": spaces.Box(
                    low=np.array([self._height - 1, self._width - 1]),
                    high=np.array([self._height - 1, self._width - 1]),
                    dtype=np.int64,
                ),
            }
        )

        # env specific
        self._observation: Mapping[str, Any] = {}
        self._seed: Optional[int] = None

    def _step(self, action: Any) -> TimeStep:
        """Updates the environment according to action and returns a `TimeStep`.

        See `step(self, action)` docstring for more details.

        Args:
            action: A policy's chosen action.
        """
        if self._observation == {}:
            raise RuntimeError(
                f"{type(self).__name__} environment needs to be reset. Call the `reset` method."
            )
        next_observation, reward = apply_action(self._observation, action)
        self._observation = next_observation
        finished = coord_from_array(self._observation[Strings.player]) in self._exits
        return copy.deepcopy(self._observation), reward, finished, False, {}

    def _reset(self) -> InitState:
        """Starts a new sequence, returns the `InitState` for this environment.

        See `reset(self)` docstring for more details
        """
        # place agent at the start
        self._observation = create_observation(
            size=self._size,
            start=self._start,
            player=self._start,
            cliffs=tuple(self._cliffs),
            exits=tuple(self._exits),
        )
        return copy.deepcopy(self._observation), {}

    def _render(self) -> RenderType:
        if self._observation == {}:
            raise RuntimeError(
                f"{type(self).__name__} environment needs to be reset. Call the `reset` method."
            )
        if self.render_mode == "rgb_array":
            return as_grid(self._observation)
        return super()._render()

    def seed(self, seed: Optional[int] = None) -> Any:
        if seed is not None:
            self._seed = seed
            np.random.seed(seed)
        return self._seed


class GridWorldMdpDiscretizer(markovdp.MdpDiscretizer):
    """
    Creates an environment discrete maps for states and actions.
    """

    def __init__(
        self,
        size: Tuple[int, int],
        cliffs: Sequence[Tuple[int, int]],
    ):
        _states_mapping = states_mapping(size=size, cliffs=cliffs)
        self.__state_fn = create_state_id_fn(states=_states_mapping)

    def state(self, observation: Any) -> int:
        """
        Maps an observation to a state ID.
        """
        return self.__state_fn(observation)

    def action(self, action: Any) -> int:
        """
        Maps an agent action to an action ID.
        """
        del self
        action_: int = npsci.item(action)
        return action_


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
        observation: np.ndarray,
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
        assert mode in self.metadata["render.modes"]
        if sleep is not None and sleep > 0:
            time.sleep(sleep)
        if mode == "raw":
            return observation
        elif mode == "rgb_array":
            if self.sprites is None:
                raise RuntimeError(f"No sprites fo reder in {mode} mode.")
            return observation_as_image(self.sprites, observation, last_move)
        elif mode == "human":
            if self.viewer is None:
                raise RuntimeError(
                    "ImageViewer is undefined. Likely cause: pyglget import failure."
                )
            if self.sprites is None:
                raise RuntimeError(f"No sprites fo reder in {mode} mode.")

            self.viewer.imshow(
                observation_as_image(self.sprites, observation, last_move)
            )
            if isinstance(caption, str):
                self.viewer.window.set_caption(caption)
            return self.viewer.isopen
        elif mode == "ansi":
            return observation_as_string(observation, last_move)
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


def create_env_spec(
    size: Tuple[int, int],
    cliffs: Sequence[Tuple[int, int]],
    exits: Sequence[Tuple[int, int]],
    start: Tuple[int, int],
) -> envspec.EnvSpec:
    """
    Creates an env spec from a gridworld config.
    """
    environment = GridWorld(size=size, cliffs=cliffs, exits=exits, start=start)
    discretizer = GridWorldMdpDiscretizer(size=size, cliffs=cliffs)
    height, width = size
    num_states = height * width - len(cliffs)
    num_actions = len(MOVES)
    env_desc = envdesc.EnvDesc(num_states=num_states, num_actions=num_actions)
    return envspec.EnvSpec(
        name=ENV_NAME,
        level=__encode_env(size=size, cliffs=cliffs, exits=exits, start=start),
        environment=environment,
        discretizer=discretizer,
        env_desc=env_desc,
    )


def __encode_env(
    size: Tuple[int, int],
    cliffs: Sequence[Tuple[int, int]],
    exits: Sequence[Tuple[int, int]],
    start: Tuple[int, int],
) -> str:
    hash_key = (size, cliffs, exits, start)
    hashing = hashlib.sha512(str(hash_key).encode("UTF-8"))
    return base64.b32encode(hashing.digest()).decode("UTF-8")


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


def apply_action(observation: Any, action: Any) -> Tuple[Any, float]:
    """
    One step transition of the MDP.

    Args:
        observation: the current state.
        action: the agent's decision.

    Returns:
        The reward for the action and new state.
    """

    next_position = _step(observation, action)
    reward = _step_reward(observation, next_position=next_position)
    if next_position in coords_from_sequence(observation[Strings.cliffs]):
        # send back to the beginning
        next_position = coord_from_array(observation[Strings.start])
    next_observation = copy.deepcopy(observation)
    next_observation[Strings.player] = np.array(next_position, dtype=np.int64)
    return next_observation, reward


def _step(observation: Any, action: Any) -> Tuple[int, int]:
    # If in exit, stay
    if coord_from_array(observation[Strings.player]) in coords_from_sequence(
        observation[Strings.exits]
    ):
        pos: np.ndarray = copy.deepcopy(observation[Strings.player])
        return coord_from_array(pos)
    pos_x, pos_y = observation[Strings.player]
    height, width = observation[Strings.size]
    if action == LEFT:
        pos_y = max(pos_y - 1, 0)
    elif action == RIGHT:
        pos_y = min(pos_y + 1, width - 1)
    elif action == UP:
        pos_x = max(pos_x - 1, 0)
    elif action == DOWN:
        pos_x = min(pos_x + 1, height - 1)
    return (pos_x, pos_y)


def _step_reward(observation: Any, next_position: Tuple[int, int]) -> float:
    # terminal state (current pos)
    if coord_from_array(observation[Strings.player]) in coords_from_sequence(
        observation[Strings.exits]
    ):
        return TERMINAL_REWARD
    if next_position in coords_from_sequence(observation[Strings.cliffs]):
        return CLIFF_PENALTY
    return MOVE_PENALTY


def create_observation(
    size: Tuple[int, int],
    start: Tuple[int, int],
    player: Tuple[int, int],
    cliffs: Sequence[Tuple[int, int]],
    exits: Sequence[Tuple[int, int]],
) -> Mapping[str, Any]:
    """
    Creates an observation representation - a mapping of the layers of the grid,
    and other information.
    """
    return {
        Strings.start: np.array(start, dtype=np.int64),
        Strings.player: np.array(player, dtype=np.int64),
        Strings.cliffs: np.array(cliffs, dtype=np.int64),
        Strings.exits: np.array(exits, dtype=np.int64),
        Strings.size: np.array(size, dtype=np.int64),
    }


def assert_dimensions(
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
        assert (
            0 <= pos_x <= height
        ), f"Cliff has invalid coordinates, ({pos_x}, _), limits [0, {height})"
        assert (
            0 <= pos_y <= width
        ), f"Cliff has invalid coordinates, (_, {pos_y}), limits [0, {width})"

    for pos_x, pos_y in exits:
        assert (
            0 <= pos_x <= height
        ), f"Exit has invalid coordinates, ({pos_x}, _), limits [0, {height})"
        assert (
            0 <= pos_y <= width
        ), f"Exit has invalid coordinates, (_, {pos_y}), limits [0, {width})"

    assert (
        0 <= start_x <= height
    ), f"Starting position has invalid coordinates, ({start_x}, _), limits [0, {height})"
    assert (
        0 <= start_y <= width
    ), f"Starting position has invalid coordinates, (_, {start_y}), limits [0, {width})"


def assert_starting_grid(
    start: Tuple[int, int],
    cliffs: Sequence[Tuple[int, int]],
    exits: Sequence[Tuple[int, int]],
) -> None:
    """
    Verifies starting grid is sensible - there is no overlap between elements.
    """
    for cliff in cliffs:
        assert cliff != start, f"Starting on a cliff: ({cliff})"

        for exit_ in exits:
            assert cliff != exit_, f"Exit and cliff overlap: ({cliff})"
            # this only needs to run once really
            assert exit_ != start, f"Starting on an exit: ({exit_})"


def create_state_id_fn(
    states: Mapping[Tuple[int, int], int]
) -> Callable[[Mapping[str, Any]], int]:
    """
    Creates a function that returns an integer state ID for a given observation.

    Args:
        states: A dictionary that maps observations to state IDs.
    Returns:
        A callable that takes an observation and returns a state ID.
    """

    def state_id(observation: Mapping[str, Any]) -> int:
        """
        A function that takes an observation and returns a state ID.

        Args:
            observation: a state observation.
        Returns:
            An integer state ID.
        """
        return states[coord_from_array(observation[Strings.player])]

    return state_id


def create_position_from_state_id_fn(
    states: Mapping[Tuple[int, int], int]
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


def as_grid(observation: Mapping[str, Any]) -> np.ndarray:
    """
    Creates a 3D array representation of the grid, with 1s where
    there is an item of the layer and 0 otherwise.

    Args:
        observation: the state observation.
    Returns:
        A stack of 2D grids, with binary flags to indicate the presence layer elements.
    """

    player = np.zeros(shape=observation[Strings.size], dtype=np.int64)
    cliff = np.zeros(shape=observation[Strings.size], dtype=np.int64)
    exit_ = np.zeros(shape=observation[Strings.size], dtype=np.int64)
    # Place agent at the start.
    # There is only one (x, y) pair.
    player[coord_from_array(observation[Strings.player])] = 1
    for pos_x, pos_y in observation[Strings.cliffs]:
        cliff[pos_x, pos_y] = 1
    for pos_x, pos_y in observation[Strings.exits]:
        exit_[pos_x, pos_y] = 1
    return np.stack([player, cliff, exit_])


def coord_from_array(array: np.ndarray) -> Tuple[int, int]:
    """
    Converts a coordinate from an arry to a 2-tuple.
    """
    coord_x, coord_y = array.tolist()
    return coord_x, coord_y


def coords_from_sequence(array: np.ndarray) -> Sequence[Tuple[int, int]]:
    """
    Converts a sequence of coordinates from an 2-D array to a sequence of 2-tuples.
    """
    return [coord_from_array(element) for element in array]


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
    observation: np.ndarray,
    last_move: Optional[Any],
) -> np.ndarray:
    _, height, width = observation.shape
    rows = []
    for x in range(height):
        row = []
        for y in range(width):
            pos = position_as_string(observation, x, y, last_move)
            if pos in set(["[X]", "[x̄]"]):
                row.append(sprites.cliff_sprite)
            elif pos in set(["[S]"] + [f"[{move}]" for move in MOVES]):
                row.append(sprites.actor_sprite)
            # TODO: add sprite for exits
            else:
                row.append(sprites.path_sprite)
            # vertical border; same size as sprite
            if y < width - 1:
                row.append(_vborder(size=row[-1].shape[0]))
        rows.append(np.hstack(row))
        # horizontal border
        rows.append(_hborder(size=sum([sprite.shape[1] for sprite in row])))
    return np.vstack(rows)


def _vborder(size: int) -> np.ndarray:
    assert size > 0, "vertical border size must be positve"
    return np.array([[COLOR_SILVER]] * size, dtype=np.uint8)


def _hborder(size: int) -> np.ndarray:
    assert size > 0, "horizontal border size must be positive"
    return np.array([[COLOR_SILVER] * size], dtype=np.uint8)


def observation_as_string(observation: np.ndarray, last_move: Optional[Any]) -> str:
    _, height, width = observation.shape
    out = io.StringIO()
    for x in range(height):
        for y in range(width):
            out.write(position_as_string(observation, x, y, last_move))
        out.write("\n")

    out.write("\n\n")

    with contextlib.closing(out):
        sys.stdout.write(out.getvalue())
        return out.getvalue()


def position_as_string(
    observation: np.ndarray, x: int, y: int, last_move: Optional[Any]
) -> str:
    """
    Given a position:

     - If the player is in a starting position -
        returns the representation for starting position - S
     - If the player is on the position and they made a move to get there,
        returns the representation of the move they made: L, R, U, D
     - If there is no player, and it's a cliff, returns the X
     - If the player is on the cliff returns x̄ (x-bar)
     - If there is no player and it's a safe zone, returns " "
     - If there is no player on an exit, returns E
     - If the player is on an exit, returns Ē

    """
    if observation[Layers.player, x, y] and observation[Layers.cliff, x, y]:
        return "[x̄]"
    elif observation[Layers.player, x, y] and observation[Layers.exit, x, y]:
        return "[Ē]"
    elif observation[Layers.cliff, x, y] == 1:
        return "[X]"
    elif observation[Layers.exit, x, y] == 1:
        return "[E]"
    elif observation[Layers.player, x, y] == 1:
        if last_move is not None:
            return f"[{MOVES[last_move]}]"
        return "[S]"
    return "[ ]"


def parse_grid(path: str):
    """
    Parses grid from text files.
    """

    cliffs = []
    exits = []
    start = None
    height, width = 0, 0
    with tf.io.gfile.GFile(path, "r") as reader:
        for x, line in enumerate(reader):
            row = line.strip()
            width = max(width, len(row))
            for y, elem in enumerate(row):
                if elem.lower() == "x":
                    cliffs.append((x, y))
                elif elem.lower() == "g":
                    exits.append((x, y))
                elif elem.lower() == "s":
                    start = (x, y)
            height += 1
    return (height, width), cliffs, exits, start


def create_environment_from_grid(path: str) -> GridWorld:
    """
    Parses a grid file and create an environment from
    the parameters.
    """
    size, cliffs, exits, start = parse_grid(path)
    return GridWorld(size=size, cliffs=cliffs, exits=exits, start=start)


def create_envspec_from_grid(grid_path: str) -> envspec.EnvSpec:
    """
    Parses a grid file and create an environment from
    the parameters.
    """
    size, cliffs, exits, start = parse_grid(grid_path)
    return create_env_spec(size=size, cliffs=cliffs, exits=exits, start=start)
