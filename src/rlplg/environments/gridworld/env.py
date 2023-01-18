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

import base64
import copy
import hashlib
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing.types import NestedArray, NestedArraySpec, Seed

from rlplg import envdesc, envspec, npsci
from rlplg.environments.gridworld import constants
from rlplg.learning.tabular import markovdp


class GridWorld(py_environment.PyEnvironment):
    """
    GridWorld environment.
    """

    metadata = {"render.modes": ["rgb_array"]}
    _num_layers = 3

    def __init__(
        self,
        size: Tuple[int, int],
        cliffs: Sequence[Tuple[int, int]],
        exits: Sequence[Tuple[int, int]],
        start: Tuple[int, int],
    ):
        super().__init__()
        assert_dimensions(size=size, start=start, cliffs=cliffs, exits=exits)
        assert_starting_grid(start=start, cliffs=cliffs, exits=exits)
        self._height, self._width = size
        self._size = size
        self._start = start
        self._cliffs = set(cliffs)
        self._exits = set(exits)

        # left, right, up, down
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name="action"
        )
        self._observation_spec = {
            "start": array_spec.BoundedArraySpec(
                shape=(2,),
                dtype=np.int32,
                minimum=np.array(start),
                maximum=np.array(start),
                name="start",
            ),
            "player": array_spec.BoundedArraySpec(
                shape=(2,),
                dtype=np.int32,
                minimum=np.zeros(shape=(2,)),
                maximum=np.array([dim - 1 for dim in size]),
                name="player",
            ),
            "cliffs": array_spec.BoundedArraySpec(
                shape=(len(cliffs), 2),
                dtype=np.int32,
                # these aren't exact, just theoretical
                # for the dims of the grid
                minimum=np.zeros(shape=(2,)),
                maximum=np.array([dim - 1 for dim in size]),
                name="cliffs",
            ),
            "exits": array_spec.BoundedArraySpec(
                shape=(len(exits), 2),
                dtype=np.int32,
                # these aren't exact, just theoretical
                # for the dims of the grid
                minimum=np.zeros(shape=(2,)),
                maximum=np.array([dim - 1 for dim in size]),
                name="exits",
            ),
            "size": array_spec.BoundedArraySpec(
                shape=(2,),
                dtype=np.int32,
                minimum=np.array(size),
                maximum=np.array(size),
                name="exits",
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
        next_observation, reward = apply_action(self._observation, action)

        self._observation = next_observation
        if self._observation[constants.Strings.player] in self._exits:
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
        # place agent at the start
        self._observation = create_observation(
            size=self._size,
            start=self._start,
            player=self._start,
            cliffs=tuple(self._cliffs),
            exits=tuple(self._exits),
        )
        return ts.restart(observation=copy.deepcopy(self._observation))

    def render(self, mode="rgb_array") -> Optional[NestedArray]:
        if self._observation is None:
            raise RuntimeError(
                f"{type(self).__name__} environment needs to be reset. Call the `reset` method."
            )
        if mode == "rgb_array":
            return as_grid(self._observation)
        return super().render(mode)

    def seed(self, seed: Seed = None) -> Any:
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
    num_actions = len(constants.MOVES)
    env_desc = envdesc.EnvDesc(num_states=num_states, num_actions=num_actions)
    return envspec.EnvSpec(
        name=constants.ENV_NAME,
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


def apply_action(
    observation: NestedArray, action: NestedArray
) -> Tuple[NestedArray, float]:
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
    if next_position in observation[constants.Strings.cliffs]:
        # send back to the beginning
        next_position = observation[constants.Strings.start]
    next_observation = copy.deepcopy(observation)
    next_observation[constants.Strings.player] = next_position
    return next_observation, reward


def _step(observation: NestedArray, action: NestedArray) -> Tuple[int, int]:
    # If in exit, stay
    if observation[constants.Strings.player] in observation[constants.Strings.exits]:
        pos: Tuple[int, int] = copy.deepcopy(observation[constants.Strings.player])
        return pos
    pos_x, pos_y = observation[constants.Strings.player]
    height, width = observation[constants.Strings.size]
    if action == constants.LEFT:
        pos_y = max(pos_y - 1, 0)
    elif action == constants.RIGHT:
        pos_y = min(pos_y + 1, width - 1)
    elif action == constants.UP:
        pos_x = max(pos_x - 1, 0)
    elif action == constants.DOWN:
        pos_x = min(pos_x + 1, height - 1)
    return (pos_x, pos_y)


def _step_reward(observation: NestedArray, next_position: Tuple[int, int]) -> float:
    # terminal state (current pos)
    if observation[constants.Strings.player] in observation[constants.Strings.exits]:
        return constants.TERMINAL_REWARD
    if next_position in observation[constants.Strings.cliffs]:
        return constants.CLIFF_PENALTY
    return constants.MOVE_PENALTY


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
        constants.Strings.start: start,
        constants.Strings.player: player,
        constants.Strings.cliffs: set(cliffs),
        constants.Strings.exits: set(exits),
        constants.Strings.size: size,
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
        return states[observation[constants.Strings.player]]

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


def as_grid(observation: Mapping[str, Any]) -> NestedArray:
    """
    Creates a 3D array representation of the grid, with 1s where
    there is an item of the layer and 0 otherwise.

    Args:
        observation: the state observation.
    Returns:
        A stack of 2D grids, with binary flags to indicate the presence layer elements.
    """

    player = np.zeros(shape=observation[constants.Strings.size], dtype=np.int32)
    cliff = np.zeros(shape=observation[constants.Strings.size], dtype=np.int32)
    exit_ = np.zeros(shape=observation[constants.Strings.size], dtype=np.int32)
    # place agent at the start
    player[observation[constants.Strings.player]] = 1
    for pos_x, pos_y in observation[constants.Strings.cliffs]:
        cliff[pos_x, pos_y] = 1
    for pos_x, pos_y in observation[constants.Strings.exits]:
        exit_[pos_x, pos_y] = 1
    return np.stack([player, cliff, exit_])
