from typing import Sequence, Tuple

import numpy as np

GRID_WIDTH = 5
GRID_HEIGHT = 5
NUM_ACTIONS = 4
NUM_STATES = GRID_HEIGHT * GRID_WIDTH

CLIFF_COLOR = (25, 50, 75)
PATH_COLOR = (50, 75, 25)
ACTOR_COLOR = (75, 25, 50)
EXIT_COLOR = (255, 204, 0)


def grid(
    x: int,
    y: int,
    height: int = GRID_HEIGHT,
    width: int = GRID_WIDTH,
    cliffs: Sequence[Tuple[int, int]] = (),
    exits: Sequence[Tuple[int, int]] = (),
) -> np.ndarray:
    obs = np.zeros(shape=(3, height, width), dtype=np.int64)
    # player
    obs[0, x, y] = 1
    for x, y in cliffs:
        obs[1, x, y] = 1
    for x, y in exits:
        obs[2, x, y] = 1
    return obs


def empty_grid() -> np.ndarray:
    return np.zeros(shape=(3, GRID_HEIGHT, GRID_WIDTH), dtype=np.int64)


def color_block(
    color: Tuple[int, int, int], width: int = GRID_WIDTH, height: int = GRID_HEIGHT
) -> np.ndarray:
    return np.array([[color] * width] * height, dtype=np.uint8)
