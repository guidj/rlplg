from typing import Sequence, Tuple

import numpy as np

WIDTH = 5
HEIGHT = 5
NUM_ACTIONS = 4
NUM_STATES = HEIGHT * WIDTH

CLIFF_COLOR = (25, 50, 75)
PATH_COLOR = (50, 75, 25)
ACTOR_COLOR = (75, 25, 50)
EXIT_COLOR = (255, 204, 0)


def grid(
    x: int,
    y: int,
    height: int = HEIGHT,
    width: int = WIDTH,
    cliffs: Sequence[Tuple[int, int]] = (),
    exits: Sequence[Tuple[int, int]] = (),
) -> np.ndarray:
    obs = np.zeros(shape=(3, height, width), dtype=np.int64)
    # agent
    obs[0, x, y] = 1
    for x, y in cliffs:
        obs[1, x, y] = 1
    for x, y in exits:
        obs[2, x, y] = 1
    return obs


def ice(
    x: int,
    y: int,
    height: int = HEIGHT,
    width: int = WIDTH,
    lakes: Sequence[Tuple[int, int]] = (),
    goals: Sequence[Tuple[int, int]] = (),
) -> np.ndarray:
    obs = np.zeros(shape=(3, height, width), dtype=np.int64)
    # agent
    obs[0, x, y] = 1
    for x, y in lakes:
        obs[1, x, y] = 1
    for x, y in goals:
        obs[2, x, y] = 1
    return obs


def empty_grid() -> np.ndarray:
    return np.zeros(shape=(3, HEIGHT, WIDTH), dtype=np.int64)


def color_block(
    color: Tuple[int, int, int], width: int = WIDTH, height: int = HEIGHT
) -> np.ndarray:
    return np.array([[color] * width] * height, dtype=np.uint8)
