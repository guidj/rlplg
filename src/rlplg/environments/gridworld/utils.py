import tensorflow as tf

from rlplg import envspec
from rlplg.environments.gridworld import env


def parse_grid(path: str):
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


def create_environment_from_grid(path: str) -> env.GridWorld:
    """
    Parses a grid file and create an environment from
    the parameters.
    """
    size, cliffs, exits, start = parse_grid(path)
    return env.GridWorld(size=size, cliffs=cliffs, exits=exits, start=start)


def create_envspec_from_grid(grid_path: str) -> envspec.EnvSpec:
    """
    Parses a grid file and create an environment from
    the parameters.
    """
    size, cliffs, exits, start = parse_grid(grid_path)
    return env.create_env_spec(size=size, cliffs=cliffs, exits=exits, start=start)
