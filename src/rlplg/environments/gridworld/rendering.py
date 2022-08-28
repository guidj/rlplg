import abc
import contextlib
import io
import logging
import os
import os.path
import sys
import time
from typing import Any, Optional

import numpy as np
from PIL import Image as image
from tf_agents.typing.types import NestedArray

from rlplg.environments.gridworld import constants


class GridWorldRenderer:
    """
    Class for handling rendering of the environment
    """

    metadata = {"render.modes": ["raw", "rgb_array", "human", "ansi"]}

    def __init__(self, sprites_dir: Optional[str]):
        self.sprites = BlueMoonSprites(sprites_dir) if sprites_dir is not None else None
        self.viewer: Optional[Any] = None
        try:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.SimpleImageViewer()
        except ImportError as err:
            logging.error(err)
            logging.info("Proceeding without rendering")

    def render(
        self,
        mode: str,
        observation: np.ndarray,
        last_move: Optional[NestedArray],
        caption: Optional[str],
        sleep: Optional[float] = 0.05,
    ):
        """
        Args:
            mode: the rendering mode
            observation: the grid, as a numpy array of shape (W, H, 3)
            last_move: index of the last move, [0, 4)
            caption: string write on rendered Window
            sleep: time between rendering frames
        """
        assert mode in self.metadata["render.modes"]

        if mode == "raw":
            output = observation
        elif mode == "rgb_array":
            output = observation_as_image(self.sprites, observation, last_move)
        elif mode == "human":
            if self.viewer is None:
                raise RuntimeError(
                    "ImageViewer is undefined. Likely cause: pyglget import failure."
                )

            self.viewer.imshow(
                observation_as_image(self.sprites, observation, last_move)
            )
            if isinstance(caption, str):
                self.viewer.window.set_caption(caption)
            output = self.viewer.isopen
        elif mode == "ansi":
            output = observation_as_string(observation, last_move)
        else:
            raise RuntimeError(
                f"Unknown mode: {mode}. Exepcted of one: {self.metadata['render.modes']}"
            )
        if sleep is not None and sleep > 0:
            time.sleep(sleep)
        return output

    def close(self):
        if self.viewer:
            self.viewer.close()


class Sprites(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def cliff_sprite(self):
        del self
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def path_sprite(self):
        del self
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def actor_sprite(self):
        del self
        raise NotImplementedError


class BlueMoonSprites(Sprites):
    def __init__(self, assets_dir: str):
        self._cliff_sprite = image_as_array(
            image.open(os.path.join(assets_dir, constants.CLIFF_BG))
        )
        self._path_sprite = image_as_array(
            image.open(os.path.join(assets_dir, constants.PATH_BG))
        )
        self._actor_sprite = image_as_array(
            image.open(os.path.join(assets_dir, constants.ACTOR))
        )

    @property
    def cliff_sprite(self):
        return self._cliff_sprite

    @property
    def path_sprite(self):
        return self._path_sprite

    @property
    def actor_sprite(self):
        return self._actor_sprite


def image_as_array(img: image.Image) -> np.ndarray:
    array = np.array(img)
    if len(array.shape) == 3 and array.shape[2] == 4:
        return array[:, :, :3]
    return array


def observation_as_image(
    sprites: Sprites,
    observation: np.ndarray,
    last_move: Optional[NestedArray],
) -> np.ndarray:
    _, height, width = observation.shape
    rows = []
    for x in range(height):
        row = []
        for y in range(width):
            pos = position_as_string(observation, x, y, last_move)
            if pos in set(["[X]", "[x̄]"]):
                row.append(sprites.cliff_sprite)
            elif pos in set(["[S]"] + [f"[{move}]" for move in constants.MOVES]):
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
    return np.array([[constants.COLOR_SILVER]] * size, dtype=np.uint8)


def _hborder(size: int) -> np.ndarray:
    assert size > 0, "horizontal border size must be positive"
    return np.array([[constants.COLOR_SILVER] * size], dtype=np.uint8)


def observation_as_string(
    observation: np.ndarray, last_move: Optional[NestedArray]
) -> str:
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
    observation: np.ndarray, x: int, y: int, last_move: Optional[NestedArray]
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
    if (
        observation[constants.Layers.player, x, y]
        and observation[constants.Layers.cliff, x, y]
    ):
        return "[x̄]"
    elif (
        observation[constants.Layers.player, x, y]
        and observation[constants.Layers.exit, x, y]
    ):
        return "[Ē]"
    elif observation[constants.Layers.cliff, x, y] == 1:
        return "[X]"
    elif observation[constants.Layers.exit, x, y] == 1:
        return "[E]"
    elif observation[constants.Layers.player, x, y] == 1:
        if last_move is not None:
            return f"[{constants.MOVES[last_move]}]"
        return "[S]"
    return "[ ]"
