from typing import Any, Iterable, Sequence, Tuple

import hypothesis
import numpy as np
import pytest
from hypothesis import strategies as st
from PIL import Image as image

from rlplg.environments.gridworld import constants, rendering
from tests.rlplg.environments.gridworld import grids


@hypothesis.given(
    size=st.tuples(
        st.integers(min_value=2, max_value=100),
        st.integers(min_value=2, max_value=100),
    ),
    color=st.tuples(
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
    ),
)
def test_image_as_array_rgb_images(size: Tuple[int, int], color: Tuple[int, int, int]):
    width, height = size
    array = np.array([[color] * width] * height, dtype=np.uint8)
    img = image.fromarray(array)
    np.testing.assert_equal(rendering.image_as_array(img), array)


@hypothesis.given(
    size=st.tuples(
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=1, max_value=100),
    ),
    color=st.tuples(
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
    ),
)
def test_image_as_array_on_rgba_images(
    size: Tuple[int, int], color: Tuple[int, int, int]
):
    img = image.new(mode="RGBA", size=size, color=color)
    np.testing.assert_equal(rendering.image_as_array(img), np.array(img)[:, :, :3])


def test_observation_as_window_simple_case(sprites: rendering.Sprites):
    obs = grids.grid(x=1, y=1, cliffs=[(4, 1), (4, 2), (4, 3)], exits=[(4, 4)])
    path = grids.color_block(grids.PATH_COLOR)
    cliff = grids.color_block(grids.CLIFF_COLOR)
    actor = grids.color_block(grids.ACTOR_COLOR)
    vsep = grids.color_block((192, 192, 192), width=1, height=grids.GRID_HEIGHT)
    hsep = grids.color_block((192, 192, 192), width=grids.GRID_WIDTH, height=1)
    dot = grids.color_block((192, 192, 192), width=1, height=1)

    elements = [
        [path, vsep, path, vsep, path, vsep, path, vsep, path],
        [
            hsep,
            dot,
            hsep,
            dot,
            hsep,
            dot,
            hsep,
            dot,
            hsep,
        ],
        [path, vsep, actor, vsep, path, vsep, path, vsep, path],
        [
            hsep,
            dot,
            hsep,
            dot,
            hsep,
            dot,
            hsep,
            dot,
            hsep,
        ],
        [path, vsep, path, vsep, path, vsep, path, vsep, path],
        [
            hsep,
            dot,
            hsep,
            dot,
            hsep,
            dot,
            hsep,
            dot,
            hsep,
        ],
        [path, vsep, path, vsep, path, vsep, path, vsep, path],
        [
            hsep,
            dot,
            hsep,
            dot,
            hsep,
            dot,
            hsep,
            dot,
            hsep,
        ],
        [path, vsep, cliff, vsep, cliff, vsep, cliff, vsep, path],
        [
            hsep,
            dot,
            hsep,
            dot,
            hsep,
            dot,
            hsep,
            dot,
            hsep,
        ],
    ]
    expected = np.vstack([np.hstack(row) for row in elements])

    output = rendering.observation_as_image(sprites, obs, last_move=None)

    np.testing.assert_equal(output, expected)


def test_hborder_with_unit_size():
    np.testing.assert_equal(rendering._hborder(1), [[[192, 192, 192]]])


def test_hborder_with_non_positive_size():
    with pytest.raises(AssertionError):
        rendering._hborder(0)


@hypothesis.given(size=st.integers(min_value=1, max_value=100))
def test_hborder_with_random_size(size: int):
    np.testing.assert_equal(rendering._hborder(size), [[[192, 192, 192]] * size])


def test_vborder_with_unit_size():
    np.testing.assert_equal(rendering._vborder(1), [[[192, 192, 192]]])


def test_vborder_with_non_positive_size():
    with pytest.raises(AssertionError):
        rendering._vborder(0)


@hypothesis.given(size=st.integers(min_value=1, max_value=100))
def test_vborder_with_random_size(size: int):
    np.testing.assert_equal(rendering._vborder(size), [[[192, 192, 192]]] * size)


def test_observation_as_string_with_empty_grid_and_no_last_move():
    obs = grids.empty_grid()
    expected = "\n".join(
        [
            "[ ][ ][ ][ ][ ]",
            "[ ][ ][ ][ ][ ]",
            "[ ][ ][ ][ ][ ]",
            "[ ][ ][ ][ ][ ]",
            "[ ][ ][ ][ ][ ]",
            "\n\n",
        ]
    )
    assert rendering.observation_as_string(obs, None) == expected


@hypothesis.given(
    last_move=st.integers(min_value=0, max_value=len(constants.MOVES) - 1),
)
def test_observation_as_string_with_empty_grid_and_last_move(last_move: int):
    obs = grids.empty_grid()
    expected = "\n".join(
        [
            "[ ][ ][ ][ ][ ]",
            "[ ][ ][ ][ ][ ]",
            "[ ][ ][ ][ ][ ]",
            "[ ][ ][ ][ ][ ]",
            "[ ][ ][ ][ ][ ]",
            "\n\n",
        ]
    )
    assert rendering.observation_as_string(obs, last_move) == expected


@hypothesis.given(
    x=st.integers(min_value=0, max_value=grids.GRID_HEIGHT - 1),
    y=st.integers(min_value=0, max_value=grids.GRID_WIDTH - 1),
)
def test_observation_as_string_with_cliffs_and_no_last_move(x: int, y: int):
    obs = grids.empty_grid()
    obs[1, x, y] = 1
    expected = [
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["\n\n"],
    ]
    expected[x][y] = "[X]"
    assert rendering.observation_as_string(obs, None) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=grids.GRID_HEIGHT - 1),
    y=st.integers(min_value=0, max_value=grids.GRID_WIDTH - 1),
)
def test_observation_as_string_with_exits(x: int, y: int):
    obs = grids.empty_grid()
    obs[2, x, y] = 1
    expected = [
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["\n\n"],
    ]
    expected[x][y] = "[E]"
    assert rendering.observation_as_string(obs, None) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=grids.GRID_HEIGHT - 1),
    y=st.integers(min_value=0, max_value=grids.GRID_WIDTH - 1),
)
def test_observation_as_string_with_player_and_no_last_move(x: int, y: int):
    obs = grids.grid(x, y)
    obs[0, x, y] = 1
    expected = [
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["\n\n"],
    ]
    expected[x][y] = "[S]"
    assert rendering.observation_as_string(obs, None) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=grids.GRID_HEIGHT - 1),
    y=st.integers(min_value=0, max_value=grids.GRID_WIDTH - 1),
    last_move=st.integers(min_value=0, max_value=len(constants.MOVES) - 1),
)
def test_observation_as_string_with_player_and_last_move(
    x: int,
    y: int,
    last_move: int,
):
    obs = grids.grid(x, y)
    expected = [
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["\n\n"],
    ]
    expected[x][y] = f"[{constants.MOVES[last_move]}]"
    assert rendering.observation_as_string(obs, last_move) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=grids.GRID_HEIGHT - 1),
    y=st.integers(min_value=0, max_value=grids.GRID_WIDTH - 1),
)
def test_observation_as_string_with_player_in_cliff_and_no_last_move(
    x: int,
    y: int,
):
    obs = grids.grid(x, y, cliffs=[(x, y)])
    expected = [
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["\n\n"],
    ]
    expected[x][y] = "[x̄]"
    assert rendering.observation_as_string(obs, None) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=grids.GRID_HEIGHT - 1),
    y=st.integers(min_value=0, max_value=grids.GRID_WIDTH - 1),
    last_move=st.integers(min_value=0, max_value=len(constants.MOVES) - 1),
)
def test_observation_as_string_with_player_in_cliff_and_last_move(
    x: int, y: int, last_move: int
):
    obs = grids.grid(x, y, cliffs=[(x, y)])
    expected = [
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["\n\n"],
    ]
    expected[x][y] = "[x̄]"
    assert rendering.observation_as_string(obs, last_move) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=grids.GRID_HEIGHT - 1),
    y=st.integers(min_value=0, max_value=grids.GRID_WIDTH - 1),
)
def test_observation_as_string_with_player_in_exit_and_no_last_move(
    x: int,
    y: int,
):
    obs = grids.grid(x, y, exits=[(x, y)])
    expected = [
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["\n\n"],
    ]
    expected[x][y] = "[Ē]"
    assert rendering.observation_as_string(obs, None) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=grids.GRID_HEIGHT - 1),
    y=st.integers(min_value=0, max_value=grids.GRID_WIDTH - 1),
    last_move=st.integers(min_value=0, max_value=len(constants.MOVES) - 1),
)
def test_observation_as_string_with_player_in_exit_and_last_move(
    x: int, y: int, last_move: int
):
    obs = grids.grid(x, y, exits=[(x, y)])
    expected = [
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["\n\n"],
    ]
    expected[x][y] = "[Ē]"
    assert rendering.observation_as_string(obs, last_move) == grid_as_string(expected)


def test_position_as_string_in_starting_position_with_player_and_no_last_move():
    x, y = grids.GRID_HEIGHT - 1, 0
    obs = grids.grid(x, y)
    assert rendering.position_as_string(obs, x, y, None) == "[S]"


@hypothesis.given(
    x=st.integers(min_value=0, max_value=grids.GRID_HEIGHT - 1),
    y=st.integers(min_value=0, max_value=grids.GRID_WIDTH - 1),
)
def test_position_as_string_on_safe_position_with_player_and_no_last_move(
    x: int, y: int
):
    obs = grids.grid(x, y)
    assert rendering.position_as_string(obs, x, y, None) == "[S]"


@hypothesis.given(
    x=st.integers(min_value=0, max_value=grids.GRID_HEIGHT - 1),
    y=st.integers(min_value=0, max_value=grids.GRID_WIDTH - 1),
    last_move=st.integers(min_value=0, max_value=len(constants.MOVES) - 1),
)
def test_position_as_string_on_safe_position_with_player_and_last_move(
    x: int, y: int, last_move: int
):
    obs = grids.grid(x, y)
    assert (
        rendering.position_as_string(obs, x, y, last_move)
        == f"[{constants.MOVES[last_move]}]"
    )


@hypothesis.given(
    x=st.integers(min_value=0, max_value=grids.GRID_HEIGHT - 1),
    y=st.integers(min_value=0, max_value=grids.GRID_WIDTH - 1),
)
def test_position_as_string_on_cliff_position_with_player_and_no_last_move(
    x: int, y: int
):
    obs = grids.grid(x, y, cliffs=[(x, y)])
    assert rendering.position_as_string(obs, x, y, None) == "[x̄]"


@hypothesis.given(
    x=st.integers(min_value=0, max_value=grids.GRID_HEIGHT - 1),
    y=st.integers(min_value=0, max_value=grids.GRID_WIDTH - 1),
    last_move=st.integers(min_value=0, max_value=len(constants.MOVES) - 1),
)
def test_position_as_string_on_cliff_position_with_player_and_last_move(
    x: int, y: int, last_move: int
):
    x = grids.GRID_HEIGHT - 1
    obs = grids.grid(x, y, cliffs=[(x, y)])
    assert rendering.position_as_string(obs, x, y, last_move) == "[x̄]"


@hypothesis.given(
    x=st.integers(min_value=0, max_value=grids.GRID_HEIGHT - 1),
    y=st.integers(min_value=0, max_value=grids.GRID_WIDTH - 1),
)
def test_position_as_string_on_cliff_position_without_player(x: int, y: int):
    # no player on the grid
    obs = grids.empty_grid()
    obs[1, x, y] = 1
    assert rendering.position_as_string(obs, x, y, None) == "[X]"


@hypothesis.given(
    x=st.integers(min_value=0, max_value=grids.GRID_HEIGHT - 1),
    y=st.integers(min_value=0, max_value=grids.GRID_WIDTH - 1),
    last_move=st.integers(min_value=0, max_value=len(constants.MOVES) - 1),
)
def test_position_as_string_on_cliff_position_without_player_and_after_move(
    x: int, y: int, last_move: int
):
    # no player on the grid
    obs = grids.empty_grid()
    obs[1, x, y] = 1
    assert rendering.position_as_string(obs, x, y, last_move) == "[X]"


def grid_as_string(rows: Iterable[Iterable[Any]]) -> Sequence[Any]:
    return "\n".join(["".join(row) for row in rows])


@pytest.fixture(scope="module")
def sprites():
    class MockSprites(rendering.Sprites):
        cliff_sprite = grids.color_block(grids.CLIFF_COLOR)
        path_sprite = grids.color_block(grids.PATH_COLOR)
        actor_sprite = grids.color_block(grids.ACTOR_COLOR)

    return MockSprites()
