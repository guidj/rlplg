from typing import Any, Iterable, Sequence, Tuple

import hypothesis
import numpy as np
import pytest
from gymnasium import spaces
from hypothesis import strategies as st
from PIL import Image as image
from rlplg import core
from rlplg.environments import iceworld

from tests.rlplg import dynamics
from tests.rlplg.environments import worlds


def test_iceworld_init():
    environment = iceworld.IceWorld(size=(4, 12), lakes=[], goals=[], start=(3, 0))
    assert environment.action_space == spaces.Discrete(4)
    assert environment.observation_space == spaces.Dict(
        {
            "id": spaces.Discrete(4 * 12),
            "start": spaces.Tuple((spaces.Discrete(4), spaces.Discrete(12))),
            "agent": spaces.Tuple((spaces.Discrete(4), spaces.Discrete(12))),
            "lakes": spaces.Sequence(
                spaces.Tuple((spaces.Discrete(4), spaces.Discrete(12)))
            ),
            "goals": spaces.Sequence(
                spaces.Tuple((spaces.Discrete(4), spaces.Discrete(12)))
            ),
            "size": spaces.Box(
                low=np.array([3, 11]),
                high=np.array([3, 11]),
                dtype=np.int64,
            ),
        }
    )
    dynamics.assert_transition_mapping(
        environment.transition, env_desc=core.EnvDesc(num_states=4 * 12, num_actions=4)
    )


def test_iceworld_reset():
    environment = iceworld.IceWorld(
        size=(4, 12), lakes=[], goals=[(3, 11)], start=(3, 0)
    )
    obs, info = environment.reset()
    assert_observation(
        obs,
        {
            "id": 12,
            "start": (3, 0),
            "agent": (3, 0),
            "lakes": [],
            "goals": [(3, 11)],
            "size": (4, 12),
        },
    )
    assert info == {}


def test_iceworld_transition_step():
    environment = iceworld.IceWorld(
        size=(4, 12), lakes=[], goals=[(3, 11)], start=(3, 0)
    )
    environment.reset()
    obs, reward, finished, truncated, info = environment.step(iceworld.UP)
    assert_observation(
        obs,
        {
            "id": 0,
            "start": (3, 0),
            "agent": (2, 0),
            "lakes": [],
            "goals": [(3, 11)],
            "size": (4, 12),
        },
    )
    assert reward == -1
    assert finished is False
    assert truncated is False
    assert info == {}


def test_iceworld_transition_into_lake():
    environment = iceworld.IceWorld(
        size=(4, 12), lakes=[(3, 1)], goals=[(3, 11)], start=(3, 0)
    )
    environment.reset()
    obs, reward, terminated, truncated, info = environment.step(iceworld.RIGHT)
    assert_observation(
        obs,
        {
            "id": 13,
            "start": (3, 0),
            "agent": (3, 1),
            "lakes": [(3, 1)],
            "goals": [(3, 11)],
            "size": (4, 12),
        },
    )
    assert reward == -2.0 * 4 * 12
    assert terminated is True
    assert truncated is False
    assert info == {}

    obs, reward, terminated, truncated, info = environment.step(iceworld.RIGHT)
    assert_observation(
        obs,
        {
            "id": 13,
            "start": (3, 0),
            "agent": (3, 1),
            "lakes": [(3, 1)],
            "goals": [(3, 11)],
            "size": (4, 12),
        },
    )
    assert reward == 0
    assert terminated is True
    assert truncated is False
    assert info == {}


def test_iceworld_final_step():
    environment = iceworld.IceWorld(
        size=(4, 12), lakes=[], goals=[(3, 1)], start=(3, 0)
    )
    environment.reset()
    obs, reward, terminated, truncated, info = environment.step(iceworld.RIGHT)
    assert_observation(
        obs,
        {
            "id": 13,
            "start": (3, 0),
            "agent": (3, 1),
            "lakes": [],
            "goals": [(3, 1)],
            "size": (4, 12),
        },
    )
    assert reward == -1.0
    assert terminated is True
    assert truncated is False
    assert info == {}

    obs, reward, terminated, truncated, info = environment.step(iceworld.RIGHT)
    assert_observation(
        obs,
        {
            "id": 13,
            "start": (3, 0),
            "agent": (3, 1),
            "lakes": [],
            "goals": [(3, 1)],
            "size": (4, 12),
        },
    )
    assert reward == 0
    assert terminated is True
    assert truncated is False
    assert info == {}


def test_iceworld_render():
    environment = iceworld.IceWorld(
        size=(4, 12), lakes=[], goals=[(3, 11)], start=(3, 0)
    )
    environment.reset()

    np.testing.assert_array_equal(
        environment.render(),
        worlds.ice(x=3, y=0, height=4, width=12, lakes=[], goals=[(3, 11)]),
    )


def test_iceworld_render_with_unsupported_mode():
    for mode in ("human",):
        with pytest.raises(NotImplementedError):
            environment = iceworld.IceWorld(
                size=(4, 12), lakes=[], goals=[(3, 11)], start=(3, 0), render_mode=mode
            )
            environment.reset()
            environment.render()


def test_iceworld_seed():
    environment = iceworld.IceWorld(
        size=(4, 12), lakes=[], goals=[(3, 11)], start=(3, 0)
    )
    assert environment.seed() is None
    assert environment.seed(1) == 1
    assert environment.seed() == 1
    assert environment.seed(213) == 213
    assert environment.seed() == 213
    assert environment.seed(117) == 117
    assert environment.seed() == 117


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_apply_action_going_up(x: int, y: int):
    obs = {
        "id": None,
        "start": (0, 0),
        "agent": (x, y),
        "lakes": [],
        "goals": [],
        "size": (worlds.HEIGHT, worlds.WIDTH),
    }
    output_observation, output_reward = iceworld.apply_action(obs, iceworld.UP)
    expected = {
        "id": 13,
        "start": (0, 0),
        "agent": (max(x - 1, 0), y),
        "lakes": [],
        "goals": [],
        "size": (worlds.HEIGHT, worlds.WIDTH),
    }
    assert_observation(output_observation, expected)
    assert output_reward == -1.0


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_apply_action_going_down(x: int, y: int):
    obs = {
        "id": None,
        "start": (0, 0),
        "agent": (x, y),
        "lakes": [],
        "goals": [],
        "size": (worlds.HEIGHT, worlds.WIDTH),
    }
    output_observation, output_reward = iceworld.apply_action(obs, iceworld.DOWN)
    expected_observation = {
        "id": 5,
        "start": (0, 0),
        "agent": (min(x + 1, worlds.HEIGHT - 1), y),
        "lakes": [],
        "goals": [],
        "size": (worlds.HEIGHT, worlds.WIDTH),
    }

    assert_observation(output_observation, expected_observation)
    assert output_reward == -1.0


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_apply_action_going_left(x: int, y: int):
    obs = {
        "id": None,
        "start": (0, 0),
        "agent": (x, y),
        "lakes": [],
        "goals": [],
        "size": (worlds.HEIGHT, worlds.WIDTH),
    }
    output_observation, output_reward = iceworld.apply_action(obs, iceworld.LEFT)
    expected = {
        "id": 0,
        "start": (0, 0),
        "agent": (x, max(0, y - 1)),
        "lakes": [],
        "goals": [],
        "size": (worlds.HEIGHT, worlds.WIDTH),
    }
    assert_observation(output_observation, expected)
    assert output_reward == -1.0


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_apply_action_going_right(x: int, y: int):
    obs = {
        "id": None,
        "start": (0, 0),
        "agent": (x, y),
        "lakes": [],
        "goals": [],
        "size": (worlds.HEIGHT, worlds.WIDTH),
    }
    output_observation, output_reward = iceworld.apply_action(obs, iceworld.RIGHT)
    expected = {
        "id": 1,
        "start": (0, 0),
        "agent": (x, min(y + 1, worlds.WIDTH - 1)),
        "lakes": [],
        "goals": [],
        "size": (worlds.HEIGHT, worlds.WIDTH),
    }
    assert_observation(output_observation, expected)
    assert output_reward == -1.0


@hypothesis.given(
    st.tuples(
        st.integers(min_value=0, max_value=99),
        st.integers(min_value=0, max_value=99),
    ),
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=99),
            st.integers(min_value=0, max_value=99),
        )
    ),
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=99),
            st.integers(min_value=0, max_value=99),
        )
    ),
)
def test_create_observation(
    starting_pos: Tuple[int, int],
    lakes: Sequence[Tuple[int, int]],
    goals: Sequence[Tuple[int, int]],
):
    output = iceworld.create_observation(
        size=(100, 100),
        start=starting_pos,
        agent=starting_pos,
        lakes=lakes,
        goals=goals,
    )
    expected = {
        "id": 0,
        "start": starting_pos,
        "agent": starting_pos,
        "lakes": lakes,
        "goals": goals,
        "size": (100, 100),
    }
    assert_observation(output, expected)


@hypothesis.given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=99),
            st.integers(min_value=0, max_value=99),
        )
    )
)
def test_create_position_from_state_id_fn(states: Sequence[Tuple[int, int]]):
    states_mapping = {state: _id for _id, state in enumerate(states)}
    position_from_state_id_fn = iceworld.create_position_from_state_id_fn(
        states_mapping
    )
    for state, _id in states_mapping.items():
        assert position_from_state_id_fn(_id) == state


def test_as_grid():
    # soxo
    # ooog
    observation = iceworld.create_observation(
        size=(2, 4), start=(0, 0), agent=(0, 0), lakes=[(0, 2)], goals=[(1, 3)]
    )

    output = iceworld.as_grid(observation)
    expected = np.array(
        [
            np.array([[1, 0, 0, 0], [0, 0, 0, 0]]),
            np.array([[0, 0, 1, 0], [0, 0, 0, 0]]),
            np.array([[0, 0, 0, 0], [0, 0, 0, 1]]),
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(output, expected)


def assert_observation(output: Any, expected: Any) -> None:
    assert len(output) == len(expected)
    np.testing.assert_array_equal(output["size"], expected["size"])
    output["agent"] == expected["agent"]
    output["start"] == expected["start"]
    output["lakes"] == expected["lakes"]
    output["goals"] == expected["goals"]


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
    np.testing.assert_equal(iceworld.image_as_array(img), array)


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
    np.testing.assert_equal(iceworld.image_as_array(img), np.array(img)[:, :, :3])


def test_observation_as_string_with_empty_grid_and_no_last_move():
    obs = worlds.empty_grid()
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
    assert iceworld.observation_as_string(obs, None) == expected


@hypothesis.given(
    last_move=st.integers(min_value=0, max_value=len(iceworld.MOVES) - 1),
)
def test_observation_as_string_with_empty_grid_and_last_move(last_move: int):
    obs = worlds.empty_grid()
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
    assert iceworld.observation_as_string(obs, last_move) == expected


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_observation_as_string_with_lakes_and_no_last_move(x: int, y: int):
    obs = worlds.empty_grid()
    obs[1, x, y] = 1
    expected = [
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["\n\n"],
    ]
    expected[x][y] = "[H]"
    assert iceworld.observation_as_string(obs, None) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_observation_as_string_with_goals(x: int, y: int):
    obs = worlds.empty_grid()
    obs[2, x, y] = 1
    expected = [
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["\n\n"],
    ]
    expected[x][y] = "[G]"
    assert iceworld.observation_as_string(obs, None) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_observation_as_string_with_agent_and_no_last_move(x: int, y: int):
    obs = worlds.ice(x, y)
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
    assert iceworld.observation_as_string(obs, None) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
    last_move=st.integers(min_value=0, max_value=len(iceworld.MOVES) - 1),
)
def test_observation_as_string_with_agent_and_last_move(
    x: int,
    y: int,
    last_move: int,
):
    obs = worlds.ice(x, y)
    expected = [
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["\n\n"],
    ]
    expected[x][y] = f"[{iceworld.MOVES[last_move]}]"
    assert iceworld.observation_as_string(obs, last_move) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_observation_as_string_with_agent_in_lake_and_no_last_move(
    x: int,
    y: int,
):
    obs = worlds.ice(x, y, lakes=[(x, y)])
    expected = [
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["\n\n"],
    ]
    expected[x][y] = "[Ħ]"
    assert iceworld.observation_as_string(obs, None) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
    last_move=st.integers(min_value=0, max_value=len(iceworld.MOVES) - 1),
)
def test_observation_as_string_with_agent_in_lake_and_last_move(
    x: int, y: int, last_move: int
):
    obs = worlds.ice(x, y, lakes=[(x, y)])
    expected = [
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["\n\n"],
    ]
    expected[x][y] = "[Ħ]"
    assert iceworld.observation_as_string(obs, last_move) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_observation_as_string_with_agent_in_exit_and_no_last_move(
    x: int,
    y: int,
):
    obs = worlds.ice(x, y, goals=[(x, y)])
    expected = [
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["\n\n"],
    ]
    expected[x][y] = "[Ğ]"
    assert iceworld.observation_as_string(obs, None) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
    last_move=st.integers(min_value=0, max_value=len(iceworld.MOVES) - 1),
)
def test_observation_as_string_with_agent_in_exit_and_last_move(
    x: int, y: int, last_move: int
):
    obs = worlds.ice(x, y, goals=[(x, y)])
    expected = [
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["\n\n"],
    ]
    expected[x][y] = "[Ğ]"
    assert iceworld.observation_as_string(obs, last_move) == grid_as_string(expected)


def test_position_as_string_in_starting_position_with_agent_and_no_last_move():
    x, y = worlds.HEIGHT - 1, 0
    obs = worlds.ice(x, y)
    assert iceworld.position_as_string(obs, x, y, None) == "[S]"


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_position_as_string_on_safe_position_with_agent_and_no_last_move(
    x: int, y: int
):
    obs = worlds.ice(x, y)
    assert iceworld.position_as_string(obs, x, y, None) == "[S]"


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
    last_move=st.integers(min_value=0, max_value=len(iceworld.MOVES) - 1),
)
def test_position_as_string_on_safe_position_with_agent_and_last_move(
    x: int, y: int, last_move: int
):
    obs = worlds.ice(x, y)
    assert (
        iceworld.position_as_string(obs, x, y, last_move)
        == f"[{iceworld.MOVES[last_move]}]"
    )


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_position_as_string_on_lake_position_with_agent_and_no_last_move(
    x: int, y: int
):
    obs = worlds.ice(x, y, lakes=[(x, y)])
    assert iceworld.position_as_string(obs, x, y, None) == "[Ħ]"


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
    last_move=st.integers(min_value=0, max_value=len(iceworld.MOVES) - 1),
)
def test_position_as_string_on_lake_position_with_agent_and_last_move(
    x: int, y: int, last_move: int
):
    x = worlds.HEIGHT - 1
    obs = worlds.ice(x, y, lakes=[(x, y)])
    assert iceworld.position_as_string(obs, x, y, last_move) == "[Ħ]"


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_position_as_string_on_lake_position_without_agent(x: int, y: int):
    # no agent on the grid
    obs = worlds.empty_grid()
    obs[1, x, y] = 1
    assert iceworld.position_as_string(obs, x, y, None) == "[H]"


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
    last_move=st.integers(min_value=0, max_value=len(iceworld.MOVES) - 1),
)
def test_position_as_string_on_lake_position_without_agent_and_after_move(
    x: int, y: int, last_move: int
):
    # no agent on the grid
    obs = worlds.empty_grid()
    obs[1, x, y] = 1
    assert iceworld.position_as_string(obs, x, y, last_move) == "[H]"


def grid_as_string(rows: Iterable[Iterable[Any]]) -> Sequence[Any]:
    return "\n".join(["".join(row) for row in rows])
