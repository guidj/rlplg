from typing import Any, Iterable, Sequence, Tuple

import hypothesis
import numpy as np
import pytest
from gymnasium import spaces
from hypothesis import strategies as st
from PIL import Image as image

from rlplg import core
from rlplg.environments import gridworld
from tests.rlplg import dynamics
from tests.rlplg.environments import worlds


def test_gridworld_init():
    environment = gridworld.GridWorld(size=(4, 12), cliffs=[], exits=[], start=(3, 0))
    assert environment.action_space == spaces.Discrete(4)
    assert environment.observation_space == spaces.Dict(
        {
            "start": spaces.Tuple((spaces.Discrete(4), spaces.Discrete(12))),
            "agent": spaces.Tuple((spaces.Discrete(4), spaces.Discrete(12))),
            "cliffs": spaces.Sequence(
                spaces.Tuple((spaces.Discrete(4), spaces.Discrete(12)))
            ),
            "exits": spaces.Sequence(
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


def test_gridworld_reset():
    environment = gridworld.GridWorld(
        size=(4, 12), cliffs=[], exits=[(3, 11)], start=(3, 0)
    )
    obs, info = environment.reset()
    assert_observation(
        obs,
        {
            "start": (3, 0),
            "agent": (3, 0),
            "cliffs": [],
            "exits": [(3, 11)],
            "size": (4, 12),
        },
    )
    assert info == {}


def test_gridworld_transition_step():
    environment = gridworld.GridWorld(
        size=(4, 12), cliffs=[], exits=[(3, 11)], start=(3, 0)
    )
    environment.reset()
    obs, reward, finished, truncated, info = environment.step(gridworld.UP)
    assert_observation(
        obs,
        {
            "start": (3, 0),
            "agent": (2, 0),
            "cliffs": [],
            "exits": [(3, 11)],
            "size": (4, 12),
        },
    )
    assert reward == -1
    assert finished is False
    assert truncated is False
    assert info == {}


def test_gridworld_transition_into_cliff():
    environment = gridworld.GridWorld(
        size=(4, 12), cliffs=[(3, 1)], exits=[(3, 11)], start=(3, 0)
    )
    environment.reset()
    obs, reward, terminated, truncated, info = environment.step(gridworld.RIGHT)
    assert_observation(
        obs,
        {
            "start": (3, 0),
            "agent": (3, 0),  # sent back to the start
            "cliffs": [(3, 1)],
            "exits": [(3, 11)],
            "size": (4, 12),
        },
    )
    assert reward == -100.0
    assert terminated is False
    assert truncated is False
    assert info == {}


def test_gridworld_final_step():
    environment = gridworld.GridWorld(
        size=(4, 12), cliffs=[], exits=[(3, 1)], start=(3, 0)
    )
    environment.reset()
    obs, reward, terminated, truncated, info = environment.step(gridworld.RIGHT)
    assert_observation(
        obs,
        {
            "start": (3, 0),
            "agent": (3, 1),
            "cliffs": [],
            "exits": [(3, 1)],
            "size": (4, 12),
        },
    )
    assert reward == -1.0
    assert terminated is True
    assert truncated is False
    assert info == {}


def test_gridworld_render():
    environment = gridworld.GridWorld(
        size=(4, 12), cliffs=[], exits=[(3, 11)], start=(3, 0)
    )
    environment.reset()

    np.testing.assert_array_equal(
        environment.render(),
        worlds.grid(x=3, y=0, height=4, width=12, cliffs=[], exits=[(3, 11)]),
    )


def test_gridworld_render_with_unsupported_mode():
    for mode in ("human",):
        with pytest.raises(NotImplementedError):
            environment = gridworld.GridWorld(
                size=(4, 12), cliffs=[], exits=[(3, 11)], start=(3, 0), render_mode=mode
            )
            environment.reset()
            environment.render()


def test_gridworld_seed():
    environment = gridworld.GridWorld(
        size=(4, 12), cliffs=[], exits=[(3, 11)], start=(3, 0)
    )
    assert environment.seed() is None
    assert environment.seed(1) == 1
    assert environment.seed() == 1
    assert environment.seed(213) == 213
    assert environment.seed() == 213
    assert environment.seed(117) == 117
    assert environment.seed() == 117


def test_states_mapping():
    # soxo
    # xoog
    output = gridworld.states_mapping(size=(2, 4), cliffs=[(0, 2), (1, 0)])
    expected = {(0, 0): 0, (0, 1): 1, (0, 3): 2, (1, 1): 3, (1, 2): 4, (1, 3): 5}
    assert output == expected


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_apply_action_going_up(x: int, y: int):
    obs = {
        "start": (0, 0),
        "agent": (x, y),
        "cliffs": [],
        "exits": [],
        "size": (worlds.HEIGHT, worlds.WIDTH),
    }
    output_observation, output_reward = gridworld.apply_action(obs, gridworld.UP)
    expected = {
        "start": (0, 0),
        "agent": (max(x - 1, 0), y),
        "cliffs": [],
        "exits": [],
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
        "start": (0, 0),
        "agent": (x, y),
        "cliffs": [],
        "exits": [],
        "size": (worlds.HEIGHT, worlds.WIDTH),
    }
    output_observation, output_reward = gridworld.apply_action(obs, gridworld.DOWN)
    expected_observation = {
        "start": (0, 0),
        "agent": (min(x + 1, worlds.HEIGHT - 1), y),
        "cliffs": [],
        "exits": [],
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
        "start": (0, 0),
        "agent": (x, y),
        "cliffs": [],
        "exits": [],
        "size": (worlds.HEIGHT, worlds.WIDTH),
    }
    output_observation, output_reward = gridworld.apply_action(obs, gridworld.LEFT)
    expected = {
        "start": (0, 0),
        "agent": (x, max(0, y - 1)),
        "cliffs": [],
        "exits": [],
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
        "start": (0, 0),
        "agent": (x, y),
        "cliffs": [],
        "exits": [],
        "size": (worlds.HEIGHT, worlds.WIDTH),
    }
    output_observation, output_reward = gridworld.apply_action(obs, gridworld.RIGHT)
    expected = {
        "start": (0, 0),
        "agent": (x, min(y + 1, worlds.WIDTH - 1)),
        "cliffs": [],
        "exits": [],
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
    cliffs: Sequence[Tuple[int, int]],
    exits: Sequence[Tuple[int, int]],
):
    output = gridworld.create_observation(
        size=(100, 100),
        start=starting_pos,
        agent=starting_pos,
        cliffs=cliffs,
        exits=exits,
    )
    expected = {
        "start": starting_pos,
        "agent": starting_pos,
        "cliffs": cliffs,
        "exits": exits,
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
def test_create_state_id_fn(states: Sequence[Tuple[int, int]]):
    states_mapping = {state: _id for _id, state in enumerate(states)}
    state_id_fn = gridworld.create_state_id_fn(states_mapping)
    for state, _id in states_mapping.items():
        # duck type an observation
        assert state_id_fn({"agent": state}) == _id


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
    position_from_state_id_fn = gridworld.create_position_from_state_id_fn(
        states_mapping
    )
    for state, _id in states_mapping.items():
        assert position_from_state_id_fn(_id) == state


def test_as_grid():
    # soxo
    # ooog
    observation = gridworld.create_observation(
        size=(2, 4), start=(0, 0), agent=(0, 0), cliffs=[(0, 2)], exits=[(1, 3)]
    )

    output = gridworld.as_grid(observation)
    expected = np.array(
        [
            np.array([[1, 0, 0, 0], [0, 0, 0, 0]]),
            np.array([[0, 0, 1, 0], [0, 0, 0, 0]]),
            np.array([[0, 0, 0, 0], [0, 0, 0, 1]]),
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(output, expected)


def test_create_env_spec():
    env_spec = gridworld.create_env_spec(
        size=(4, 12), cliffs=[], exits=[], start=(3, 0)
    )
    assert env_spec.name == "GridWorld"
    assert len(env_spec.level) > 0
    assert isinstance(env_spec.environment, gridworld.GridWorld)
    assert isinstance(env_spec.discretizer, gridworld.GridWorldMdpDiscretizer)
    assert env_spec.mdp.env_desc.num_states == 48
    assert env_spec.mdp.env_desc.num_actions == 4
    assert len(env_spec.mdp.transition) == 48


def assert_observation(output: Any, expected: Any) -> None:
    np.testing.assert_array_equal(output["size"], expected["size"])
    np.testing.assert_array_equal(output["agent"], expected["agent"])
    np.testing.assert_array_equal(output["start"], expected["start"])
    np.testing.assert_array_equal(output["cliffs"], expected["cliffs"])
    np.testing.assert_array_equal(output["exits"], expected["exits"])


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
    np.testing.assert_equal(gridworld.image_as_array(img), array)


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
    np.testing.assert_equal(gridworld.image_as_array(img), np.array(img)[:, :, :3])


def test_observation_as_window_simple_case(sprites: gridworld.Sprites):
    obs = worlds.grid(x=1, y=1, cliffs=[(4, 1), (4, 2), (4, 3)], exits=[(4, 4)])
    path = worlds.color_block(worlds.PATH_COLOR)
    cliff = worlds.color_block(worlds.CLIFF_COLOR)
    actor = worlds.color_block(worlds.ACTOR_COLOR)
    vsep = worlds.color_block((192, 192, 192), width=1, height=worlds.HEIGHT)
    hsep = worlds.color_block((192, 192, 192), width=worlds.WIDTH, height=1)
    dot = worlds.color_block((192, 192, 192), width=1, height=1)

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

    output = gridworld.observation_as_image(sprites, obs, last_move=None)

    np.testing.assert_equal(output, expected)


def test_hborder_with_unit_size():
    np.testing.assert_equal(gridworld._hborder(1), [[[192, 192, 192]]])


def test_hborder_with_non_positive_size():
    with pytest.raises(ValueError):
        gridworld._hborder(0)


@hypothesis.given(size=st.integers(min_value=1, max_value=100))
def test_hborder_with_random_size(size: int):
    np.testing.assert_equal(gridworld._hborder(size), [[[192, 192, 192]] * size])


def test_vborder_with_unit_size():
    np.testing.assert_equal(gridworld._vborder(1), [[[192, 192, 192]]])


def test_vborder_with_non_positive_size():
    with pytest.raises(ValueError):
        gridworld._vborder(0)


@hypothesis.given(size=st.integers(min_value=1, max_value=100))
def test_vborder_with_random_size(size: int):
    np.testing.assert_equal(gridworld._vborder(size), [[[192, 192, 192]]] * size)


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
    assert gridworld.observation_as_string(obs, None) == expected


@hypothesis.given(
    last_move=st.integers(min_value=0, max_value=len(gridworld.MOVES) - 1),
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
    assert gridworld.observation_as_string(obs, last_move) == expected


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_observation_as_string_with_cliffs_and_no_last_move(x: int, y: int):
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
    expected[x][y] = "[X]"
    assert gridworld.observation_as_string(obs, None) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_observation_as_string_with_exits(x: int, y: int):
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
    expected[x][y] = "[E]"
    assert gridworld.observation_as_string(obs, None) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_observation_as_string_with_agent_and_no_last_move(x: int, y: int):
    obs = worlds.grid(x, y)
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
    assert gridworld.observation_as_string(obs, None) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
    last_move=st.integers(min_value=0, max_value=len(gridworld.MOVES) - 1),
)
def test_observation_as_string_with_agent_and_last_move(
    x: int,
    y: int,
    last_move: int,
):
    obs = worlds.grid(x, y)
    expected = [
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["\n\n"],
    ]
    expected[x][y] = f"[{gridworld.MOVES[last_move]}]"
    assert gridworld.observation_as_string(obs, last_move) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_observation_as_string_with_agent_in_cliff_and_no_last_move(
    x: int,
    y: int,
):
    obs = worlds.grid(x, y, cliffs=[(x, y)])
    expected = [
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["\n\n"],
    ]
    expected[x][y] = "[x̄]"
    assert gridworld.observation_as_string(obs, None) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
    last_move=st.integers(min_value=0, max_value=len(gridworld.MOVES) - 1),
)
def test_observation_as_string_with_agent_in_cliff_and_last_move(
    x: int, y: int, last_move: int
):
    obs = worlds.grid(x, y, cliffs=[(x, y)])
    expected = [
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["\n\n"],
    ]
    expected[x][y] = "[x̄]"
    assert gridworld.observation_as_string(obs, last_move) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_observation_as_string_with_agent_in_exit_and_no_last_move(
    x: int,
    y: int,
):
    obs = worlds.grid(x, y, exits=[(x, y)])
    expected = [
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["\n\n"],
    ]
    expected[x][y] = "[Ē]"
    assert gridworld.observation_as_string(obs, None) == grid_as_string(expected)


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
    last_move=st.integers(min_value=0, max_value=len(gridworld.MOVES) - 1),
)
def test_observation_as_string_with_agent_in_exit_and_last_move(
    x: int, y: int, last_move: int
):
    obs = worlds.grid(x, y, exits=[(x, y)])
    expected = [
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]"],
        ["\n\n"],
    ]
    expected[x][y] = "[Ē]"
    assert gridworld.observation_as_string(obs, last_move) == grid_as_string(expected)


def test_position_as_string_in_starting_position_with_agent_and_no_last_move():
    x, y = worlds.HEIGHT - 1, 0
    obs = worlds.grid(x, y)
    assert gridworld.position_as_string(obs, x, y, None) == "[S]"


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_position_as_string_on_safe_position_with_agent_and_no_last_move(
    x: int, y: int
):
    obs = worlds.grid(x, y)
    assert gridworld.position_as_string(obs, x, y, None) == "[S]"


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
    last_move=st.integers(min_value=0, max_value=len(gridworld.MOVES) - 1),
)
def test_position_as_string_on_safe_position_with_agent_and_last_move(
    x: int, y: int, last_move: int
):
    obs = worlds.grid(x, y)
    assert (
        gridworld.position_as_string(obs, x, y, last_move)
        == f"[{gridworld.MOVES[last_move]}]"
    )


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_position_as_string_on_cliff_position_with_agent_and_no_last_move(
    x: int, y: int
):
    obs = worlds.grid(x, y, cliffs=[(x, y)])
    assert gridworld.position_as_string(obs, x, y, None) == "[x̄]"


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
    last_move=st.integers(min_value=0, max_value=len(gridworld.MOVES) - 1),
)
def test_position_as_string_on_cliff_position_with_agent_and_last_move(
    x: int, y: int, last_move: int
):
    x = worlds.HEIGHT - 1
    obs = worlds.grid(x, y, cliffs=[(x, y)])
    assert gridworld.position_as_string(obs, x, y, last_move) == "[x̄]"


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
)
def test_position_as_string_on_cliff_position_without_agent(x: int, y: int):
    # no agent on the grid
    obs = worlds.empty_grid()
    obs[1, x, y] = 1
    assert gridworld.position_as_string(obs, x, y, None) == "[X]"


@hypothesis.given(
    x=st.integers(min_value=0, max_value=worlds.HEIGHT - 1),
    y=st.integers(min_value=0, max_value=worlds.WIDTH - 1),
    last_move=st.integers(min_value=0, max_value=len(gridworld.MOVES) - 1),
)
def test_position_as_string_on_cliff_position_without_agent_and_after_move(
    x: int, y: int, last_move: int
):
    # no agent on the grid
    obs = worlds.empty_grid()
    obs[1, x, y] = 1
    assert gridworld.position_as_string(obs, x, y, last_move) == "[X]"


def grid_as_string(rows: Iterable[Iterable[Any]]) -> Sequence[Any]:
    return "\n".join(["".join(row) for row in rows])


@pytest.fixture(scope="module")
def sprites():
    class MockSprites(gridworld.Sprites):
        cliff_sprite = worlds.color_block(worlds.CLIFF_COLOR)
        path_sprite = worlds.color_block(worlds.PATH_COLOR)
        actor_sprite = worlds.color_block(worlds.ACTOR_COLOR)

    return MockSprites()
