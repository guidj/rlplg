from typing import Sequence, Tuple

import hypothesis
import numpy as np
import pytest
from hypothesis import strategies as st
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from rlplg.environments.gridworld import constants, env
from tests.rlplg.environments.gridworld import grids


def test_gridworld_init():
    environment = env.GridWorld(size=(4, 12), cliffs=[], exits=[], start=(3, 0))

    assert environment.observation_spec() == {
        "start": array_spec.BoundedArraySpec(
            shape=(2,),
            dtype=np.int64,
            minimum=np.array([3, 0]),
            maximum=np.array([3, 0]),
            name="player",
        ),
        "player": array_spec.BoundedArraySpec(
            shape=(2,),
            dtype=np.int64,
            minimum=np.array([0, 0]),
            maximum=np.array([3, 11]),
            name="player",
        ),
        "cliffs": array_spec.BoundedArraySpec(
            shape=(0, 2),
            dtype=np.int64,
            minimum=np.array([0, 0]),
            maximum=np.array([3, 11]),
            name="cliffs",
        ),
        "exits": array_spec.BoundedArraySpec(
            shape=(0, 2),
            dtype=np.int64,
            minimum=np.array([0, 0]),
            maximum=np.array([3, 11]),
            name="exits",
        ),
        "size": array_spec.BoundedArraySpec(
            shape=(2,),
            dtype=np.int64,
            minimum=np.array([4, 12]),
            maximum=np.array([4, 12]),
            name="size",
        ),
    }
    assert environment.action_spec() == array_spec.BoundedArraySpec(
        shape=(), dtype=np.int64, minimum=0, maximum=3, name="action"
    )


def test_gridworld_reset():
    environment = env.GridWorld(size=(4, 12), cliffs=[], exits=[(3, 11)], start=(3, 0))
    step = environment.reset()
    expected = ts.TimeStep(
        step_type=ts.StepType.FIRST,
        reward=0.0,
        discount=1.0,
        observation={
            "start": (3, 0),
            "player": (3, 0),
            "cliffs": set([]),
            "exits": set([(3, 11)]),
            "size": (4, 12),
        },
    )
    assert step.step_type == expected.step_type
    assert step.reward == expected.reward
    assert step.discount == expected.discount
    assert step.observation == expected.observation


def test_gridworld_transition_step():
    environment = env.GridWorld(size=(4, 12), cliffs=[], exits=[(3, 11)], start=(3, 0))
    environment.reset()
    step = environment.step(constants.UP)
    expected = ts.TimeStep(
        step_type=ts.StepType.MID,
        reward=-1.0,
        discount=1.0,
        observation={
            "start": (3, 0),
            "player": (2, 0),
            "cliffs": set([]),
            "exits": set([(3, 11)]),
            "size": (4, 12),
        },
    )
    assert step.step_type == expected.step_type
    assert step.reward == expected.reward
    assert step.discount == expected.discount
    assert step.observation == expected.observation


def test_gridworld_transition_into_cliff():
    environment = env.GridWorld(
        size=(4, 12), cliffs=[(3, 1)], exits=[(3, 11)], start=(3, 0)
    )
    environment.reset()
    step = environment.step(constants.RIGHT)
    expected = ts.TimeStep(
        step_type=ts.StepType.MID,
        reward=-100.0,
        discount=1.0,
        observation={
            "start": (3, 0),
            "player": (3, 0),  # sent back to the start
            "cliffs": set([(3, 1)]),
            "exits": set([(3, 11)]),
            "size": (4, 12),
        },
    )
    assert step.step_type == expected.step_type
    assert step.reward == expected.reward
    assert step.discount == expected.discount
    assert step.observation == expected.observation


def test_gridworld_final_step():
    environment = env.GridWorld(size=(4, 12), cliffs=[], exits=[(3, 1)], start=(3, 0))
    environment.reset()
    step = environment.step(constants.RIGHT)
    expected = ts.TimeStep(
        step_type=ts.StepType.LAST,
        reward=-1.0,
        discount=0.0,
        observation={
            "start": (3, 0),
            "player": (3, 1),
            "cliffs": set([]),
            "exits": set([(3, 1)]),
            "size": (4, 12),
        },
    )
    assert step.step_type == expected.step_type
    assert step.reward == expected.reward
    assert step.discount == expected.discount
    assert step.observation == expected.observation


def test_gridworld_render():
    environment = env.GridWorld(size=(4, 12), cliffs=[], exits=[(3, 11)], start=(3, 0))
    environment.reset()

    np.testing.assert_array_equal(
        environment.render(),
        grids.grid(x=3, y=0, height=4, width=12, cliffs=[], exits=[(3, 11)]),
    )


def test_gridworld_render_with_unsupported_mode():
    environment = env.GridWorld(size=(4, 12), cliffs=[], exits=[(3, 11)], start=(3, 0))
    environment.reset()
    for mode in ("human",):
        with pytest.raises(NotImplementedError):
            environment.render(mode)


def test_gridworld_seed():
    environment = env.GridWorld(size=(4, 12), cliffs=[], exits=[(3, 11)], start=(3, 0))
    assert environment.seed() is None


@hypothesis.given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_gridworld_seed_with_value_provided(seed: int):
    environment = env.GridWorld(size=(4, 12), cliffs=[], exits=[], start=(3, 0))
    assert environment.seed(seed) == seed


def test_states_mapping():
    # soxo
    # xoog
    output = env.states_mapping(size=(2, 4), cliffs=[(0, 2), (1, 0)])
    expected = {(0, 0): 0, (0, 1): 1, (0, 3): 2, (1, 1): 3, (1, 2): 4, (1, 3): 5}
    assert output == expected


@hypothesis.given(
    x=st.integers(min_value=0, max_value=grids.GRID_HEIGHT - 1),
    y=st.integers(min_value=0, max_value=grids.GRID_WIDTH - 1),
)
def test_apply_action_going_up(x: int, y: int):
    obs = {
        "start": (0, 0),
        "player": (x, y),
        "cliffs": set([]),
        "exits": set([]),
        "size": (grids.GRID_HEIGHT, grids.GRID_WIDTH),
    }
    output_observation, output_reward = env.apply_action(obs, constants.UP)
    expected = {
        "start": (0, 0),
        "player": (max(x - 1, 0), y),
        "cliffs": set([]),
        "exits": set([]),
        "size": (grids.GRID_HEIGHT, grids.GRID_WIDTH),
    }
    assert output_observation == expected
    assert output_reward == -1.0


@hypothesis.given(
    x=st.integers(min_value=0, max_value=grids.GRID_HEIGHT - 1),
    y=st.integers(min_value=0, max_value=grids.GRID_WIDTH - 1),
)
def test_apply_action_going_down(x: int, y: int):
    obs = {
        "player": (x, y),
        "cliffs": set([]),
        "exits": set([]),
        "size": (grids.GRID_HEIGHT, grids.GRID_WIDTH),
    }
    output_observation, output_reward = env.apply_action(obs, constants.DOWN)
    expected_observation = {
        "player": (min(x + 1, grids.GRID_HEIGHT - 1), y),
        "cliffs": set([]),
        "exits": set([]),
        "size": (grids.GRID_HEIGHT, grids.GRID_WIDTH),
    }

    assert output_observation == expected_observation
    assert output_reward == -1.0


@hypothesis.given(
    x=st.integers(min_value=0, max_value=grids.GRID_HEIGHT - 1),
    y=st.integers(min_value=0, max_value=grids.GRID_WIDTH - 1),
)
def test_apply_action_going_left(x: int, y: int):
    obs = {
        "player": (x, y),
        "cliffs": set([]),
        "exits": set([]),
        "size": (grids.GRID_HEIGHT, grids.GRID_WIDTH),
    }
    output_observation, output_reward = env.apply_action(obs, constants.LEFT)
    expected = {
        "player": (x, max(0, y - 1)),
        "cliffs": set([]),
        "exits": set([]),
        "size": (grids.GRID_HEIGHT, grids.GRID_WIDTH),
    }
    assert output_observation == expected
    assert output_reward == -1.0


@hypothesis.given(
    x=st.integers(min_value=0, max_value=grids.GRID_HEIGHT - 1),
    y=st.integers(min_value=0, max_value=grids.GRID_WIDTH - 1),
)
def test_apply_action_going_right(x: int, y: int):
    obs = {
        "player": (x, y),
        "cliffs": set([]),
        "exits": set([]),
        "size": (grids.GRID_HEIGHT, grids.GRID_WIDTH),
    }
    output_observation, output_reward = env.apply_action(obs, constants.RIGHT)
    expected = {
        "player": (x, min(y + 1, grids.GRID_WIDTH - 1)),
        "cliffs": set([]),
        "exits": set([]),
        "size": (grids.GRID_HEIGHT, grids.GRID_WIDTH),
    }
    assert output_observation == expected
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
    output = env.create_observation(
        size=(100, 100),
        start=starting_pos,
        player=starting_pos,
        cliffs=cliffs,
        exits=exits,
    )
    expected = {
        "start": starting_pos,
        "player": starting_pos,
        "cliffs": set(cliffs),
        "exits": set(exits),
        "size": (100, 100),
    }
    assert output == expected


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
    state_id_fn = env.create_state_id_fn(states_mapping)
    for state, _id in states_mapping.items():
        # duck type an observation
        assert state_id_fn({"player": state}) == _id


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
    position_from_state_id_fn = env.create_position_from_state_id_fn(states_mapping)
    for state, _id in states_mapping.items():
        assert position_from_state_id_fn(_id) == state


def test_as_grid():
    # soxo
    # ooog
    observation = env.create_observation(
        size=(2, 4), start=(0, 0), player=(0, 0), cliffs=[(0, 2)], exits=[(1, 3)]
    )

    output = env.as_grid(observation)
    expected = np.array(
        [
            np.array([[1, 0, 0, 0], [0, 0, 0, 0]]),
            np.array([[0, 0, 1, 0], [0, 0, 0, 0]]),
            np.array([[0, 0, 0, 0], [0, 0, 0, 1]]),
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(output, expected)
