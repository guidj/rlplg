import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest
from gymnasium import spaces

from rlplg import core
from rlplg.core import TimeStep
from rlplg.environments import towerhanoi
from tests.rlplg import dynamics


@hypothesis.given(disks=st.integers(min_value=1, max_value=4))
@hypothesis.settings(deadline=None)
def test_towerofhanoi_init(disks: int):
    environment = towerhanoi.TowerOfHanoi(num_disks=disks)
    assert environment.num_disks == disks
    assert environment.action_space == spaces.Discrete(6)
    assert environment.observation_space == spaces.Dict(
        {
            "id": spaces.Discrete(
                3**disks,
            ),
            "num_pegs": spaces.Box(low=3, high=3, dtype=np.int64),
            "towers": spaces.Tuple([spaces.Discrete(3) for _ in range(disks)]),
        }
    )
    dynamics.assert_transition_mapping(
        environment.transition,
        env_desc=core.EnvDesc(num_states=3**disks, num_actions=6),
    )


@hypothesis.given(disks=st.integers(min_value=10, max_value=100))
@hypothesis.settings(deadline=None)
def test_towerofhanoi_init_with_too_many_disks(disks: int):
    with pytest.raises(ValueError):
        towerhanoi.TowerOfHanoi(num_disks=disks)


def test_towerofhanoi_with_two_disks():
    environment = towerhanoi.TowerOfHanoi(num_disks=2)
    obs, info = environment.reset()
    assert obs == {
        "id": 0,
        "num_pegs": 3,
        "towers": (0, 0),
    }
    assert info == {}

    # Try to move from peg 2 to peg 1 (no change)
    # actions: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    assert_time_step(
        environment.step(5),
        (
            {
                "id": 0,
                "num_pegs": 3,
                "towers": (0, 0),
            },
            -1.0,
            False,
            False,
            {},
        ),
    )

    # Move from peg 0 to 1
    # actions: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    assert_time_step(
        environment.step(0),
        (
            {
                "id": 3,
                "num_pegs": 3,
                "towers": (1, 0),
            },
            -1.0,
            False,
            False,
            {},
        ),
    )

    # Move from peg 0 to 2
    # actions: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    assert_time_step(
        environment.step(1),
        (
            {
                "id": 5,
                "num_pegs": 3,
                "towers": (1, 2),
            },
            -1.0,
            False,
            False,
            {},
        ),
    )

    # Try to move from peg 2 (large disk) to peg 1 (small disk)
    # No change in state
    # actions: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    assert_time_step(
        environment.step(5),
        (
            {
                "id": 5,
                "num_pegs": 3,
                "towers": (1, 2),
            },
            -1.0,
            False,
            False,
            {},
        ),
    )

    # Complete game, by moving from peg 1 to peg 2
    # actions: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    assert_time_step(
        environment.step(3),
        (
            {
                "id": 8,
                "num_pegs": 3,
                "towers": (2, 2),
            },
            -1.0,
            True,
            False,
            {},
        ),
    )

    # Try any change to no avail
    # actions: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    for action in range(6):
        assert_time_step(
            environment.step(action),
            (
                {
                    "id": 8,
                    "num_pegs": 3,
                    "towers": (2, 2),
                },
                0.0,
                True,
                False,
                {},
            ),
        )


def test_towerofhanoi_with_three_disks():
    environment = towerhanoi.TowerOfHanoi(num_disks=3)
    obs, info = environment.reset()
    assert obs == {
        "id": 0,
        "num_pegs": 3,
        "towers": (0, 0, 0),
    }
    assert info == {}

    # Try to move from peg 2 to peg 1 (no change)
    # actions: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    assert_time_step(
        environment.step(5),
        (
            {
                "id": 0,
                "num_pegs": 3,
                "towers": (0, 0, 0),
            },
            -1.0,
            False,
            False,
            {},
        ),
    )

    # Move from peg 0 to peg 2
    # actions: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    assert_time_step(
        environment.step(1),
        (
            {
                "id": 18,
                "num_pegs": 3,
                "towers": (2, 0, 0),
            },
            -1.0,
            False,
            False,
            {},
        ),
    )

    # Move from peg 0 to peg 1
    # actions: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    assert_time_step(
        environment.step(0),
        (
            {
                "id": 21,
                "num_pegs": 3,
                "towers": (2, 1, 0),
            },
            -1.0,
            False,
            False,
            {},
        ),
    )

    # Try to move from peg 0 (large) to peg 1 (medium)
    # actions: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    assert_time_step(
        environment.step(0),
        (
            {
                "id": 21,
                "num_pegs": 3,
                "towers": (2, 1, 0),
            },
            -1.0,
            False,
            False,
            {},
        ),
    )

    # Move from peg 2 to peg 1
    # actions: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    assert_time_step(
        environment.step(5),
        (
            {
                "id": 12,
                "num_pegs": 3,
                "towers": (1, 1, 0),
            },
            -1.0,
            False,
            False,
            {},
        ),
    )

    # Move from peg 0 to peg 2
    # actions: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    assert_time_step(
        environment.step(1),
        (
            {
                "id": 14,
                "num_pegs": 3,
                "towers": (1, 1, 2),
            },
            -1.0,
            False,
            False,
            {},
        ),
    )

    # Move from peg 1 to peg 0
    # actions: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    assert_time_step(
        environment.step(2),
        (
            {
                "id": 5,
                "num_pegs": 3,
                "towers": (0, 1, 2),
            },
            -1.0,
            False,
            False,
            {},
        ),
    )

    # Move from peg 1 to peg 2
    # actions: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    assert_time_step(
        environment.step(3),
        (
            {
                "id": 8,
                "num_pegs": 3,
                "towers": (0, 2, 2),
            },
            -1.0,
            False,
            False,
            {},
        ),
    )

    # Move from peg 0 to peg 2
    # actions: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    assert_time_step(
        environment.step(1),
        (
            {
                "id": 26,
                "num_pegs": 3,
                "towers": (2, 2, 2),
            },
            -1.0,
            True,
            False,
            {},
        ),
    )

    # Try any change to no avail
    # actions: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    for action in range(6):
        assert_time_step(
            environment.step(action),
            (
                {
                    "id": 26,
                    "num_pegs": 3,
                    "towers": (2, 2, 2),
                },
                0.0,
                True,
                False,
                {},
            ),
        )


def test_towerofhanoi_render():
    environment = towerhanoi.TowerOfHanoi(num_disks=3)
    environment.reset()

    np.testing.assert_array_equal(environment.render(), np.array((0,) * 3))


def test_towerofhanoi_render_with_invalid_modes():
    modes = ("human",)
    for mode in modes:
        environment = towerhanoi.TowerOfHanoi(num_disks=3, render_mode=mode)
        environment.reset()
        with pytest.raises(NotImplementedError):
            environment.render()


def assert_time_step(output: TimeStep, expected: TimeStep) -> None:
    assert output[0] == expected[0]
    assert output[1] == expected[1]
    assert output[2] is expected[2]
    assert output[3] is expected[3]
    assert output[4] == expected[4]
