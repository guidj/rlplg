import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest
from gymnasium import spaces

from rlplg import core
from rlplg.core import TimeStep
from rlplg.environments import towerhanoi


@hypothesis.given(disks=st.integers(min_value=1, max_value=4))
@hypothesis.settings(deadline=None)
def test_towerofhanoi_init(disks: int):
    environment = towerhanoi.TowerOfHanoi(num_disks=disks)
    assert environment.num_disks == disks
    assert environment.action_space == spaces.Discrete(6)
    assert environment.observation_space == spaces.Dict(
        {
            "num_pegs": spaces.Box(low=3, high=3, dtype=np.int64),
            "state": spaces.Tuple([spaces.Discrete(3) for _ in range(disks)]),
        }
    )
    assert len(environment.transition) == 3**disks


@hypothesis.given(disks=st.integers(min_value=10, max_value=100))
@hypothesis.settings(deadline=None)
def test_towerofhanoi_init_with_too_many_disks(disks: int):
    with pytest.raises(ValueError):
        towerhanoi.TowerOfHanoi(num_disks=disks)


def test_towerofhanoi_with_two_disks():
    environment = towerhanoi.TowerOfHanoi(num_disks=2)
    obs, info = environment.reset()
    assert obs == {
        "num_pegs": 3,
        "state": (0, 0),
    }
    assert info == {}

    # Try to move from peg 2 to peg 1 (no change)
    # actions: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    assert_time_step(
        environment.step(5),
        (
            {
                "num_pegs": 3,
                "state": (0, 0),
            },
            -2.0,
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
                "num_pegs": 3,
                "state": (1, 0),
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
                "num_pegs": 3,
                "state": (1, 2),
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
                "num_pegs": 3,
                "state": (1, 2),
            },
            -2.0,
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
                "num_pegs": 3,
                "state": (2, 2),
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
                    "num_pegs": 3,
                    "state": (2, 2),
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
        "num_pegs": 3,
        "state": (0, 0, 0),
    }
    assert info == {}

    # Try to move from peg 2 to peg 1 (no change)
    # actions: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)
    assert_time_step(
        environment.step(5),
        (
            {
                "num_pegs": 3,
                "state": (0, 0, 0),
            },
            -2.0,
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
                "num_pegs": 3,
                "state": (2, 0, 0),
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
                "num_pegs": 3,
                "state": (2, 1, 0),
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
                "num_pegs": 3,
                "state": (2, 1, 0),
            },
            -2.0,
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
                "num_pegs": 3,
                "state": (1, 1, 0),
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
                "num_pegs": 3,
                "state": (1, 1, 2),
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
                "num_pegs": 3,
                "state": (0, 1, 2),
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
                "num_pegs": 3,
                "state": (0, 2, 2),
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
                "num_pegs": 3,
                "state": (2, 2, 2),
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
                    "num_pegs": 3,
                    "state": (2, 2, 2),
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

    discretizer = towerhanoi.TowerOfHanoiMdpDiscretizer()
    assert discretizer.state({"num_pegs": 3, "state": (0, 0)}) == 0
    assert discretizer.state({"num_pegs": 3, "state": (0, 1)}) == 1
    assert discretizer.state({"num_pegs": 3, "state": (0, 2)}) == 2
    assert discretizer.state({"num_pegs": 3, "state": (1, 0)}) == 3
    assert discretizer.state({"num_pegs": 3, "state": (1, 1)}) == 4
    assert discretizer.state({"num_pegs": 3, "state": (1, 2)}) == 5
    assert discretizer.state({"num_pegs": 3, "state": (2, 0)}) == 6
    assert discretizer.state({"num_pegs": 3, "state": (2, 1)}) == 7
    assert discretizer.state({"num_pegs": 3, "state": (2, 2)}) == 8

    assert discretizer.action(0) == 0
    assert discretizer.action(1) == 1
    assert discretizer.action(2) == 2


def test_create_env_spec():
    num_disks = 3
    output = towerhanoi.create_env_spec(num_disks=num_disks)
    assert isinstance(output, core.EnvSpec)
    assert output.name == "TowerOfHanoi"
    assert output.level == str(num_disks)
    assert isinstance(output.discretizer, towerhanoi.TowerOfHanoiMdpDiscretizer)
    assert output.mdp.env_desc.num_states == 3**num_disks
    assert output.mdp.env_desc.num_actions == 6
    assert len(output.mdp.transition) == 3**num_disks


def assert_time_step(output: TimeStep, expected: TimeStep) -> None:
    assert output[0] == expected[0]
    assert output[1] == expected[1]
    assert output[2] is expected[2]
    assert output[3] is expected[3]
    assert output[4] == expected[4]
