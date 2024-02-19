import random
from typing import Any, Sequence

import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest
from gymnasium import spaces

from rlplg import core
from rlplg.core import TimeStep
from rlplg.environments import redgreen
from tests.rlplg import dynamics

VALID_ACTIONS = ["red", "green", "wait"]


@hypothesis.given(cure=st.lists(st.sampled_from(elements=VALID_ACTIONS), min_size=1))
@hypothesis.settings(deadline=None)
def test_redgreen_init(cure: Sequence[str]):
    cure_sequence = [redgreen.ACTION_NAME_MAPPING[step] for step in cure]
    environment = redgreen.RedGreenSeq(cure)
    assert environment.cure_sequence == cure_sequence
    assert environment.action_space == spaces.Discrete(3)
    assert environment.observation_space == spaces.Dict(
        {
            "cure_sequence": spaces.Box(
                low=np.zeros(len(cure)),
                high=np.array([2] * len(cure)),
                dtype=np.int64,
            ),
            "position": spaces.Box(low=0, high=len(cure), dtype=np.int64),
        }
    )
    dynamics.assert_transition_mapping(
        environment.transition,
        env_desc=core.EnvDesc(num_states=len(cure_sequence), num_actions=3),
    )


def test_redgreen_simple_sequence():
    cure = ["red", "green", "wait"]
    environment = redgreen.RedGreenSeq(cure)
    obs, info = environment.reset()
    assert_observation(
        obs,
        {
            "cure_sequence": [0, 1, 2],
            "position": 0,
        },
    )
    assert info == {}
    # last treatment step, prematurely
    assert_time_step(
        environment.step(2),
        (
            {
                "cure_sequence": [0, 1, 2],
                "position": 0,
            },
            -2.0,
            False,
            False,
            {},
        ),
    )
    # first treatment step
    assert_time_step(
        environment.step(0),
        (
            {
                "cure_sequence": [0, 1, 2],
                "position": 1,
            },
            -1.0,
            False,
            False,
            {},
        ),
    )
    # second treatment step
    assert_time_step(
        environment.step(1),
        (
            {
                "cure_sequence": [0, 1, 2],
                "position": 2,
            },
            -1.0,
            False,
            False,
            {},
        ),
    )
    # wrong treatment step
    assert_time_step(
        environment.step(0),
        (
            {
                "cure_sequence": [0, 1, 2],
                "position": 2,
            },
            -2.0,
            False,
            False,
            {},
        ),
    )
    # third and final treatment step
    assert_time_step(
        environment.step(2),
        (
            {
                "cure_sequence": [0, 1, 2],
                "position": 3,
            },
            -1.0,
            True,
            False,
            {},
        ),
    )

    # another treatment step in the terminal state
    assert_time_step(
        environment.step(0),
        (
            {
                "cure_sequence": [0, 1, 2],
                "position": 3,
            },
            0.0,
            True,
            False,
            {},
        ),
    )


@hypothesis.given(cure=st.lists(st.sampled_from(elements=VALID_ACTIONS), min_size=1))
def test_redgreen_render(cure: Sequence[str]):
    environment = redgreen.RedGreenSeq(cure, render_mode="rgb_array")
    environment.reset()
    # starting point
    np.testing.assert_array_equal(
        environment.render(),  # type: ignore
        [0] * len(cure),
    )
    # one move
    environment.step(redgreen.ACTION_NAME_MAPPING[cure[0]])
    np.testing.assert_array_equal(environment.render(), [1] + [0] * (len(cure) - 1))  # type: ignore


@hypothesis.given(cure=st.lists(st.sampled_from(elements=VALID_ACTIONS), min_size=1))
def test_redgreen_render_with_invalid_modes(cure: Sequence[str]):
    modes = ("human",)
    for mode in modes:
        environment = redgreen.RedGreenSeq(cure, render_mode=mode)
        environment.reset()
        with pytest.raises(NotImplementedError):
            environment.render()


def test_redgreenmdpdiscretizer():
    discretizer = redgreen.RedGreenMdpDiscretizer()
    assert (
        discretizer.state({"cure_sequence": ["green", "red", "wait"], "position": 0})
        == 0
    )
    assert (
        discretizer.state({"cure_sequence": ["green", "red", "wait"], "position": 1})
        == 1
    )
    assert (
        discretizer.state({"cure_sequence": ["green", "red", "wait"], "position": 2})
        == 2
    )

    assert discretizer.action(0) == 0
    assert discretizer.action(1) == 1
    assert discretizer.action(2) == 2


@hypothesis.given(
    cure_sequence=st.lists(
        st.sampled_from(elements=list(range(len(redgreen.ACTIONS)))), min_size=2
    )
)
def test_apply_action_with_correct_next_action(cure_sequence: Sequence[int]):
    obs = {
        "cure_sequence": cure_sequence,
        "position": 0,
    }
    output_obs, output_reward = redgreen.apply_action(obs, cure_sequence[0])
    assert output_obs == {
        "cure_sequence": cure_sequence,
        "position": 1,
    }
    assert output_reward == -1.0


@hypothesis.given(
    cure_sequence=st.lists(
        st.sampled_from(elements=list(range(len(redgreen.ACTIONS)))), min_size=2
    )
)
def test_apply_action_with_wrong_next_action(cure_sequence: Sequence[int]):
    obs = {
        "cure_sequence": cure_sequence,
        "position": 0,
    }
    wrong_actions = [
        action for action in redgreen.ACTIONS if action != cure_sequence[0]
    ]
    wrong_action = random.sample(population=wrong_actions, k=1)[0]
    output_obs, output_reward = redgreen.apply_action(obs, wrong_action)
    assert output_obs == {
        "cure_sequence": cure_sequence,
        "position": 0,
    }
    assert output_reward == -2.0


@hypothesis.given(
    cure_sequence=st.lists(
        st.sampled_from(elements=list(range(len(redgreen.ACTIONS)))), min_size=2
    )
)
def test_apply_action_with_action_going_into_terminal_state(
    cure_sequence: Sequence[int],
):
    obs = {
        "cure_sequence": cure_sequence,
        "position": len(cure_sequence) - 1,
    }
    output_obs, output_reward = redgreen.apply_action(obs, cure_sequence[-1])
    assert output_obs == {
        "cure_sequence": cure_sequence,
        "position": len(cure_sequence),
    }
    assert output_reward == -1.0


@hypothesis.given(
    cure_sequence=st.lists(
        st.sampled_from(elements=list(range(len(redgreen.ACTIONS)))), min_size=2
    ),
    action=st.integers(min_value=0, max_value=len(redgreen.ACTIONS)),
)
def test_apply_action_with_env_in_terminal_state(
    cure_sequence: Sequence[int], action: int
):
    obs = {
        "cure_sequence": cure_sequence,
        "position": len(cure_sequence),
    }
    output_obs, output_reward = redgreen.apply_action(obs, action)
    assert output_obs == {
        "cure_sequence": cure_sequence,
        "position": len(cure_sequence),
    }
    assert output_reward == 0.0


def test_is_finished():
    assert not redgreen.is_finished({"cure_sequence": [0], "position": 0})
    assert redgreen.is_finished({"cure_sequence": [0], "position": 1})
    assert not redgreen.is_finished({"cure_sequence": [0, 1, 0], "position": 0})
    assert not redgreen.is_finished({"cure_sequence": [0, 1, 0], "position": 1})
    assert redgreen.is_finished({"cure_sequence": [0, 1, 0], "position": 3})


def test_get_state_id():
    assert redgreen.get_state_id({"position": 0}) == 0
    assert redgreen.get_state_id({"position": 1}) == 1
    assert redgreen.get_state_id({"position": 2}) == 2


@hypothesis.given(
    cure_sequence=st.lists(
        st.sampled_from(elements=list(range(len(redgreen.ACTIONS)))), min_size=2
    )
)
def test_state_observation(cure_sequence: Sequence[int]):
    pos = random.randint(0, len(cure_sequence))
    assert redgreen.state_observation(position=pos, cure_sequence=cure_sequence) == {
        "cure_sequence": cure_sequence,
        "position": pos,
    }


@hypothesis.given(
    cure_sequence=st.lists(
        st.sampled_from(elements=list(range(len(redgreen.ACTIONS)))), min_size=2
    )
)
def test_state_observation_with_invalid_position(cure_sequence: Sequence[int]):
    pos = random.choice([-1, len(cure_sequence) + 1])
    with pytest.raises(ValueError):
        redgreen.state_observation(cure_sequence, position=pos)


def test_state_representation():
    assert redgreen.state_representation(
        {"cure_sequence": [0, 1, 0], "position": 0}
    ) == [
        0,
        0,
        0,
    ]
    assert redgreen.state_representation(
        {"cure_sequence": [0, 1, 0], "position": 1}
    ) == [
        1,
        0,
        0,
    ]
    assert redgreen.state_representation(
        {"cure_sequence": [0, 1, 0], "position": 2}
    ) == [
        1,
        1,
        0,
    ]
    assert redgreen.state_representation(
        {"cure_sequence": [0, 1, 0], "position": 3}
    ) == [
        1,
        1,
        1,
    ]


@hypothesis.given(
    cure=st.lists(st.sampled_from(elements=["red", "green", "wait"]), min_size=1)
)
@hypothesis.settings(deadline=None)
def test_create_env_spec(cure: Sequence[str]):
    env_spec = redgreen.create_env_spec(cure=cure)
    assert env_spec.name == "RedGreenSeq"
    assert len(env_spec.level) > 0
    assert isinstance(env_spec.environment, redgreen.RedGreenSeq)
    assert isinstance(env_spec.discretizer, redgreen.RedGreenMdpDiscretizer)
    assert env_spec.mdp.env_desc.num_states == len(cure) + 1
    assert env_spec.mdp.env_desc.num_actions == 3
    assert len(env_spec.mdp.transition) == len(cure)


def assert_time_step(output: TimeStep, expected: TimeStep) -> None:
    assert_observation(output[0], expected[0])
    assert output[1] == expected[1]
    assert output[2] is expected[2]
    assert output[3] is expected[3]
    assert output[4] == expected[4]


def assert_observation(output: Any, expected: Any) -> None:
    assert len(output) == 2
    assert output["cure_sequence"] == expected["cure_sequence"]
    assert output["position"] == expected["position"]
