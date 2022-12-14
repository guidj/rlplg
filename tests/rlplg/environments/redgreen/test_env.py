import random
from typing import Mapping, Sequence, Text

import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from rlplg.environments.redgreen import constants, env

VALID_ACTIONS = ["red", "green", "wait"]


@hypothesis.given(cure=st.lists(st.sampled_from(elements=VALID_ACTIONS), min_size=1))
def test_redgreen_init(cure: Sequence[Text]):
    cure_sequence = [constants.ACTION_NAME_MAPPING[step] for step in cure]
    environment = env.RedGreenSeq(cure)
    assert environment.cure_sequence == cure_sequence
    assert environment.action_spec() == action_spec()
    assert environment.observation_spec() == observation_spec(cure_sequence)


def test_redgreen_simple_sequence():
    cure = ["red", "green", "wait"]
    environment = env.RedGreenSeq(cure)
    assert_time_step(
        environment.reset(),
        ts.TimeStep(
            step_type=ts.StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation={
                "cure_sequence": [0, 1, 2],
                "position": 0,
            },
        ),
    )
    # final treatment step, prematurely
    assert_time_step(
        environment.step(2),
        ts.TimeStep(
            step_type=ts.StepType.MID,
            reward=-2.0,
            discount=1.0,
            observation={
                "cure_sequence": [0, 1, 2],
                "position": 0,
            },
        ),
    )
    # first treatment step
    assert_time_step(
        environment.step(0),
        ts.TimeStep(
            step_type=ts.StepType.MID,
            reward=-1.0,
            discount=1.0,
            observation={
                "cure_sequence": [0, 1, 2],
                "position": 1,
            },
        ),
    )
    # second treatment step
    assert_time_step(
        environment.step(1),
        ts.TimeStep(
            step_type=ts.StepType.MID,
            reward=-1.0,
            discount=1.0,
            observation={
                "cure_sequence": [0, 1, 2],
                "position": 2,
            },
        ),
    )
    # wrong treatment step
    assert_time_step(
        environment.step(0),
        ts.TimeStep(
            step_type=ts.StepType.MID,
            reward=-2.0,
            discount=1.0,
            observation={
                "cure_sequence": [0, 1, 2],
                "position": 2,
            },
        ),
    )
    # third and final treatment step
    assert_time_step(
        environment.step(2),
        ts.TimeStep(
            step_type=ts.StepType.LAST,
            reward=-1.0,
            discount=0.0,
            observation={
                "cure_sequence": [0, 1, 2],
                "position": 3,
            },
        ),
    )

    # another treatment step in the terminal state
    assert_time_step(
        environment.step(0),
        ts.TimeStep(
            step_type=ts.StepType.LAST,
            reward=0.0,
            discount=0.0,
            observation={
                "cure_sequence": [0, 1, 2],
                "position": 3,
            },
        ),
    )


@hypothesis.given(cure=st.lists(st.sampled_from(elements=VALID_ACTIONS), min_size=1))
def test_redgreen_render(cure: Sequence[Text]):
    environment = env.RedGreenSeq(cure)
    environment.reset()
    # starting point
    np.testing.assert_array_equal(
        environment.render("rgb_array"),
        [0] * len(cure),
    )
    # one move
    environment.step(constants.ACTION_NAME_MAPPING[cure[0]])
    np.testing.assert_array_equal(
        environment.render("rgb_array"), [1] + [0] * (len(cure) - 1)
    )


@hypothesis.given(cure=st.lists(st.sampled_from(elements=VALID_ACTIONS), min_size=1))
def test_redgreen_render_with_invalid_modes(cure: Sequence[Text]):
    modes = ("human",)
    environment = env.RedGreenSeq(cure)
    for mode in modes:
        with pytest.raises(NotImplementedError):
            environment.render(mode)


@hypothesis.given(
    cure_sequence=st.lists(
        st.sampled_from(elements=list(range(len(constants.ACTIONS)))), min_size=2
    )
)
def test_apply_action_with_correct_next_action(cure_sequence: Sequence[int]):
    obs = {
        "cure_sequence": cure_sequence,
        "position": 0,
    }
    output_obs, output_reward = env.apply_action(obs, cure_sequence[0])
    assert output_obs == {
        "cure_sequence": cure_sequence,
        "position": 1,
    }
    assert output_reward == -1.0


@hypothesis.given(
    cure_sequence=st.lists(
        st.sampled_from(elements=list(range(len(constants.ACTIONS)))), min_size=2
    )
)
def test_apply_action_with_wrong_next_action(cure_sequence: Sequence[int]):
    obs = {
        "cure_sequence": cure_sequence,
        "position": 0,
    }
    wrong_actions = [
        action for action in constants.ACTIONS if action != cure_sequence[0]
    ]
    wrong_action = random.sample(population=wrong_actions, k=1)[0]
    output_obs, output_reward = env.apply_action(obs, wrong_action)
    assert output_obs == {
        "cure_sequence": cure_sequence,
        "position": 0,
    }
    assert output_reward == -2.0


@hypothesis.given(
    cure_sequence=st.lists(
        st.sampled_from(elements=list(range(len(constants.ACTIONS)))), min_size=2
    )
)
def test_apply_action_with_action_going_into_terminal_state(
    cure_sequence: Sequence[int],
):
    obs = {
        "cure_sequence": cure_sequence,
        "position": len(cure_sequence) - 1,
    }
    output_obs, output_reward = env.apply_action(obs, cure_sequence[-1])
    assert output_obs == {
        "cure_sequence": cure_sequence,
        "position": len(cure_sequence),
    }
    assert output_reward == -1.0


@hypothesis.given(
    cure_sequence=st.lists(
        st.sampled_from(elements=list(range(len(constants.ACTIONS)))), min_size=2
    ),
    action=st.integers(min_value=0, max_value=len(constants.ACTIONS)),
)
def test_apply_action_with_env_in_terminal_state(
    cure_sequence: Sequence[int], action: int
):
    obs = {
        "cure_sequence": cure_sequence,
        "position": len(cure_sequence),
    }
    output_obs, output_reward = env.apply_action(obs, action)
    assert output_obs == {
        "cure_sequence": cure_sequence,
        "position": len(cure_sequence),
    }
    assert output_reward == 0.0


@hypothesis.given(
    cure_sequence=st.lists(
        st.sampled_from(elements=list(range(len(constants.ACTIONS)))), min_size=2
    )
)
def test_beginning_state(cure_sequence: Sequence[int]):
    output = env.beginning_state(cure_sequence)
    assert output == {"cure_sequence": cure_sequence, "position": 0}


def test_is_finished():
    assert not env.is_finished({"cure_sequence": [0], "position": 0})
    assert env.is_finished({"cure_sequence": [0], "position": 1})
    assert not env.is_finished({"cure_sequence": [0, 1, 0], "position": 0})
    assert not env.is_finished({"cure_sequence": [0, 1, 0], "position": 1})
    assert env.is_finished({"cure_sequence": [0, 1, 0], "position": 3})


def test_get_state_id():
    assert env.get_state_id({"position": 0}) == 0
    assert env.get_state_id({"position": 1}) == 1
    assert env.get_state_id({"position": 2}) == 2


@hypothesis.given(
    cure_sequence=st.lists(
        st.sampled_from(elements=list(range(len(constants.ACTIONS))))
    ),
    pos=st.integers(),
)
def test_state_observation(cure_sequence: Sequence[int], pos: int):
    assert env.state_observation(state_id=pos, cure_sequence=cure_sequence)


def test_state_representation():
    assert env.state_representation({"cure_sequence": [0, 1, 0], "position": 0}) == [
        0,
        0,
        0,
    ]
    assert env.state_representation({"cure_sequence": [0, 1, 0], "position": 1}) == [
        1,
        0,
        0,
    ]
    assert env.state_representation({"cure_sequence": [0, 1, 0], "position": 2}) == [
        1,
        1,
        0,
    ]
    assert env.state_representation({"cure_sequence": [0, 1, 0], "position": 3}) == [
        1,
        1,
        1,
    ]


def action_spec() -> array_spec.BoundedArraySpec:
    return array_spec.BoundedArraySpec(
        shape=(),
        dtype=np.int32,
        minimum=0,
        maximum=2,
        name="action",
    )


def observation_spec(
    cure_actions: Sequence[int],
) -> Mapping[Text, array_spec.BoundedArraySpec]:
    return {
        "cure_sequence": array_spec.BoundedArraySpec(
            shape=(len(cure_actions),),
            dtype=np.int32,
            minimum=[0] * len(cure_actions),
            maximum=[2] * len(cure_actions),
            name="cure_sequence",
        ),
        "position": array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=len(cure_actions),
            name="position",
        ),
    }


def assert_time_step(output: ts.TimeStep, expected: ts.TimeStep) -> None:
    assert output.step_type == expected.step_type
    assert output.reward == expected.reward
    assert output.discount == expected.discount
    assert len(output.observation) == len(expected.observation)
    assert output.observation["cure_sequence"] == expected.observation["cure_sequence"]
    assert output.observation["position"] == expected.observation["position"]
