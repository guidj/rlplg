import tempfile
import uuid

import hypothesis
import numpy as np
import pytest
from hypothesis import strategies as st

from rlplg import envdesc
from rlplg.learning.tabular import empiricmdp, policies
from tests import defaults


def test_inferred_mdp_init():
    mdp_functions = empiricmdp.MdpFunctions(transition={}, reward={})
    env_desc = envdesc.EnvDesc(num_states=0, num_actions=0)

    inferred_mdp = empiricmdp.InferredMdp(
        mdp_functions=mdp_functions, env_desc=env_desc
    )

    assert inferred_mdp.env_desc() == env_desc


def test_inferred_mdp_transition_probability_with_defined_transitions():
    transitions = {(0, 0, 0): 0.8, (0, 0, 1): 0.2, (1, 0, 1): 1.0}
    mdp_functions = empiricmdp.MdpFunctions(transition=transitions, reward={})
    env_desc = envdesc.EnvDesc(num_states=0, num_actions=0)

    inferred_mdp = empiricmdp.InferredMdp(
        mdp_functions=mdp_functions, env_desc=env_desc
    )

    assert inferred_mdp.transition_probability(0, 0, 0) == 0.8
    assert inferred_mdp.transition_probability(0, 0, 1) == 0.2
    assert inferred_mdp.transition_probability(1, 0, 1) == 1.0


@hypothesis.given(
    st.tuples(st.integers(), st.integers(), st.integers()).filter(
        lambda x: x not in {(0, 0, 0), (0, 0, 1), (1, 0, 1)}
    )
)
def test_inferred_mdp_transition_probability_with_undefined_transitions(
    state_action_next_state,
):
    transitions = {(0, 0, 0): 0.8, (0, 0, 1): 0.2, (1, 0, 1): 1.0}
    mdp_functions = empiricmdp.MdpFunctions(transition=transitions, reward={})
    env_desc = envdesc.EnvDesc(num_states=0, num_actions=0)

    inferred_mdp = empiricmdp.InferredMdp(
        mdp_functions=mdp_functions, env_desc=env_desc
    )
    state, action, next_state = state_action_next_state
    assert inferred_mdp.transition_probability(state, action, next_state) == 0.0


def test_inferred_mdp_reward_with_defined_transitions():
    rewards = {(0, 0, 0): -2.0, (0, 0, 1): -1.0, (1, 0, 1): 0.0}
    mdp_functions = empiricmdp.MdpFunctions(transition={}, reward=rewards)
    env_desc = envdesc.EnvDesc(num_states=0, num_actions=0)

    inferred_mdp = empiricmdp.InferredMdp(
        mdp_functions=mdp_functions, env_desc=env_desc
    )

    assert inferred_mdp.reward(0, 0, 0) == -2.0
    assert inferred_mdp.reward(0, 0, 1) == -1.0
    assert inferred_mdp.reward(1, 0, 1) == 0.0


@hypothesis.given(
    st.tuples(st.integers(), st.integers(), st.integers()).filter(
        lambda x: x not in {(0, 0, 0), (0, 0, 1), (1, 0, 1)}
    )
)
def test_inferred_mdp_reward_with_undefined_transitions(
    state_action_next_state,
):
    rewards = {(0, 0, 0): -2.0, (0, 0, 1): -1.0, (1, 0, 1): 0.0}
    mdp_functions = empiricmdp.MdpFunctions(transition={}, reward=rewards)
    env_desc = envdesc.EnvDesc(num_states=0, num_actions=0)

    inferred_mdp = empiricmdp.InferredMdp(
        mdp_functions=mdp_functions, env_desc=env_desc
    )
    state, action, next_state = state_action_next_state
    assert inferred_mdp.transition_probability(state, action, next_state) == 0.0


def test_collect_mdp_stats():
    environment = defaults.CountEnv()
    mdp = defaults.CountEnvMDP()
    policy = policies.PyRandomPolicy(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
        num_actions=mdp.env_desc().num_actions,
    )

    # (state,action) -> Mapping[next_state, visits]
    output = empiricmdp.collect_mdp_stats(
        environment=environment,
        policy=policy,
        state_id_fn=defaults.item,
        action_id_fn=defaults.item,
        num_episodes=100,
        logging_frequency_episodes=1,
    )
    flattened_transitions = {
        (state, action, next_state): visits
        for ((state, action), next_states) in output.transitions.items()
        for next_state, visits in next_states.items()
    }
    flattened_rewards = {
        (state, action, next_state): rewards
        for ((state, action), next_states) in output.rewards.items()
        for next_state, rewards in next_states.items()
    }

    assert len(output.transitions) == len(output.rewards)
    assert set(output.transitions.keys()) == set(output.rewards.keys())
    assert len(flattened_transitions) == len(flattened_rewards)
    # all visits counters are positive
    assert np.sum(
        (np.array(list(flattened_transitions.values())) > 0).astype(int)
    ) == len(flattened_transitions)


def test_aggregate_stats():
    inputs = [
        empiricmdp.MdpStats(
            transitions={(0, 0): {0: 8, 1: 2}},
            rewards={(0, 0): {0: -16.0, 1: -2.0}},
        ),
        empiricmdp.MdpStats(transitions={(1, 0): {1: 10}}, rewards={(1, 0): {1: 0.0}}),
    ]
    expected = empiricmdp.MdpStats(
        transitions={(0, 0): {0: 8, 1: 2}, (1, 0): {1: 10}},
        rewards={(0, 0): {0: -16.0, 1: -2.0}, (1, 0): {1: 0.0}},
    )
    output = empiricmdp.aggregate_stats(inputs)

    assert output == expected


def test_create_mdp_functions():
    mdp_stats = empiricmdp.MdpStats(
        transitions={(0, 0): {0: 8, 1: 2}, (1, 0): {1: 10}},
        rewards={(0, 0): {0: -16.0, 1: -2.0}, (1, 0): {1: 0.0}},
    )
    expected = empiricmdp.MdpFunctions(
        transition={(0, 0, 0): 0.8, (0, 0, 1): 0.2, (1, 0, 1): 1.0},
        reward={(0, 0, 0): -2, (0, 0, 1): -1.0, (1, 0, 1): 0.0},
    )
    output = empiricmdp.create_mdp_functions(mdp_stats)

    assert output == expected


def test_export_and_load_stats_roundtrip():
    mdp_stats = empiricmdp.MdpStats(
        transitions={(0, 0): {0: 8, 1: 2}, (1, 0): {1: 10}},
        rewards={(0, 0): {0: -16.0, 1: -2.0}, (1, 0): {1: 0.0}},
    )
    path = tempfile.gettempdir()
    filename = str(uuid.uuid4())
    empiricmdp.export_stats(path, filename=filename, mdp_stats=mdp_stats)
    output = empiricmdp.load_stats(path, filename)

    assert output == mdp_stats


def test_export_stats_with_bad_path():
    mdp_stats = empiricmdp.MdpStats(
        transitions={(0, 0): {0: 8, 1: 2}, (1, 0): {1: 10}},
        rewards={(0, 0): {0: -16.0, 1: -2.0}, (1, 0): {1: 0.0}},
    )
    path = "unrealistic-path"
    filename = str(uuid.uuid4())

    with pytest.raises(IOError):
        empiricmdp.export_stats(path, filename=filename, mdp_stats=mdp_stats)
