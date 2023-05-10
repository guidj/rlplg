"""
Policy evaluation methods tests.

Note: once the terminal state is entered, the MDP trajectory ends.
So only rewards from the transitions up until the terminal state matter.
Hence, no actions taken in the terminal state are used
in policy evaluation algorithms.

"""
import numpy as np
import pytest

from rlplg import core
from rlplg.learning.opt import schedules
from rlplg.learning.tabular import policies
from rlplg.learning.tabular.evaluation import onpolicy
from tests import defaults


def test_first_visit_monte_carlo_action_values_with_one_episode(
    environment: core.PyEnvironment,
    policy: core.PyPolicy,
):
    """
    At each step, except the last, there are value updates.
    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    Reversed: (3, 1), (2, 1), (1, 1), (0, 1)
    Reversed Rewards: 0, -1, -1, -1
    (undiscoutned) Returns: 0, -1, -1, -2
    (gamma=0.95) Returns: 0, 0, -1, -1.95

    """

    results = onpolicy.first_visit_monte_carlo_action_values(
        policy=policy,
        environment=environment,
        num_episodes=1,
        gamma=0.95,
        state_id_fn=defaults.item,
        action_id_fn=defaults.item,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    steps, qtable = next(iter(output))
    assert steps == 4
    np.testing.assert_array_almost_equal(
        qtable, np.array([[0, -2.8525], [0, -1.95], [0, -1], [0, 0]])
    )


def test_first_visit_monte_carlo_action_values_with_one_episode_convering_every_action(
    environment: core.PyEnvironment,
):
    """
    At each step, except the last, there are value updates.

    First moves are wrong, the ones that follow are right.
    init Q(s,a) = 1
    G = 0
    Q(s,a) = Q(s,a) + [R + gamma * G]

    Trajectory: (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0)
    Reversed: (3, 0), (2, 1), (2, 0), (1, 1), (1, 0), (0, 1), (0, 0)
    Reversed Rewards: 0, 0, -10, -1, -10, -1, -10
    (undiscoutned) Returns: 0, 0, -10, -11, -21, -22, -32
    (gamma=0.95) Returns: 0, 0, -10, -10.5, -19.975, -19.97625, -28.9774375

    ...
    """
    stochastic_policy = defaults.RoundRobinActionsPolicy(
        actions=[0, 1, 0, 1, 0, 1],
    )

    results = onpolicy.first_visit_monte_carlo_action_values(
        policy=stochastic_policy,
        environment=environment,
        num_episodes=1,
        gamma=0.95,
        state_id_fn=defaults.item,
        action_id_fn=defaults.item,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    steps, qtable = next(iter(output))
    assert steps == 7
    np.testing.assert_array_almost_equal(
        qtable,
        np.array(
            [[-29.751219, -20.790756], [-20.832375, -11.4025], [-10.95, -1], [0, 0]]
        ),
    )


def test_sarsa_action_values_with_one_episode(
    environment: core.PyEnvironment,
    policy: core.PyPolicy,
):
    """
    At each step, except the last, there are value updates.
    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    init Q(s,a) = 0
    Q(0,0) = Q(0,0) + 0.1 * [R + 0.95 * Q(0,1) - Q(0,0)]
    Q(0,0) = 0 + 0.1 * [-10 + 0.95 * 0 - 0]``
    Q(0,0) = 0 + 0.1 * (-10) = -0.1
    """

    results = onpolicy.sarsa_action_values(
        policy=policy,
        environment=environment,
        num_episodes=1,
        lrs=schedules.LearningRateSchedule(
            initial_learning_rate=0.1, schedule=constant_learning_rate
        ),
        gamma=0.95,
        state_id_fn=defaults.item,
        action_id_fn=defaults.item,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    steps, qtable = next(iter(output))
    assert steps == 4
    np.testing.assert_array_almost_equal(
        qtable, np.array([[0, -0.1], [0, -0.1], [0, -0.1], [0, 0]])
    )


def test_sarsa_action_values_with_one_episode_convering_every_action(
    environment: core.PyEnvironment,
):
    """
    At each step, except the last, there are value updates.

    First moves are wrong, the ones that follow are right.
    init Q(s,a) = 0
    Q(s,a) = Q(s,a) + alpha * [R + gamma * Q(s',a') - Q(s,a)]
    Q(0,0) = Q(0,0) + 0.1 * [R + 0.95 * Q(0,1) - Q(0,0)]
    Q(0,0) = 0 + 0.1 * [-10 + 0.95 * 0 - 0]
    Q(0,0) = 0 + 0.1 * (-10) = -0.1
    ...
    """
    stochastic_policy = defaults.RoundRobinActionsPolicy(
        actions=[0, 1, 0, 1, 0, 1, 0],
    )

    results = onpolicy.sarsa_action_values(
        policy=stochastic_policy,
        environment=environment,
        num_episodes=1,
        lrs=schedules.LearningRateSchedule(
            initial_learning_rate=0.1, schedule=constant_learning_rate
        ),
        gamma=0.95,
        state_id_fn=defaults.identity,
        action_id_fn=defaults.identity,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    steps, qtable = next(iter(output))
    assert steps == 7
    np.testing.assert_array_almost_equal(
        qtable, np.array([[-1, -0.1], [-1, -0.1], [-1, -0.1], [0, 0]])
    )


def test_first_visit_monte_carlo_state_values_with_one_episode(
    environment: core.PyEnvironment,
    policy: core.PyPolicy,
):
    """
    Every (state, action) pair is updated, since
    the behavior and target policies match on every step.

    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    Reversed: (3, 1), (2, 1), (1, 1), (0, 1)
    Reversed Rewards: 0, -1, -1, -1
    Reversed Returns: 0, -1, -2, -3
    """

    results = onpolicy.first_visit_monte_carlo_state_values(
        policy=policy,
        environment=environment,
        num_episodes=1,
        gamma=1.0,
        state_id_fn=defaults.item,
        initial_values=np.zeros(shape=[4], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    steps, values = next(iter(output))
    assert steps == 4
    np.testing.assert_array_equal(values, [-3, -2, -1, 0])


def test_first_visit_monte_carlo_state_values_with_two_episodes(
    environment: core.PyEnvironment,
    policy: core.PyPolicy,
):
    """
    Every (state, action) pair is updated, since
    the behavior and target policies match on every step.

    Since the rewards are the same at each episode,
    the average should be the same.

    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    Reversed: (3, 1), (2, 1), (1, 1), (0, 1)
    Reversed Rewards: 0, -1, -1, -1
    Reversed Returns: 0, -1, -2, -3
    """

    results = onpolicy.first_visit_monte_carlo_state_values(
        policy=policy,
        environment=environment,
        num_episodes=2,
        gamma=1.0,
        state_id_fn=defaults.item,
        initial_values=np.zeros(shape=[4], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 2
    output_iter = iter(output)
    steps, values = next(output_iter)
    assert steps == 4
    np.testing.assert_array_equal(values, [-3, -2, -1, 0])
    steps, values = next(output_iter)
    assert steps == 4
    np.testing.assert_array_equal(values, [-3, -2, -1, 0])


def test_one_step_td_state_values_with_one_episode(
    environment: core.PyEnvironment,
    policy: core.PyPolicy,
):
    """
    Every (state, action) pair is updated, since
    the behavior and target policies match on every step.

    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    Rewards: -1, -1, -1, 0
    """

    results = onpolicy.one_step_td_state_values(
        policy=policy,
        environment=environment,
        num_episodes=1,
        lrs=schedules.LearningRateSchedule(
            initial_learning_rate=0.1, schedule=constant_learning_rate
        ),
        gamma=1.0,
        state_id_fn=defaults.item,
        initial_values=np.zeros(shape=[4], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    steps, values = next(iter(output))
    assert steps == 4
    np.testing.assert_allclose(values, [-0.1, -0.1, -0.1, 0])


def test_one_step_td_state_values_with_two_episodes(
    environment: core.PyEnvironment,
    policy: core.PyPolicy,
):
    """
    Every (state, action) pair is updated, since
    the behavior and target policies match on every step.

    Since the rewards are the same at each episode,
    the average should be the same.

    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    Rewards: -1, -1, -1, 0

    Episode 1:
    V(s) = V(s) + alpha * (r + gamma * V(s') - V(s))
    V(0) = V(0) + 0.1 * (-1 + 1.0 * V(1) - V(0))
    V(0) = 0 + 0.1 * (-1 + 1.0 * 0 - 0)
    V(0) = 0 + (-0.1) = -0.1
    ..
    V(1) = -0.1
    ...
    V(2) = V(2) + 0.1 * (-1 + 1.0 * V(3) - V(2))
    V(2) = 0 + 0.1 * (-1 + 1.0 * 0 - 0)
    V(2) = -0.1

    Episode 2:
    V(0) = V(0) + 0.1 * (-1 + 1.0 * V(1) - V(0))
    V(0) = -0.1 + 0.1 * (-1 + 1.0 * -0.1 - (-0.1))
    V(0) = -0.2
    ...
    V(2) = V(2) + 0.1 * (-1 + 1.0 * V(3) - V(2))
    V(2) = -0.1 + 0.1 * (-1 + 1.0 * 0 - (-0.1))
    V(2) = -0.19
    """

    results = onpolicy.one_step_td_state_values(
        policy=policy,
        environment=environment,
        num_episodes=2,
        lrs=schedules.LearningRateSchedule(
            initial_learning_rate=0.1, schedule=constant_learning_rate
        ),
        gamma=1.0,
        state_id_fn=defaults.item,
        initial_values=np.zeros(shape=[4], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 2
    output_iter = iter(output)
    steps, values = next(output_iter)
    assert steps == 4
    np.testing.assert_allclose(values, [-0.1, -0.1, -0.1, 0])
    steps, values = next(output_iter)
    assert steps == 4
    np.testing.assert_allclose(values, [-0.2, -0.2, -0.19, 0])


def constant_learning_rate(initial_lr: float, episode: int, step: int):
    del episode
    del step
    return initial_lr


@pytest.fixture(scope="function")
def policy(environment: core.PyEnvironment, qtable: np.ndarray) -> core.PyPolicy:
    return policies.PyQGreedyPolicy(
        state_id_fn=defaults.identity,
        action_values=qtable,
        emit_log_probability=True,
    )


@pytest.fixture(scope="function")
def qtable() -> np.ndarray:
    """
    Q-table for optimal actions.
    """
    return np.array([[-1, 0], [-1, 0], [-1, 0], [0, 0]], np.float32)


@pytest.fixture(scope="function")
def environment() -> core.PyEnvironment:
    """
    Test environment.
    """
    return defaults.CountEnv()
