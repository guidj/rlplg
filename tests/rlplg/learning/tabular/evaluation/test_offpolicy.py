"""
Policy evaluation methods tests.

Note: terminal states should be initialized to zero.
We ignore that in some tests to verify the computation.
"""
import numpy as np
import pytest

from rlplg import core
from rlplg.learning.opt import schedules
from rlplg.learning.tabular import policies
from rlplg.learning.tabular.evaluation import offpolicy
from tests import defaults


def test_monte_carlo_action_values_with_one_episode(
    environment: core.PyEnvironment,
    policy: core.PyPolicy,
):
    """
    Every (state, action) pair is updated, since
    the behavior and target policies match on every step.

    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    Reversed: (3, 1), (2, 1), (1, 1), (0, 1)
    """

    results = offpolicy.monte_carlo_action_values(
        policy=policy,
        collect_policy=policy,
        environment=environment,
        num_episodes=1,
        gamma=1.0,
        policy_probability_fn=policy_prob_fn,
        collect_policy_probability_fn=collect_policy_prob_fn,
        state_id_fn=defaults.identity,
        action_id_fn=defaults.identity,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    steps, qtable = next(iter(output))
    assert steps == 4
    np.testing.assert_array_equal(qtable, [[0, -3], [0, -2], [0, -1], [0, 0]])


def test_monte_carlo_action_values_with_two_episodes(
    environment: core.PyEnvironment,
    policy: core.PyPolicy,
):
    """
    Every (state, action) pair is updated, since
    the behavior and target policies match on every step.
    After one episode, the value for the best actions match.

    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    Reversed: (3, 1), (2, 1), (1, 1), (0, 1)
    """

    results = offpolicy.monte_carlo_action_values(
        policy=policy,
        collect_policy=policy,
        environment=environment,
        num_episodes=2,
        gamma=1.0,
        policy_probability_fn=policy_prob_fn,
        collect_policy_probability_fn=collect_policy_prob_fn,
        state_id_fn=defaults.identity,
        action_id_fn=defaults.identity,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    output_iter = iter(output)
    assert len(output) == 2
    steps, qtable = next(output_iter)
    assert steps == 4
    np.testing.assert_array_equal(qtable, [[0, -3], [0, -2], [0, -1], [0, 0]])
    steps, qtable = next(output_iter)
    assert steps == 4
    np.testing.assert_array_equal(qtable, [[0, -3], [0, -2], [0, -1], [0, 0]])


def test_monte_carlo_action_values_with_one_episode_covering_every_action(
    environment: core.PyEnvironment,
    policy: core.PyPolicy,
):
    """
    Only the last two state-action pairs are updated because
    after that, the 2nd to last action mistmatches the policy,
    turning W into zero - i.e. the importance sample ratio collapses.
    """

    collect_policy = defaults.RoundRobinActionsPolicy(
        actions=[0, 1, 0, 1, 0, 1],
    )
    results = offpolicy.monte_carlo_action_values(
        policy=policy,
        collect_policy=collect_policy,
        environment=environment,
        num_episodes=1,
        gamma=1.0,
        policy_probability_fn=policy_prob_fn,
        collect_policy_probability_fn=collect_policy_prob_fn,
        state_id_fn=defaults.identity,
        action_id_fn=defaults.identity,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    steps, qtable = next(iter(output))
    assert steps == 7
    # we only learn about the last two (state, action) pairs.
    np.testing.assert_array_equal(qtable, [[0, 0], [0, 0], [-11, -1], [0, 0]])


def test_monte_carlo_action_values_step_with_reward_discount():
    """
    G <- 0.9*10 + 100 = 109
    C(s,a) <- 50 + 2 = 52
    Q(s,a) <- 20 + 2/52 * (109-20) = 255/13
    W <- 2 * 0.8 = 1.6

    """
    mc_update = offpolicy.monte_carlo_action_values_step(
        reward=100, returns=10, cu_sum=50, weight=2.0, value=20, rho=0.8, gamma=0.9
    )

    np.testing.assert_approx_equal(mc_update.returns, 109)
    np.testing.assert_approx_equal(mc_update.cu_sum, 52)
    np.testing.assert_approx_equal(mc_update.value, 609 / 26)
    np.testing.assert_approx_equal(mc_update.weight, 1.6)


def test_nstep_sarsa_action_values_with_one_nstep_and_one_episode(
    environment: core.PyEnvironment,
    policy: core.PyPolicy,
):
    """
    Each step value updates.
    Trajectory: (0, 1), (1, 1), (2, 1), (3, 1)
    """

    results = offpolicy.nstep_sarsa_action_values(
        policy=policy,
        collect_policy=policy,
        environment=environment,
        num_episodes=1,
        lrs=schedules.LearningRateSchedule(
            initial_learning_rate=0.1, schedule=constant_learning_rate
        ),
        gamma=0.95,
        nstep=1,
        policy_probability_fn=policy_prob_fn,
        collect_policy_probability_fn=collect_policy_prob_fn,
        state_id_fn=defaults.identity,
        action_id_fn=defaults.identity,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    steps, qtable = next(iter(output))
    assert steps == 4
    np.testing.assert_array_almost_equal(
        qtable, np.array([[0, -0.1], [0, -0.1], [0, -0.1], [0, 0]])
    )


def test_nstep_sarsa_action_values_with_two_nsteps_and_two_episodes(
    environment: core.PyEnvironment,
    policy: core.PyPolicy,
):
    """
    Every step is updated after n+2 steps, so the final step isn't updated.
    """
    results = offpolicy.nstep_sarsa_action_values(
        policy=policy,
        collect_policy=policy,
        environment=environment,
        num_episodes=2,
        lrs=schedules.LearningRateSchedule(
            initial_learning_rate=0.1, schedule=constant_learning_rate
        ),
        gamma=0.95,
        nstep=2,
        policy_probability_fn=policy_prob_fn,
        collect_policy_probability_fn=collect_policy_prob_fn,
        state_id_fn=defaults.identity,
        action_id_fn=defaults.identity,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    output_iter = iter(output)
    assert len(output) == 2
    steps, qtable = next(output_iter)
    assert steps == 4
    np.testing.assert_array_almost_equal(
        qtable, np.array([[0, -0.195], [0, -0.195], [0, -0.1], [0, 0]])
    )
    steps, qtable = next(output_iter)
    assert steps == 4
    np.testing.assert_array_almost_equal(
        qtable, np.array([[0, -0.379525], [0, -0.3705], [0, -0.19], [0, 0]])
    )


def test_nstep_sarsa_action_values_with_one_nstep_and_one_episode_covering_every_action(
    environment: core.PyEnvironment,
    policy: core.PyPolicy,
):
    """
    Every step preceding a correct step is updated.
    Every step following a mistep isn't.
    """
    collect_policy = defaults.RoundRobinActionsPolicy(
        actions=[0, 1, 0, 1, 0, 1],
    )

    results = offpolicy.nstep_sarsa_action_values(
        policy=policy,
        collect_policy=collect_policy,
        environment=environment,
        num_episodes=1,
        lrs=schedules.LearningRateSchedule(
            initial_learning_rate=0.1, schedule=constant_learning_rate
        ),
        gamma=0.95,
        nstep=1,
        policy_probability_fn=policy_prob_fn,
        collect_policy_probability_fn=collect_policy_prob_fn,
        state_id_fn=defaults.identity,
        action_id_fn=defaults.identity,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    steps, qtable = next(iter(output))
    assert steps == 7
    np.testing.assert_array_almost_equal(
        qtable, np.array([[-2, 0], [-2, 0], [-2, -0.2], [0, 0]])
    )


def test_nstep_sarsa_action_values_with_two_nsteps_and_one_episode_covering_every_action(
    environment: core.PyEnvironment,
    policy: core.PyPolicy,
):
    """
    No step gets updated.
    """
    collect_policy = defaults.RoundRobinActionsPolicy(
        actions=[0, 1, 0, 1, 0, 1],
    )

    results = offpolicy.nstep_sarsa_action_values(
        policy=policy,
        collect_policy=collect_policy,
        environment=environment,
        num_episodes=1,
        lrs=schedules.LearningRateSchedule(
            initial_learning_rate=0.1, schedule=constant_learning_rate
        ),
        gamma=0.95,
        nstep=2,
        policy_probability_fn=policy_prob_fn,
        collect_policy_probability_fn=collect_policy_prob_fn,
        state_id_fn=defaults.identity,
        action_id_fn=defaults.identity,
        initial_qtable=np.zeros(shape=[4, 2], dtype=np.float32),
    )

    output = list(results)
    assert len(output) == 1
    steps, qtable = next(iter(output))
    assert steps == 7
    np.testing.assert_array_almost_equal(
        qtable, np.array([[0, 0], [0, 0], [-4.38, -0.2], [0, 0]])
    )


def policy_prob_fn(policy: core.PyPolicy, traj: core.Trajectory) -> float:
    """The policy we're evaluating is assumed to be greedy w.r.t. Q(s, a).
    So the best action has probability 1.0, and all the others 0.0.
    """

    time_step = core.TimeStep(
        step_type=traj.step_type,
        reward=traj.reward,
        discount=traj.discount,
        observation=traj.observation,
    )
    policy_step = policy.action(time_step)
    return np.where(np.array_equal(policy_step.action, traj.action), 1.0, 0.0).item()  # type: ignore


def collect_policy_prob_fn(policy: core.PyPolicy, traj: core.Trajectory) -> float:
    """The behavior policy is assumed to be fixed over the evaluation window.
    We log probabilities when choosing actions, so we can just use that information.
    For a random policy on K arms, log_prob = log(1/K).
    We just have to return exp(log_prob).
    """
    del policy
    prob: float = np.exp(traj.policy_info["log_probability"])
    return prob


def constant_learning_rate(initial_lr: float, episode: int, step: int):
    del episode
    del step
    return initial_lr


@pytest.fixture(scope="function")
def policy(qtable: np.ndarray) -> core.PyPolicy:
    """
    Creates a greedy policy using a table.
    """
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
