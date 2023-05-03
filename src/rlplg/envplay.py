"""
This module contains utilities for rollouts, and
trajectory data.
"""


from typing import Generator

from rlplg import core


def generate_episodes(
    environment: core.PyEnvironment,
    policy: core.PyPolicy,
    num_episodes: int,
) -> Generator[core.Trajectory, None, None]:
    """
    Generates `num_episodes` episodes using the environment
    and policy provided for rollout.

    Args:
        enviroment: environment to use.
        policy: for rollout.
        num_episodes: number of rollout episodes.

    Yields:
        Trajectory instances from episodic rollouts, one step at a time.
    """
    for _ in range(num_episodes):
        time_step = environment.reset()
        while True:
            policy_step = policy.action(time_step)
            next_time_step = environment.step(policy_step.action)
            yield core.Trajectory.from_transition(
                time_step, policy_step, next_time_step
            )
            if time_step.step_type == core.StepType.LAST:
                break
            time_step = next_time_step


def identity_replay(
    event: core.Trajectory,
) -> Generator[core.Trajectory, None, None]:
    """
    Yields:
        The given trajectory, as is.
    """
    yield event
