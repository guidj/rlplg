"""
This module contains utilities for rollouts, and
trajectory data.
"""


from typing import Generator

import numpy as np

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


def unroll_trajectory(
    traj: core.Trajectory,
) -> Generator[core.Trajectory, None, None]:
    """
    Unrolls a (batched) trajectory of N transitions into
    individual ones.

    Args:
        traj: A batched trajectory
    Yields:
        An instance of a trajectory for each transition
    """
    if traj.step_type.shape == ():
        # single value - just yield
        yield traj
    else:
        steps = traj.step_type.shape[0]
        for step in range(steps):
            if traj.policy_info not in ((), None):
                policy_info = {
                    "log_probability": traj.policy_info["log_probability"][step]
                }
            else:
                policy_info = traj.policy_info
            yield core.Trajectory(
                step_type=traj.step_type[step],
                observation=traj.observation[step],
                action=traj.action[step],
                policy_info=policy_info,
                next_step_type=traj.next_step_type[step],
                reward=traj.reward[step],
                discount=traj.discount[step],
            )


def slice_trajectory(traj: core.Trajectory, start: int, size: int) -> core.Trajectory:
    """
    Slices a batched trajectory.

    Args
        traj: a possibly batched trajectory.
        start: start index for slice.
        size: number of elements to slice from `start`.

    Returns:
        A slice of the possibly batch trajectory input.
    """
    if traj.policy_info not in ((), None):
        policy_info = {
            "log_probability": _slice_nd_array(
                traj.policy_info["log_probability"], begin=start, size=size
            )
        }
    else:
        policy_info = traj.policy_info
    return core.Trajectory(
        step_type=_slice_nd_array(traj.step_type, begin=start, size=size),
        observation=_slice_nd_array(traj.observation, begin=start, size=size),
        action=_slice_nd_array(traj.action, begin=start, size=size),
        policy_info=policy_info,
        next_step_type=_slice_nd_array(traj.next_step_type, begin=start, size=size),
        reward=_slice_nd_array(traj.reward, begin=start, size=size),
        discount=_slice_nd_array(traj.discount, begin=start, size=size),
    )


def fold_trajectories(*trajs: core.Trajectory) -> core.Trajectory:
    """
    Folds a sequence of trajectory instance into a single
    batched trajectory.

    Note: this class assume each input trajectory corresponds to a single step.

    Args:
        trajs: A sequence of trajectory instances.
    Returns:
        A batched trajectory instance, with the inputs in order.
    """
    sample = next(iter(trajs))
    step_type = []
    observation = []
    action = []
    log_prob = []
    next_step_type = []
    reward = []
    discount = []
    for traj in trajs:
        step_type.append(traj.step_type)
        observation.append(traj.observation)
        action.append(traj.action)
        next_step_type.append(traj.next_step_type)
        reward.append(traj.reward)
        discount.append(traj.discount)
        if traj.policy_info not in ((), None):
            log_prob.append(traj.policy_info["log_probability"])

    if len(log_prob) > 0:
        policy_info = {
            "log_probability": np.array(
                log_prob, dtype=sample.policy_info["log_probability"].dtype
            )
        }
    else:
        policy_info = traj.policy_info

    return core.Trajectory(
        step_type=np.array(step_type, sample.step_type.dtype),
        observation=np.array(observation, sample.observation.dtype),
        action=np.array(action, sample.action.dtype),
        policy_info=policy_info,
        next_step_type=np.array(next_step_type, sample.next_step_type.dtype),
        reward=np.array(reward, sample.reward.dtype),
        discount=np.array(discount, sample.discount.dtype),
    )


def _slice_nd_array(array: np.ndarray, begin: int, size: int):
    """
    Slices the outer dimensions of an nd tensor
    """
    if len(array.shape) > 1:
        return array[begin : begin + size, ::]

    return array[begin : begin + size]


def identity_replay(
    event: core.Trajectory,
) -> Generator[core.Trajectory, None, None]:
    """
    Yields:
        The given trajectory, as is.
    """
    yield event
