"""
Example on using random policy to explore an environment: ABCSeq.
No learning happens.
"""

import argparse
import dataclasses
import logging

from tf_agents.trajectories import time_step as ts

from rlplg import envsuite, tracking
from rlplg.learning.tabular import policies


@dataclasses.dataclass(frozen=True)
class Args:
    """
    Example args.

    Args:
        num_letters: size of sequence length.
        num_episodes: for Q-learning.
    """

    num_letters: int
    num_episodes: int


def parse_args() -> Args:
    """
    Parses std in arguments and returns an instanace of Args.
    """
    arg_parser = argparse.ArgumentParser(prog="Alphabet - Random Policy Play Example")
    arg_parser.add_argument("--num-letters", type=int, default=4)
    arg_parser.add_argument("--num-episodes", type=int, default=10)
    args, _ = arg_parser.parse_known_args()
    return Args(**vars(args))


def main(args: Args):
    """
    Entry point.
    """
    # init env and policy
    env_spec = envsuite.load(name="ABCSeq", length=args.num_letters)
    policy = policies.PyRandomPolicy(
        time_step_spec=env_spec.environment.time_step_spec(),
        action_spec=env_spec.environment.action_spec(),
        num_actions=env_spec.env_desc.num_actions,
        emit_log_probability=False,
    )

    # reset env and state (get initial values)
    time_step = env_spec.environment.reset()
    policy_state = policy.get_initial_state(None)

    # stats tracking
    stats = tracking.EpisodeStats()

    episode = 0
    logging.info(
        "Stats: %s",
        stats,
    )

    while episode < args.num_episodes:
        policy_step = policy.action(time_step, policy_state)
        policy_state = policy_step.state
        time_step = env_spec.environment.step(policy_step.action)

        stats.new_reward(time_step.reward)

        if time_step.step_type == ts.StepType.LAST:
            episode += 1
            time_step = env_spec.environment.reset()
            stats.end_episode(success=True)
            logging.info(
                "Stats: %s",
                stats,
            )

    env_spec.environment.close()


if __name__ == "__main__":
    main(args=parse_args())
