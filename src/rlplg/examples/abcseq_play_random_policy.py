"""
Example on using random policy to explore an environment: ABCSeq.
No learning happens.
"""

import argparse
import dataclasses
import logging
import math

from rlplg import envsuite, tracking
from rlplg.core import TimeStep
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
    play_episodes: int


def parse_args() -> Args:
    """
    Parses std in arguments and returns an instanace of Args.
    """
    arg_parser = argparse.ArgumentParser(prog="Alphabet - Random Policy Play Example")
    arg_parser.add_argument("--num-letters", type=int, default=4)
    arg_parser.add_argument("--play-episodes", type=int, default=3)
    args, _ = arg_parser.parse_known_args()
    return Args(**vars(args))


def main(args: Args):
    """
    Entry point.
    """
    # init env and policy
    env_spec = envsuite.load(name="ABCSeq", length=args.num_letters)
    policy = policies.PyRandomPolicy(
        num_actions=env_spec.env_desc.num_actions,
        emit_log_probability=False,
    )

    # stats tracking
    stats = tracking.EpisodeStats()

    episode = 0
    logging.info(
        "Stats: %s",
        stats,
    )

    # play N times
    for episode in range(args.play_episodes):
        # reset env and state (get initial values)
        obs, _ = env_spec.environment.reset()
        policy_state = policy.get_initial_state()
        time_step: TimeStep = obs, math.nan, False, False, {}
        while True:
            obs, _, terminated, _, _ = time_step
            policy_step = policy.action(obs, policy_state)
            next_time_step = env_spec.environment.step(policy_step.action)
            _, next_reward, _, _, _ = next_time_step
            stats.new_reward(next_reward)
            if terminated:
                stats.end_episode(success=True)
                logging.info("Stats: %s, from episode %d", stats, episode + 1)
                break
            policy_state = policy_step.state
            time_step = next_time_step

    env_spec.environment.close()


if __name__ == "__main__":
    main(args=parse_args())
