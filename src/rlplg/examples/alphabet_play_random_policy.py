"""
Example on using random policy to explore an environment: ABCSeq.
No learning happens.
"""

import argparse
import dataclasses
import logging

from tf_agents.trajectories import time_step as ts

from rlplg import tracking
from rlplg.environments.alphabet import env
from rlplg.learning.tabular import policies


@dataclasses.dataclass(frozen=True)
class Args:
    num_letters: int
    num_episodes: int


def parse_args() -> Args:
    arg_parser = argparse.ArgumentParser(prog="Alphabet - Random Policy Play Example")
    arg_parser.add_argument("--num-letters", type=int, default=4)
    arg_parser.add_argument("--num-episodes", type=int, default=10)
    args, _ = arg_parser.parse_known_args()
    return Args(**vars(args))


def main(args: Args):
    # init env and policy
    environment = env.ABCSeq(length=args.num_letters)
    policy = policies.PyRandomPolicy(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
        num_actions=args.num_letters + 1,
        emit_log_probability=False,
    )

    # reset env and state (get initial values)
    time_step = environment.reset()
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
        time_step = environment.step(policy_step.action)

        stats.new_reward(time_step.reward)

        if time_step.step_type == ts.StepType.LAST:
            episode += 1
            time_step = environment.reset()
            stats.end_episode(success=True)
            logging.info(
                "Stats: %s",
                stats,
            )

    environment.close()


if __name__ == "__main__":
    main(args=parse_args())
