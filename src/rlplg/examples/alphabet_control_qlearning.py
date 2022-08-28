"""
Example on using Q-learning to find an optimal policy.
"""

import argparse
import dataclasses
import logging

import numpy as np
from tf_agents.trajectories import time_step as ts

from rlplg import tracking
from rlplg.environments.alphabet import env
from rlplg.examples import qlearning


@dataclasses.dataclass(frozen=True)
class Args:
    num_letters: int
    num_episodes: int


def parse_args() -> Args:
    arg_parser = argparse.ArgumentParser(prog="ABCSeq - Q-learning Control Example")
    arg_parser.add_argument("--num-letters", type=int, default=2)
    arg_parser.add_argument("--num-episodes", type=int, default=1000)
    args, _ = arg_parser.parse_known_args()
    return Args(**vars(args))


def vis_learned_qtable(
    qtable: np.ndarray,
):
    return "\n{}".format(np.around(qtable, 3))


def initial_table(num_letters: int):
    qtable = np.zeros(shape=(num_letters + 1, num_letters))
    return qtable


def main(args: Args):
    # init env and policy
    environment = env.ABCSeq(length=args.num_letters)
    epsilon = 0.5
    alpha = 0.1
    gamma = 1.0
    # Policy Control with Q-learning
    learned_policy, learned_qtable = qlearning.control(
        environment,
        num_episodes=args.num_episodes,
        state_id_fn=env.get_state_id,
        initial_qtable=initial_table(num_letters=args.num_letters),
        epsilon=epsilon,
        gamma=gamma,
        alpha=alpha,
    )

    logging.info("Using trained policy to play")
    logging.info("\n%s", learned_qtable)
    logging.info(vis_learned_qtable(learned_qtable))
    # stats tracking
    stats = tracking.EpisodeStats()
    time_step = environment.reset()
    policy_state = learned_policy.get_initial_state(None)
    episode = 0
    steps = 0
    # play 3 times
    while episode < 3:
        policy_step = learned_policy.action(time_step, policy_state)
        policy_state = policy_step.state
        time_step = environment.step(policy_step.action)

        stats.new_reward(time_step.reward)

        if time_step.step_type == ts.StepType.LAST:
            episode += 1
            steps = 0
            time_step = environment.reset()
            stats.end_episode(success=True)
            logging.info(stats)

        steps += 1
        if steps > args.num_letters * 10:
            logging.warning("Stopping game play - policy doesn't solve the problem!")
            break

    environment.close()


if __name__ == "__main__":
    main(args=parse_args())
