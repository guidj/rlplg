"""
Example on using Q-learning to find an optimal policy.

Problem: RedGreeSeq
"""

import argparse
import dataclasses
import logging

import numpy as np
from tf_agents.trajectories import time_step as ts

from rlplg import tracking
from rlplg.environments.redgreen import constants, env
from rlplg.examples import qlearning


@dataclasses.dataclass(frozen=True)
class Args:
    num_episodes: int


def parse_args() -> Args:
    arg_parser = argparse.ArgumentParser(prog="RedGreen - Q-learning Control Example")
    arg_parser.add_argument("--num-episodes", type=int, default=1000)
    args, _ = arg_parser.parse_known_args()
    return Args(**vars(args))


def initial_table(num_states: int):
    return np.zeros(shape=(num_states, len(constants.ACTIONS)))


def main():
    args = parse_args()

    # Init env and policy
    cure = ["red", "green", "red", "wait"]
    num_states = len(cure) + 1
    terminal_state = len(cure)
    environment = env.RedGreenSeq(cure)
    episode = 0
    # Policy Control with Q-learning
    learned_policy, qtable = qlearning.control(
        environment=environment,
        num_episodes=args.num_episodes,
        state_id_fn=env.get_state_id,
        initial_qtable=initial_table(num_states),
        epsilon=0.5,
        gamma=1.0,
        alpha=0.1,
    )

    logging.info("Using trained policy to play")
    logging.info("\n%s", np.around(qtable, 2))
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
            success = (
                time_step.observation[constants.KEY_OBS_POSITION] == terminal_state
            )
            time_step = environment.reset()
            stats.end_episode(success=success)
            logging.info(str(stats))

        logging.info(environment.render())

        steps += 1
        if steps > num_states * 10:
            logging.warning("Stopping game play - policy doesn't solve the problem!")
            break

    environment.close()


if __name__ == "__main__":
    main()
