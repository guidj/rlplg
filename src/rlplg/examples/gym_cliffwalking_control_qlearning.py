"""
Example on using OpenAI Gym's Toy Text CliffWalking:
https://www.gymlibrary.ml/environments/toy_text/cliff_walking/
"""

import argparse
import dataclasses
import logging

import numpy as np
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

from rlplg import npsci
from rlplg.examples import qlearning


@dataclasses.dataclass(frozen=True)
class Args:
    num_episodes: int


def parse_args() -> Args:
    arg_parser = argparse.ArgumentParser(
        prog="CliffWalking - Q-learning Control Example"
    )
    arg_parser.add_argument("--num-episodes", type=int, default=1000)
    args, _ = arg_parser.parse_known_args()
    return Args(**vars(args))


def initial_table(num_states: int):
    return np.zeros(shape=(num_states, 4))


def main():
    args = parse_args()

    # Init env and policy
    num_states = 48
    environment = suite_gym.load("CliffWalking-v0")
    episode = 0
    # Policy Control with Q-learning
    learned_policy, qtable = qlearning.control(
        environment=environment,
        num_episodes=args.num_episodes,
        state_id_fn=npsci.item,
        initial_qtable=initial_table(num_states),
        epsilon=0.5,
        gamma=1.0,
        alpha=0.1,
    )

    logging.info("Using trained policy to play")
    logging.info("\n%s", np.around(qtable, 2))
    time_step = environment.reset()
    policy_state = learned_policy.get_initial_state(None)
    episode = 0
    steps = 0
    # play 3 times
    while episode < 3:
        policy_step = learned_policy.action(time_step, policy_state)
        policy_state = policy_step.state
        time_step = environment.step(policy_step.action)

        if time_step.step_type == ts.StepType.LAST:
            episode += 1
            steps = 0

        logging.info(environment.render(mode="human"))

        steps += 1
        if steps > num_states * 10:
            logging.warning("Stopping game play - policy doesn't solve the problem!")
            break

    environment.close()


if __name__ == "__main__":
    main()
