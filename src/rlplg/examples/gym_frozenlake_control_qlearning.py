"""
Example on using OpenAI Gym's Toy Text CliffWalking:
https://www.gymlibrary.ml/environments/toy_text/cliff_walking/
"""

import argparse
import dataclasses
import logging
import math

from rlplg import envsuite, npsci
from rlplg.core import TimeStep
from rlplg.examples import factories, qlearning, rendering


@dataclasses.dataclass(frozen=True)
class Args:
    """
    Example args.

    Args:
        num_episodes: for Q-learning.
    """

    num_episodes: int
    play_episodes: int


def parse_args() -> Args:
    """
    Parses std in arguments and returns an instanace of Args.
    """
    arg_parser = argparse.ArgumentParser(prog="FrozenLake - Q-learning Control Example")
    arg_parser.add_argument("--num-episodes", type=int, default=1000)
    arg_parser.add_argument("--play-episodes", type=int, default=3)
    args, _ = arg_parser.parse_known_args()
    return Args(**vars(args))


def main(args: Args):
    """
    Entry point.
    """
    # Init env and policy
    env_spec = envsuite.load("FrozenLake-v1", render_mode="ansi")
    episode = 0
    # Policy Control with Q-learning
    learned_policy, qtable = qlearning.control(
        environment=env_spec.environment,
        num_episodes=args.num_episodes,
        state_id_fn=npsci.item,
        initial_qtable=factories.initialize_action_values(
            num_states=env_spec.env_desc.num_states,
            num_actions=env_spec.env_desc.num_actions,
        ),
        epsilon=0.5,
        gamma=1.0,
        alpha=0.1,
    )

    logging.info("Using trained policy to play")
    logging.info("\n%s", rendering.vis_learned_array(qtable))
    # play N times
    for episode in range(args.play_episodes):
        obs, _ = env_spec.environment.reset()
        policy_state = learned_policy.get_initial_state()
        time_step: TimeStep = obs, math.nan, False, False, {}
        steps = 0
        while True:
            obs, _, terminated, truncated, _ = time_step
            policy_step = learned_policy.action(obs, policy_state)
            next_time_step = env_spec.environment.step(policy_step.action)

            logging.info(env_spec.environment.render())
            if terminated or truncated:
                logging.info("Completed episode %d", episode + 1)
                break

            steps += 1
            if steps > env_spec.env_desc.num_states * 10:
                logging.warning(
                    "Stopping game play - policy doesn't solve the problem!"
                )
                break
            policy_state = policy_step.state
            time_step = next_time_step

    env_spec.environment.close()


if __name__ == "__main__":
    main(args=parse_args())
