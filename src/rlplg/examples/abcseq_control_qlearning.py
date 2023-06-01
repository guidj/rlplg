"""
Example on using Q-learning to find an optimal policy.
"""

import argparse
import dataclasses
import logging
import math

from rlplg import envsuite, tracking
from rlplg.core import TimeStep
from rlplg.examples import factories, qlearning, rendering


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
    play_episodes: int


def parse_args() -> Args:
    """
    Parses std in arguments and returns an instanace of Args.
    """
    arg_parser = argparse.ArgumentParser(prog="ABCSeq - Q-learning Control Example")
    arg_parser.add_argument("--num-letters", type=int, default=2)
    arg_parser.add_argument("--num-episodes", type=int, default=1000)
    arg_parser.add_argument("--play-episodes", type=int, default=3)
    args, _ = arg_parser.parse_known_args()
    return Args(**vars(args))


def main(args: Args):
    """
    Entry point.
    """
    # init env and policy
    env_spec = envsuite.load(name="ABCSeq", length=args.num_letters)
    epsilon = 0.5
    alpha = 0.1
    gamma = 1.0
    # Policy Control with Q-learning
    learned_policy, learned_qtable = qlearning.control(
        env_spec.environment,
        num_episodes=args.num_episodes,
        state_id_fn=env_spec.discretizer.state,
        initial_qtable=factories.initialize_action_values(
            num_states=env_spec.env_desc.num_states,
            num_actions=env_spec.env_desc.num_actions,
        ),
        epsilon=epsilon,
        gamma=gamma,
        alpha=alpha,
    )

    logging.info("Using trained policy to play")
    logging.info("\n%s", learned_qtable)
    logging.info(rendering.vis_learned_array(learned_qtable))
    # stats tracking
    stats = tracking.EpisodeStats()
    # play N times
    for episode in range(args.play_episodes):
        obs, _ = env_spec.environment.reset()
        policy_state = learned_policy.get_initial_state()
        time_step: TimeStep = obs, math.nan, False, False, {}
        steps = 0
        while True:
            obs, _, terminated, _, _ = time_step
            policy_step = learned_policy.action(obs, policy_state)
            next_time_step = env_spec.environment.step(policy_step.action)
            _, next_reward, _, _, _ = next_time_step
            stats.new_reward(next_reward)

            if terminated:
                stats.end_episode(success=True)
                logging.info("Episode %d stats: %s", episode + 1, stats)
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
