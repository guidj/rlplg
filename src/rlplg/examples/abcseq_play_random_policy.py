"""
Example on using random policy to explore an environment: ABCSeq.
No learning happens.
"""

import argparse
import dataclasses
import logging
import math
import random


from rlplg.environments import abcseq
from rlplg.core import TimeStep


@dataclasses.dataclass(frozen=True)
class Args:
    """
    Example args.

    Args:
        num_letters: size of sequence length.
        num_episodes: episodes to play.
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
    environment = abcseq.ABCSeq(length=args.num_letters, distance_penalty=False)

    # play N times
    for episode in range(args.play_episodes):
        # reset env and state (get initial values)
        obs, _ = environment.reset()
        time_step: TimeStep = obs, math.nan, False, False, {}
        returns = 0.0
        while True:
            obs, _, terminated, _, _ = time_step
            action = random.randint(0, environment.action_space.n)
            next_time_step = environment.step(action)
            _, next_reward, _, _, _ = next_time_step
            returns += next_reward
            if terminated:
                logging.info(
                    "Episode %d terminated. Episodic return: %f", episode + 1, returns
                )
                break
            time_step = next_time_step

    environment.close()


if __name__ == "__main__":
    main(args=parse_args())
