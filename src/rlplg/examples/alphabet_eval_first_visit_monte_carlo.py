"""
Example on using Q-learning to find an optimal policy.
"""

import argparse
import dataclasses
import logging

import numpy as np

from rlplg.environments.alphabet import env
from rlplg.learning.tabular import policies
from rlplg.learning.tabular.evaluation import onpolicy


@dataclasses.dataclass(frozen=True)
class Args:
    num_letters: int
    num_episodes: int


def parse_args() -> Args:
    arg_parser = argparse.ArgumentParser(
        prog="ABCSeq - SARSA Policy Evaluation Example"
    )
    arg_parser.add_argument("--num-letters", type=int, default=3)
    arg_parser.add_argument("--num-episodes", type=int, default=1000)
    args, _ = arg_parser.parse_known_args()
    return Args(**vars(args))


def vis_learned_qtable(
    qtable: np.ndarray,
):
    return "\n{}".format(np.around(qtable, 3))


def initial_table(num_letters: int):
    qtable = np.zeros(shape=(num_letters + 1,))
    return qtable


def main(args: Args):
    # init env and policy
    environment = env.ABCSeq(length=args.num_letters)
    gamma = 0.99
    policy = policies.PyRandomPolicy(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
        num_actions=args.num_letters + 1,
        emit_log_probability=False,
    )

    results = onpolicy.first_visit_monte_carlo_state_values(
        policy=policy,
        environment=environment,
        num_episodes=args.num_episodes,
        gamma=gamma,
        state_id_fn=env.get_state_id,
        initial_values=initial_table(num_letters=args.num_letters),
    )

    *_, (_, learned_values) = results

    logging.info("\n%s", learned_values)
    logging.info("\n%s", vis_learned_qtable(learned_values))
    environment.close()


if __name__ == "__main__":
    main(args=parse_args())
