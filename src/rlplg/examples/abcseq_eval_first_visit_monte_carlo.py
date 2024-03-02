"""
Example on using Q-learning to find an optimal policy.
"""

import argparse
import dataclasses
import logging

from rlplg import envsuite
from rlplg.examples import factories, rendering
from rlplg.learning.tabular import policies
from rlplg.learning.tabular.policyeval import onpolicy


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
    arg_parser = argparse.ArgumentParser(
        prog="ABCSeq - SARSA Policy Evaluation Example"
    )
    arg_parser.add_argument("--num-letters", type=int, default=3)
    arg_parser.add_argument("--num-episodes", type=int, default=1000)
    args, _ = arg_parser.parse_known_args()
    return Args(**vars(args))


def main(args: Args):
    """
    Entry point.
    """
    # init env and policy
    env_spec = envsuite.load(name="ABCSeq", length=args.num_letters)
    gamma = 0.99
    policy = policies.PyRandomPolicy(
        num_actions=env_spec.mdp.env_desc.num_actions,
        emit_log_probability=False,
    )

    results = onpolicy.first_visit_monte_carlo_state_values(
        policy=policy,
        environment=env_spec.environment,
        num_episodes=args.num_episodes,
        gamma=gamma,
        state_id_fn=env_spec.discretizer.state,
        initial_values=factories.initialize_state_values(
            num_states=env_spec.mdp.env_desc.num_states,
        ),
    )

    *_, (_, learned_values) = results

    logging.info("\n%s", learned_values)
    logging.info("\n%s", rendering.vis_learned_array(learned_values))
    env_spec.environment.close()


if __name__ == "__main__":
    main(args=parse_args())
