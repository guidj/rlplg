"""
Example on using Q-learning to find an optimal policy.
"""

import argparse
import dataclasses
import logging
from typing import Any, Mapping

from tf_agents.trajectories import time_step as ts

from rlplg import envsuite, tracking
from rlplg.examples import factories, qlearning, rendering


@dataclasses.dataclass(frozen=True)
class Args:
    """
    Problem args.

    Args:
        num_episodes: for Q-learning.
    """

    env_name: str
    env_args: Mapping[str, Any]
    num_episodes: int
    epsilon: float
    alpha: float
    gamma: float
    play_episodes: int


def parse_args() -> Args:
    """
    Parses std in arguments and returns an instanace of Args.
    """
    arg_parser = argparse.ArgumentParser(
        prog="""Q-learning Control Example. Provide env name and args. Usage:
        python -m control_qlearning --env-name [name] --num-episodes [value] --env-specific-arg [value]
        """
    )
    arg_parser.add_argument("--env-name", type=str, required=True)
    arg_parser.add_argument("--num-episodes", type=int, default=1000)
    arg_parser.add_argument("--epsilon", type=float, default=0.5)
    arg_parser.add_argument("--alpha", type=float, default=0.1)
    arg_parser.add_argument("--gamma", type=float, default=1.0)
    arg_parser.add_argument("--play-episodes", type=float, default=3)
    args = vars(arg_parser.parse_args())
    fields = dataclasses.fields(Args)
    known_args = {
        field.name: args.pop(field.name) for field in fields if field.name in args
    }
    return Args(**known_args, env_args=args)


def main(args: Args):
    """
    Entry point.
    """
    # init env and policy
    env_spec = envsuite.load(name=args.env_name, **args.env_args)
    # Policy Control with Q-learning
    learned_policy, learned_qtable = qlearning.control(
        env_spec.environment,
        num_episodes=args.num_episodes,
        state_id_fn=env_spec.discretizer.state,
        initial_qtable=factories.initialize_action_values(
            num_states=env_spec.env_desc.num_states,
            num_actions=env_spec.env_desc.num_actions,
        ),
        epsilon=args.epsilon,
        gamma=args.gamma,
        alpha=args.alpha,
    )

    logging.info("Using trained policy to play")
    logging.info("\n%s", learned_qtable)
    logging.info(rendering.vis_learned_array(learned_qtable))
    # stats tracking
    stats = tracking.EpisodeStats()
    # play N times
    for episode in range(args.play_episodes):
        time_step = env_spec.environment.reset()
        policy_state = learned_policy.get_initial_state(None)
        steps = 0
        while True:
            policy_step = learned_policy.action(time_step, policy_state)
            policy_state = policy_step.state
            time_step = env_spec.environment.step(policy_step.action)

            stats.new_reward(time_step.reward)

            if time_step.step_type == ts.StepType.LAST:
                stats.end_episode(success=True)
                logging.info("Episode %d stats: %s", episode + 1, stats)
                break

            steps += 1
            if steps > env_spec.env_desc.num_states * 10:
                logging.warning(
                    "Stopping game play - policy doesn't solve the problem!"
                )
                break

    env_spec.environment.close()


if __name__ == "__main__":
    main(args=parse_args())
