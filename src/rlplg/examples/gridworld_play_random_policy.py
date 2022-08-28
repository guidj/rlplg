"""
Example on using random policy to explore an environment: GridWorld.
No learning happens.
"""

import argparse
import dataclasses
import logging
import os
import os.path

from tf_agents.trajectories import time_step as ts

import rlplg
from rlplg import tracking
from rlplg.environments.gridworld import constants, env, rendering
from rlplg.learning.tabular import policies


@dataclasses.dataclass(frozen=True)
class Args:
    num_episodes: int
    render: False


def parse_args() -> Args:
    arg_parser = argparse.ArgumentParser(prog="GridWorld - Random Policy Play Example")
    arg_parser.add_argument("--num-episodes", type=int, default=2)
    arg_parser.add_argument("--render", action="store_true")
    arg_parser.set_defaults(render=False)
    args, _ = arg_parser.parse_known_args()
    return Args(**vars(args))


def main(args: Args):
    # init env and policy
    environment = env.GridWorld(
        size=(4, 6),
        cliffs=[
            (3, 3),
        ],
        exits=[(3, 5)],
        start=(3, 0),
    )
    policy = policies.PyRandomPolicy(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
        num_actions=len(constants.MOVES),
    )

    # reset env and state (get initial values)
    time_step = environment.reset()
    policy_state = policy.get_initial_state(None)
    last_move = None

    # stats tracking
    stats = tracking.EpisodeStats()
    episode = 0
    if args.render:
        sprites_dir = os.path.join(
            rlplg.SRC_DIR, os.pardir, "assets/env/gridworld/sprite"
        )
        renderer = rendering.GridWorldRenderer(sprites_dir)
        renderer.render(
            mode="human",
            observation=env.as_grid(time_step.observation),
            last_move=last_move,
            caption=str(stats),
        )

    while episode < args.num_episodes:
        policy_step = policy.action(time_step, policy_state)
        policy_state = policy_step.state
        time_step = environment.step(policy_step.action)

        stats.new_reward(time_step.reward)

        if time_step.step_type == ts.StepType.LAST:
            episode += 1
            success = (
                time_step.observation[constants.Strings.player]
                in time_step.observation[constants.Strings.exits]
            )
            time_step = environment.reset()
            stats.end_episode(success=success)
            logging.info(stats)

        last_move = policy_step.action

        if args.render:
            renderer.render(
                mode="human",
                observation=env.as_grid(time_step.observation),
                last_move=last_move,
                caption=str(stats),
            )

    environment.close()


if __name__ == "__main__":
    main(args=parse_args())
