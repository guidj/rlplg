"""
Example on using Q-learning to find an optimal policy.

Some grid configurations:
    2 x 2, ~ 50-100 episodes, e-greedy @ 50%
        x E
        S O

    3 x 3: ~50-100 episodes, e-greedy @ 50%
        O O E
        O x O
        S O O
"""

import argparse
import dataclasses
import logging
import os
import os.path
from typing import Mapping, Sequence, Tuple

import numpy as np
from tf_agents.trajectories import time_step as ts

import rlplg
from rlplg import tracking
from rlplg.environments.gridworld import constants, env, rendering, utils
from rlplg.examples import qlearning


@dataclasses.dataclass(frozen=True)
class Args:
    grid_file: str
    num_episodes: int
    render: False


def parse_args() -> Args:
    arg_parser = argparse.ArgumentParser(prog="GridWorld - Q-learning Control Example")
    arg_parser.add_argument(
        "--grid-file",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.path.pardir,
            os.path.pardir,
            os.path.pardir,
            "assets",
            "env",
            "gridworld",
            "levels",
            "gridworld_cliff_02.txt",
        ),
        type=str,
    )
    arg_parser.add_argument("--num-episodes", type=int, default=1000)
    arg_parser.add_argument("--render", action="store_true")
    arg_parser.set_defaults(render=False)
    args, _ = arg_parser.parse_known_args()
    return Args(**vars(args))


def vis_learned_policy(
    grid_size: Tuple[int, int],
    qtable: np.ndarray,
    cliffs: Sequence[Tuple[int, int]],
    exits: Sequence[Tuple[int, int]],
    states: Mapping[Tuple[int, int], int],
):
    vis_grid = []
    value_grid = []
    height, width = grid_size

    for pos_x in range(height):
        vis_row = []
        value_row = []
        for pos_y in range(width):

            if (pos_x, pos_y) in states:
                state = states[(pos_x, pos_y)]
                argmax_action = np.argmax(qtable[state]).item()
                vis_row.append(constants.MOVE_SYMBOLS[argmax_action])
                value_row.append(qtable[state, argmax_action].item())
            else:
                vis_row.append("")
                value_row.append(-np.inf)

        vis_grid.append(vis_row)
        value_grid.append(np.array(value_row))

    for pos_x, pos_y in cliffs:
        vis_grid[pos_x][pos_y] = "x"
    for pos_x, pos_y in exits:
        vis_grid[pos_x][pos_y] = "G"

    return (
        "\n".join(["\n"] + ["".join([f"[{v}]" for v in row]) for row in vis_grid]),
        np.array(value_grid),
    )


def initial_table(num_states: int):
    return np.zeros(shape=(num_states, len(constants.MOVES)))


def main(args: Args):
    # Init env and policy
    logging.info("Reading grid from %s", args.grid_file)
    size, cliffs, exits, starting_position = utils.parse_grid(args.grid_file)
    height, width = size
    environment = env.GridWorld(
        size=size,
        cliffs=cliffs,
        exits=exits,
        start=starting_position,
    )
    states = env.states_mapping(size=size, cliffs=cliffs)
    num_states = len(states)

    episode = 0
    # Policy Control with Q-learning
    learned_policy, qtable = qlearning.control(
        environment=environment,
        num_episodes=args.num_episodes,
        state_id_fn=env.create_state_id_fn(states),
        initial_qtable=initial_table(num_states),
        epsilon=0.5,
        gamma=1.0,
        alpha=0.1,
    )

    if args.render:
        sprites_dir = os.path.join(
            rlplg.SRC_DIR, os.pardir, "assets/env/gridworld/sprite"
        )
        renderer = rendering.GridWorldRenderer(sprites_dir)

    logging.info("Using trained policy to play")
    logging.info("\n%s", np.around(qtable, 2))
    vis_grid, value_grid = vis_learned_policy(
        (height, width), qtable, cliffs=cliffs, exits=exits, states=states
    )
    logging.info(vis_grid)
    logging.info("\n%s", value_grid)
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
                time_step.observation[constants.Strings.player]
                in time_step.observation[constants.Strings.exits]
            )
            time_step = environment.reset()
            stats.end_episode(success=success)
            logging.info(str(stats))

        if args.render:
            last_move = policy_step.action
            renderer.render(
                mode="human",
                observation=env.as_grid(time_step.observation),
                last_move=last_move,
                caption=str(stats),
            )

        steps += 1
        if steps > height * width:
            logging.warning("Stopping game play - policy doesn't solve the problem!")
            break

    environment.close()


if __name__ == "__main__":
    main(args=parse_args())
