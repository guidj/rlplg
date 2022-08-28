import argparse
import dataclasses
import logging

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.policies import py_policy

from rlplg import metrics, npsci, runtime, tracking
from rlplg.environments.alphabet import env
from rlplg.learning import utils
from rlplg.learning.tabular import policies
from rlplg.learning.tabular.evaluation import offpolicy


@dataclasses.dataclass(frozen=True)
class Args:
    run_id: str
    policy_dir: str
    output_dir: str
    num_letters: int
    num_episodes: int
    control_epsilon: float
    control_alpha: float
    control_gamma: float


def parse_args() -> Args:
    arg_parser = argparse.ArgumentParser(
        prog="Alphabet - n-step SARSA Policy Evaluation Example"
    )
    arg_parser.add_argument("--run-id", type=str, default=runtime.run_id())
    arg_parser.add_argument("--policy-dir", type=str, required=True)
    arg_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )
    arg_parser.add_argument("--num-letters", type=int, default=7)
    arg_parser.add_argument("--num-episodes", type=int, default=1000)
    arg_parser.add_argument("--control-epsilon", type=float, default=0.0)
    arg_parser.add_argument("--control-alpha", type=float, default=0.1)
    arg_parser.add_argument("--control-gamma", type=float, default=0.99)

    args, _ = arg_parser.parse_known_args()
    return Args(**vars(args))


def policy_evaluation(
    run_id: str,
    num_states: int,
    num_actions: int,
    ref_qtable: np.ndarray,
    policy: policies.PyQGreedyPolicy,
    collect_policy: py_policy.PyPolicy,
    environment: py_environment.PyEnvironment,
    num_episodes: int,
    control_epsilon: float,
    control_alpha: float,
    control_gamma: float,
    output_dir: str,
):
    del control_epsilon
    results = offpolicy.nstep_sarsa_action_values(
        policy=policy,
        collect_policy=collect_policy,
        environment=environment,
        num_episodes=num_episodes,
        alpha=control_alpha,
        gamma=control_gamma,
        nstep=1,
        policy_probability_fn=utils.policy_prob_fn,
        collect_policy_probability_fn=utils.collect_policy_prob_fn,
        state_id_fn=env.get_state_id,
        action_id_fn=npsci.item,
        initial_qtable=utils.initial_table(
            num_states=num_states, num_actions=num_actions
        ),
    )

    with tracking.ExperimentLogger(
        output_dir,
        name="qpolicy/eval",
        params={
            "algorithm": "n-step SARSA/Off-Policy",
            "alpha": control_alpha,
            "gamma": control_gamma,
            "epsilon": 1.0,
        },
    ) as exp_logger:
        for episode, (steps, qtable) in enumerate(results):
            rmse = metrics.rmse(pred=qtable, actual=ref_qtable)
            logging.info(
                "Task %s, Episode %d: %d steps, %f RMSE", run_id, episode, steps, rmse
            )
            exp_logger.log(
                episode=episode,
                steps=steps,
                returns=0.0,
                metadata={"qtable": qtable.tolist(), "rmse": str(rmse)},
            )


def initial_table(num_states: int, num_actions: int) -> np.ndarray:
    """
    Value of terminal state should be zero.
    """
    q_table = np.zeros(shape=(num_states, num_actions))
    return q_table


def main(args: Args):
    # init env and agent
    collect_env = env.ABCSeq(args.num_letters)
    logging.info(
        "Task %s, Igoring control_args.epsilon [%f] - using 1.0 exploration probability for collection policy.",
        args.run_id,
        args.control_args.epsilon,
    )
    collect_policy = policies.PyRandomPolicy(
        time_step_spec=collect_env.time_step_spec(),
        action_spec=collect_env.action_spec(),
        num_actions=args.num_letters + 1,
        emit_log_probability=True,
    )
    saved_policy = tf.saved_model.load(args.policy_dir)
    ref_qtable = next(iter(saved_policy.model_variables)).numpy()
    policy = policies.PyQGreedyPolicy(
        time_step_spec=collect_env.time_step_spec(),
        action_spec=collect_env.action_spec(),
        state_id_fn=env.get_state_id,
        action_values=ref_qtable,
        emit_log_probability=True,
    )

    policy_evaluation(
        run_id=args.run_id,
        num_states=args.num_letters,
        num_actions=args.num_letters,
        ref_qtable=ref_qtable,
        policy=policy,
        collect_policy=collect_policy,
        environment=collect_env,
        num_episodes=args.num_episodes,
        control_epsilon=args.control_epsilon,
        control_alpha=args.control_alpha,
        control_gamma=args.control_gamma,
        output_dir=args.output_dir,
    )
    collect_env.close()


if __name__ == "__main__":
    main(args=parse_args())
