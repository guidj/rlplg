"""
Example on dynamic programming to estimate the value
of a random policy.
"""

import logging

from rlplg import envsuite
from rlplg.learning.tabular import dynamicprog, policies


def main():
    """
    Entry point.
    """
    # init env and policy
    env_spec = envsuite.load("GridWorld", grid=["xsog"])
    policy = policies.PyRandomPolicy(
        num_actions=env_spec.mdp.env_desc.num_actions,
        emit_log_probability=False,
    )
    state_values = dynamicprog.iterative_policy_evaluation(
        mdp=env_spec.mdp, policy=policy
    )
    action_values = dynamicprog.action_values_from_state_values(
        env_spec.mdp, state_values
    )
    logging.info("State values: \n%s", state_values)
    logging.info("Action values: \n%s", action_values)
    env_spec.environment.close()


if __name__ == "__main__":
    main()
