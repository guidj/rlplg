# RedGreen

The environment simulates learning a working sequence of treatments that lead to a cure.
Given a finite set of options - red pill, green pill, wait - at each step, the agent must decide the best course of action.
The cure is provided as input to the environment (e.g. red, red, wait, red, wait, green).
The sequence can be arbitrarily large.

**States**

The state represents at which point of treatment administration sequence the patient is in.
If the sequence for the cure is size $n$, then there are $n+1$ states (incudling the starting state).

**Actions**

There are three actions - RED, GREEN, WAIT.

**Rewards**

There is a penalty of $-1$ for every move, with the exception of actions taken in the terminal state.

## Example Instatiation

There is an [example](../../src/rlplg/examples/abcseq_play_random_policy.py) of solving the problem with uniform random policy.
