# GridWorld

A grid of size $W \times H$, with cliffs and one exit. The goal of the agent is to nagivate from their starting position to the exit. If the agent falls in a cliff, they go back to the start, so cliffs aren't part of the state space.

The grid is configurable with a string:
- `x` represents cliffs
- `g` represents the exit (goal)
- `s` represents the starting state
- `o` is for safe passage

Example 2 x 3 grid:

```
sox
xog
```

**States**

The total number of states is $W \times H - |\text{cliffs}|$, where cliffs is a set.

**Actions**

There are four actions - LEFT, RIGHT, UP, DOWN.

**Rewards**

A reward of $-1$ is given for every action, except actions taken in the terminal state.
Additionally, if the agent falls into a cliff, there is an extra penalty of -100.

## Example Instatiation

There is an [example](../../src/rlplg/examples/gridworld_dynamicprog.py) of doing dynamic programming with GridWorld.