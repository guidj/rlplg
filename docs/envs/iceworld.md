# IceWorld

The objective of the environment to move from a starting point to a goal.
There can be lakes, and if the agent falls into one the game ends.
The reward for every other action (including reaching the goal) is -1.

The environment incentivizes moving towards the goal.

The agent can go up, down, left and right.
If an action takes the agent outside the grid, they stay in the same position.

A key distinction between Iceworld and [Gridworld](gridworld.md) is that in the latter,
cliffs teleport the agent back to the start, whilst in the former, the game ends.

Based on [Gynamsium's FrozkenLake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) environment:


**States**

The total number of states is $W \times H$.

**Actions**

There are four actions - LEFT, RIGHT, UP, DOWN.

**Rewards**

If an agent falls into a lake, the reward is $-2 \times W \times H$.
Every other action has a reward of $-1$, including the first transition to the goal.

## Example Instatiation

```
from rlplg import envsuite

env_spec = envsuite.load(name="IceWorld", map_name="4x4")
```

or

```
from rlplg import envsuite

env_spec = envsuite.load(name="IceWorld", map="FFFG\nSFHH")
```