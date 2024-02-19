# StateRandomWalk

An environment where an agent is meant to go right, until the end.
The agent can terminate on the left or right ends of the chain.
By default, only termination on the right end yields a positive reward.

**States**

There are $n$ states, given as input. $n > 2$.

**Actions**

There are two actions - LEFT, RIGHT.

**Rewards**

A reward of $0$ for ending on the left, and of $1$ for ending on the right.

## Example Instatiation

```
from rlplg import envsuite

env_spec = envsuite.load(name="StateRandomWalk", steps=7, left_end_reward=-1)
```
