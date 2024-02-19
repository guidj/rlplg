# TariffFrozenLake-v1

This environment is a modified version of Gymansium's [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/).

**States**

Same as the base environment.

**Actions**

Same as teh base environment.

**Rewards**

The rewards from the base environmet are shifted so that every move has a penalty of $-1$, and the reaching the goal has a reward of $0$.

## Example Instatiation

```
from rlplg import envsuite

env_spec = envsuite.load(name="TariffFrozenLake-v1", is_slippery=False)
```
