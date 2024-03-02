# ABCSeq

The objective of the environment is to sort tokens.
Starting from the first token, the agent is suppose to choose the one that follows it,
and then the one that follows after.
The player reaches the end of the game when they choose the final token.

**States**

There are $N + 1$ state, the first being an beginning state, and the
others corresponding to each position in the sequence.

**Actions**

There are $N$ actions, each corresponding to a position in the sequence.

**Rewards**

There is a penalty of $-1$ for every move, with the exception of actions taken in the terminal state.
