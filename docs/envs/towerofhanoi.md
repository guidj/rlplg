# Tower of Hanoi

There are three pegs, and a given number of disks - three or four.
The game starts with the disks stacked on the first peg, sorted by size with the smallest disk on top.
The game ends when all disks are stacked on the last peg, sorted by size with the smallest disk on top.

Rules:
1. Only one disk can be moved at a time
2. A legal move consitutes moving the top most disk
from its current peg onto another
3. No disk can be placed on top of a smaller disk

Since there are three pegs, there are only six possible
moves: moving the top most disk from peg 1 to either
2 or 3; from peg 2 to either 1 or 3, and so on.

The number of moves required to solve the puzzle is
is $2^{n} - 1$, where $n$ is the number of disks.

Code based on https://github.com/xadahiya/toh-gym.

**States**

The peg in which every disk if place.
Given $n$ disks, it's tuple of size $n$, where each value ranges from $(0, p-1)$, where $p$ is the number of pegs.

E.g. with 4 disks, (0, 0, 2, 1) indicates the first and second disk (by size) are on the first peg, with the smallest on top.
The third largest disk is on the third peg, and the largest disk is in the middle.

**Actions**

The movements consist of taking the top disk from one peg to another.
Thus, there are six possible moves: (0, 1), (0, 2), (1, 0), (1, 2), (2, 0) and (2, 1).

**Rewards**

There is a penalty of $-1$ for every move, with the exception of actions taken in the terminal state.

## Example Instatiation

```
from rlplg import envsuite

env_spec = envsuite.load(name="TowerOfHanoi")
```
