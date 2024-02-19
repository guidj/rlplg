# rlplg

RL Playground is a library that implements a set of discrete state and action space environments,
as well as algorithms for policy evaluation and control.

## Supported Environments

| **Name** | **Desc** | **State Space** | **Action Space** | **Reward** |  |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Gridworld | A grid of size $n \times m$, with cliffs and one exit. The goal of the exit is to nagivate from their starting position to the exit. If the agent falls in a cliff, they go back to the start, so cliffs aren't part of the state space.<br><br>  <br><br>The grid is configurable with a string:<br><br>- `x` represents cliffs<br><br>- `g` represents the exit (goal)<br><br>- `s` represents the starting state<br><br>- `o` is for safe passage<br><br>  <br><br>Example 2 x 3 grid:<br><br>```<br><br>sox<br><br>xog<br><br>``` | The total number of states is $n \times m - \|cliffs\|$, where cliffs is a set. | There are four actions - left, right, up, down. | A reward of -1 is given for every action, except actions taken in the terminal state.<br><br>Additionally, if the agent falls into a cliff, there is an extra penalty of -100. |  |
| ABCSeq | The environment simulates ordering a sequence of items, provided as actions. <br><br>What's challenging in this environment is that there are roughly as many states as there are actions. <br><br>Given $n$ states, the agent can choose directly to go to any other state, starting from an initial state `0`. <br><br>However, the only action that results in a transitions is going to the state next to the agent's current position. For every other action, the agent remains where they are. | There are $n+1$ states. $n < 26$ | There are $n$ actions. | There is a penalty of -1 for every move.<br><br>Additionally, choosing the wrong action yields an extra reward proportional to the distance between the chosen item and the correct one. |  |
| StateRandomWalk | An environment where an agent is meant to go right, until the end.<br><br>The environment can terminate on the last left or right states.<br><br>By default, only the right direction yields a positive reward. | There are $n$ states, given as input. $n > 2$. | There are two actions - left, right. | A reward of 0 for ending on the left, and of 1 for ending on the right. |  |
| RedGreen | The environment simulates learning a working sequence of treatments that lead to a cure.<br><br>Given a finite set of options - red pill, green pill, wait - at each step, the agent must decide the best course of action.<br><br>The cure is provided as input to the environment (e.g. red, red, wait, red, wait, green).<br><br>The sequently can be arbitrarily large. | The state represents at which point of treatment the patient is in. If the sequence for the cure is size $n$, then there are $n+1$ states (incudling the starting state). | There are three actions - red, green, wait. | There is a penalty of -1 for every move.<br><br>Additionally, choosing the wrong action yields an extra reward of -1 |  |
| FrozenLake-v1 | [FrozenLake - Gymnasium](http://frozenlake-v1/) | \|\|Similar to FrozenLake, with the rewards shift so that every move has a penalty of -1, and the reaching the goal has a reward of 0. |  |  |  |
| TariffFrozenLake-v1 | This environment is a modified version of [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) | \|\|\| |  |  |  |
| Taxi-v3 | [Taxi - Gymnasium](https://gymnasium.farama.org/environments/toy_text/taxi/) | \|\|\| |  |  |  |
| CliffWalking-v0 | [Cliff Walking - Gymnasium](https://gymnasium.farama.org/environments/toy_text/cliff_walking/) | \|\|\| |  |  |  |
                        

## Usage

You can find examples of evaluation/control under [src/rlplg/examples/](src/rlplg/examples/).

## Setup Development

Instructions are provided in [docs/dev.md](docs/dev.md).
