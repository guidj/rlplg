# rlplg

RL Playground is a library that implements a set of discrete state and action space environments,
as well as algorithms for policy evaluation and control.

## Supported Environments

  - In library
    - [ABCSeq](docs/envs/abcseq.md)
    - [GridWorld](docs/envs/gridworld.md)
    - [StateRandomWalk](docs/envs/staterandomwalk.md)
    - [RedGreen](docs/envs/redgreen.md)
    - [TowerOfHanoir](docs/envs/towerofhanoi.md)
  - From [Gymnasium](https://gymnasium.farama.org/)
    - [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
    - [TariffFrozenLake-v1](docs/envs/tarifffrozenlake-v1.md)
    - [Taxi-v3](https://gymnasium.farama.org/environments/toy_text/taxi/)
    - [CliffWalking-v0](https://gymnasium.farama.org/environments/toy_text/cliff_walking/)

## Usage

You can find examples of evaluation/control under [src/rlplg/examples/](src/rlplg/examples/).

Some of them have 2D rendering - to run the GUI, install packages in [rendering-requirements.txt](rendering-requirements.txt).

## Setup Development

Instructions are provided in [docs/dev.md](docs/dev.md).
