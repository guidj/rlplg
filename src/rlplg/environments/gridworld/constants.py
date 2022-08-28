ENV_NAME = "GridWorld"
CLIFF_PENALTY = -100.0
MOVE_PENALTY = -1.0
TERMINAL_REWARD = 0.0
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
MOVES = ["L", "R", "U", "D"]
MOVE_SYMBOLS = ["←", "→", "↑", "↓"]


class Layers:
    player = 0
    cliff = 1
    exit = 2


class Strings:
    size = "size"
    player = "player"
    cliffs = "cliffs"
    exits = "exits"
    start = "start"


COLOR_SILVER = (192, 192, 192)

PATH_BG = "path-bg.png"
CLIFF_BG = "cliff-bg.png"
ACTOR = "actor.png"
