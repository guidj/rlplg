"""
This module has constants for the StateRandomWalk environment.
"""


ENV_NAME = "StateRandomWalk"
GO_LEFT = 0
GO_RIGHT = 1
ACTIONS = [GO_LEFT, GO_RIGHT]
RIGHT_REWARD = 1
LEFT_REWARD = 0
STEP_REWARD = 0
OBS_KEY_POSITION = "position"
OBS_KEY_STEPS = "steps"
OBS_KEY_LEFT_END_REWARD = "left_end_reward"
OBS_KEY_RIGHT_END_REWARD = "right_end_reward"
OBS_KEY_STEP_REWARD = "step_reward"
