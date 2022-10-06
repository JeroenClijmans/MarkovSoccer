"""
Configuration file.
"""


# ----- Adjust settings below ------------------------------------------------ #

WIDTH = 12  # Width of the grid overlay of the soccer field
LENGTH = 16  # Length of the grid overlay of the soccer field


# ----- Do not change: constants below are auto-determined ------------------- #

# numbers identifying particular states
INITIAL_STATE = WIDTH * LENGTH
BALL_REGAIN_STATE = INITIAL_STATE + 1
THROW_IN_STATE = INITIAL_STATE + 2
SHORT_FREE_KICK_STATE = INITIAL_STATE + 3
GOALKICK_STATE = INITIAL_STATE + 4
KICKOFF_STATE = INITIAL_STATE + 5
MOVE_NOT_SUCCESSFUL_STATE = INITIAL_STATE + 6
SHOT_NOT_SUCCESSFUL_STATE = INITIAL_STATE + 7
GOAL_STATE = INITIAL_STATE + 8

NB_FIELD_STATES = WIDTH * LENGTH

# state groups
START_STATES = (BALL_REGAIN_STATE, THROW_IN_STATE, SHORT_FREE_KICK_STATE, GOALKICK_STATE, KICKOFF_STATE)
FIELD_STATES = tuple(s for s in range(NB_FIELD_STATES))
NB_TRANSIENT_STATES = 1 + len(START_STATES) + len(FIELD_STATES)
TRANSIENT_STATES = tuple(s for s in range(NB_TRANSIENT_STATES))

ABSORBING_STATES = (MOVE_NOT_SUCCESSFUL_STATE, SHOT_NOT_SUCCESSFUL_STATE, GOAL_STATE)
NB_ABSORBING_STATES = len(ABSORBING_STATES)
SHOT_STATES = (SHOT_NOT_SUCCESSFUL_STATE, GOAL_STATE)

NB_STATES = NB_TRANSIENT_STATES + NB_ABSORBING_STATES
ALL_STATES = tuple(s for s in range(NB_STATES))

LEFT_STATES = tuple(s for s in range(int(NB_FIELD_STATES / 4)))
CENTER_STATES = tuple(s for s in range(int(NB_FIELD_STATES / 4), int(3 * NB_FIELD_STATES / 4)))
RIGHT_STATES = tuple(s for s in range(int(3 * NB_FIELD_STATES / 4), int(NB_FIELD_STATES)))


def _calc_own_half_states():
    result = set()
    for cell in FIELD_STATES:
        (cell_y, cell_x) = _get_cell_indices_from_flat(cell)
        if cell_x < LENGTH // 2:
            result.add(cell)
    return tuple(result)


def _calc_opponent_half_states():
    result = set()
    for cell in FIELD_STATES:
        (cell_y, cell_x) = _get_cell_indices_from_flat(cell)
        if cell_x >= LENGTH // 2:
            result.add(cell)
    return tuple(result)


def _get_cell_indices_from_flat(flat_cell_index: int) -> (int, int):
    x = flat_cell_index % LENGTH  # length index
    y = flat_cell_index // LENGTH  # width index
    return y, x


OWN_HALF_STATES = _calc_own_half_states()
OPPONENT_HALF_STATES = _calc_opponent_half_states()


def _calc_thirds_states():
    defensive_third = set()
    middle_third = set()
    offensive_third = set()
    for cell in FIELD_STATES:
        (cell_y, cell_x) = _get_cell_indices_from_flat(cell)
        if cell_x < LENGTH // 3:
            defensive_third.add(cell)
        elif cell_x >= LENGTH - LENGTH // 3:
            offensive_third.add(cell)
        else:
            middle_third.add(cell)
    return defensive_third, middle_third, offensive_third


DEFENSIVE_THIRD_STATES, MIDDLE_THIRD_STATES, OFFENSIVE_THIRD_STATES = _calc_thirds_states()
