"""
Module for team models.
 - Constructing a team model from event stream data.
 - Reading/writing a team model from/to a file.
 - Basic operations on the team model.
"""
from typing import Tuple

import pandas as pd
import numpy as np
import socceraction.spadl.config as spadlconfig
from markovsoccer.config import *
from markovsoccer.dtmc import DTMC
from markovsoccer.prism import PrismModel


class TeamModel(DTMC):

    # ----- Public ----------------------------------------------------------- #

    def __init__(self, transition_matrix: np.ndarray, team_name: str):
        super().__init__(transition_matrix, start_absorbing_states=NB_TRANSIENT_STATES)
        self.team_name = team_name

    @staticmethod
    def build_from(actions: pd.DataFrame, team_name: str):
        transition_matrix = TeamModel._calc_transition_matrix(actions, team_name)
        model = TeamModel(transition_matrix, team_name)
        return model

    def convert_to_prism_file(self, path: str):
        prism_model = PrismModel.construct_from(self)
        prism_model.write_to_file(path)

    # ----- Private: model construction -------------------------------------- #

    @staticmethod
    def _calc_transition_matrix(actions: pd.DataFrame, team_name: str) -> np.ndarray:
        init_to_start = _init_to_start_states_matrix(actions, team_name)
        start_states_to_field = _start_states_to_field_matrix(actions, team_name)
        field_to_move_or_shot = _field_to_move_or_shot_matrix(actions, team_name)
        move_to_success_or_fail, move_success_to_field = _move_transition_matrices(actions, team_name)
        shot_to_success_or_fail = _scoring_prob(actions, team_name)
        transition_matrix = np.zeros((NB_STATES, NB_STATES))

        # add initial state to start states transitions
        transition_matrix[INITIAL_STATE, BALL_REGAIN_STATE] = init_to_start.at['ball_regain']
        transition_matrix[INITIAL_STATE, THROW_IN_STATE] = init_to_start.at['throw_in']
        transition_matrix[INITIAL_STATE, SHORT_FREE_KICK_STATE] = init_to_start.at['freekick_short']
        transition_matrix[INITIAL_STATE, GOALKICK_STATE] = init_to_start.at['goalkick']
        transition_matrix[INITIAL_STATE, KICKOFF_STATE] = init_to_start.at['kick_off']

        # add start states to field states transitions
        transition_matrix[BALL_REGAIN_STATE, :NB_FIELD_STATES] = start_states_to_field.loc[:, 'ball_regain']
        transition_matrix[THROW_IN_STATE, :NB_FIELD_STATES] = start_states_to_field.loc[:, 'throw_in']
        transition_matrix[SHORT_FREE_KICK_STATE, :NB_FIELD_STATES] = start_states_to_field.loc[:, 'freekick_short']
        transition_matrix[GOALKICK_STATE, :NB_FIELD_STATES] = start_states_to_field.loc[:, 'goalkick']
        transition_matrix[KICKOFF_STATE, :NB_FIELD_STATES] = start_states_to_field.loc[:, 'kick_off']

        # add field states to other field states & absorbing states transitions
        for field_state in range(NB_FIELD_STATES):
            move_probability = field_to_move_or_shot.at[field_state, 'move']
            shot_probability = field_to_move_or_shot.at[field_state, 'shot']
            move_success_probability = move_probability * move_to_success_or_fail.at[field_state, 'success']
            move_fail_probability = move_probability * move_to_success_or_fail.at[field_state, 'fail']
            shot_success_probability = shot_probability * shot_to_success_or_fail.at[field_state, 'goal']
            shot_fail_probability = shot_probability * shot_to_success_or_fail.at[field_state, 'shot_not_successful']
            transition_matrix[field_state, MOVE_NOT_SUCCESSFUL_STATE] = move_fail_probability
            transition_matrix[field_state, GOAL_STATE] = shot_success_probability
            transition_matrix[field_state, SHOT_NOT_SUCCESSFUL_STATE] = shot_fail_probability
            for destination in range(NB_FIELD_STATES):
                transition_matrix[field_state, destination] = move_success_probability * move_success_to_field[
                    field_state, destination]

        return transition_matrix


# ----- Private: probability calculations ------------------------------------ #


def _count(x: pd.Series, y: pd.Series) -> np.ndarray:
    """ Count the number of actions occurring in each cell of the grid.

    :param x: The x-coordinates of the actions.
    :param y: The y-coordinates of the actions.
    :return: A matrix, denoting the amount of actions occurring in each cell. The top-left corner is the origin.
    """
    x = x[~np.isnan(x) & ~np.isnan(y)]
    y = y[~np.isnan(x) & ~np.isnan(y)]
    flat_indexes = _get_flat_indexes(x, y)
    vc = flat_indexes.value_counts(sort=False)
    vector = np.zeros(WIDTH * LENGTH)
    vector[vc.index] = vc
    return vector.reshape((WIDTH, LENGTH))


def _get_flat_indexes(x: pd.Series, y: pd.Series) -> pd.Series:
    xi, yj = _get_cell_indexes(x, y)
    return LENGTH * (WIDTH - 1 - yj) + xi


def _get_cell_indexes(x: pd.Series, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
    xmin, ymin = 0, 0
    xi = (x - xmin) / spadlconfig.field_length * LENGTH
    yj = (y - ymin) / spadlconfig.field_width * WIDTH
    xi = xi.astype(int).clip(0, LENGTH - 1)
    yj = yj.astype(int).clip(0, WIDTH - 1)
    return xi, yj


def _safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def _init_to_start_states_matrix(actions: pd.DataFrame, team_name: str) -> pd.Series:
    # action filters
    ball_regain_filter = actions['ball_recovery']
    throw_in_filter = actions['type_name'] == 'throw_in'
    short_free_kick_filter = actions['type_name'] == 'freekick_short'
    goalkick_filter = actions['type_name'] == 'goalkick'
    kick_off_filter = actions['type_name'] == 'kick_off'
    team_filter = actions['team_name'] == team_name
    successful_action_filter = actions['result_name'] == 'success'
    possession_seq_filter = actions['modelled_possession_sequence']

    # modelled actions
    modelled_ball_regains = actions[ball_regain_filter & team_filter & possession_seq_filter]
    modelled_throw_ins = actions[throw_in_filter & team_filter & successful_action_filter & possession_seq_filter]
    modelled_short_free_kicks = actions[
        short_free_kick_filter & team_filter & successful_action_filter & possession_seq_filter]
    modelled_goalkicks = actions[goalkick_filter & team_filter & successful_action_filter & possession_seq_filter]
    modelled_kick_offs = actions[kick_off_filter & team_filter & successful_action_filter & possession_seq_filter]

    # calculate frequencies
    nb_modelled_start_actions = len(modelled_ball_regains) + len(modelled_throw_ins) + len(modelled_short_free_kicks) \
                                + len(modelled_goalkicks) + len(modelled_kick_offs)
    d = {
        'ball_regain': len(modelled_ball_regains) / nb_modelled_start_actions,
        'throw_in': len(modelled_throw_ins) / nb_modelled_start_actions,
        'freekick_short': len(modelled_short_free_kicks) / nb_modelled_start_actions,
        'goalkick': len(modelled_goalkicks) / nb_modelled_start_actions,
        'kick_off': len(modelled_kick_offs) / nb_modelled_start_actions
    }

    result = pd.Series(data=d)
    return result


def _start_states_to_field_matrix(actions: pd.DataFrame, team_name: str) -> pd.DataFrame:
    ball_regain_to_field = _ball_regain_to_field_matrix(actions, team_name).flatten()
    throw_in_to_field = _throw_in_to_field_matrix(actions, team_name).flatten()
    short_free_kick_to_field = _short_free_kick_to_field_matrix(actions, team_name).flatten()
    goalkick_to_field = _goalkick_to_field_matrix(actions, team_name).flatten()
    kick_off_to_field = _kick_off_to_field_matrix(actions, team_name).flatten()
    d = {
        'ball_regain': ball_regain_to_field,
        'throw_in': throw_in_to_field,
        'freekick_short': short_free_kick_to_field,
        'goalkick': goalkick_to_field,
        'kick_off': kick_off_to_field
    }
    result = pd.DataFrame(data=d)
    return result


def _ball_regain_to_field_matrix(actions: pd.DataFrame, team_name: str) -> np.ndarray:
    ball_regain_filter = actions['ball_recovery']
    team_filter = actions['team_name'] == team_name
    possession_seq_filter = actions['modelled_possession_sequence']
    modelled_ball_regains = actions[ball_regain_filter & team_filter & possession_seq_filter]
    counts = _count(modelled_ball_regains.start_x, modelled_ball_regains.start_y)
    frequencies = counts / len(modelled_ball_regains)
    return frequencies


def _throw_in_to_field_matrix(actions: pd.DataFrame, team_name: str) -> np.ndarray:
    throw_in_filter = actions['type_name'] == 'throw_in'
    team_filter = actions['team_name'] == team_name
    successful_action_filter = actions['result_name'] == 'success'
    possession_seq_filter = actions['modelled_possession_sequence']
    modelled_throw_ins = actions[throw_in_filter & team_filter & successful_action_filter & possession_seq_filter]
    counts = _count(modelled_throw_ins.end_x, modelled_throw_ins.end_y)
    frequencies = counts / len(modelled_throw_ins)
    return frequencies


def _short_free_kick_to_field_matrix(actions: pd.DataFrame, team_name: str) -> np.ndarray:
    short_free_kick_filter = actions['type_name'] == 'freekick_short'
    team_filter = actions['team_name'] == team_name
    successful_action_filter = actions['result_name'] == 'success'
    possession_seq_filter = actions['modelled_possession_sequence']
    modelled_short_free_kicks = actions[
        short_free_kick_filter & team_filter & successful_action_filter & possession_seq_filter]
    counts = _count(modelled_short_free_kicks.end_x, modelled_short_free_kicks.end_y)
    frequencies = counts / len(modelled_short_free_kicks)
    return frequencies


def _goalkick_to_field_matrix(actions: pd.DataFrame, team_name: str) -> np.ndarray:
    goalkick_filter = actions['type_name'] == 'goalkick'
    team_filter = actions['team_name'] == team_name
    successful_action_filter = actions['result_name'] == 'success'
    possession_seq_filter = actions['modelled_possession_sequence']
    modelled_goal_kicks = actions[goalkick_filter & team_filter & successful_action_filter & possession_seq_filter]
    counts = _count(modelled_goal_kicks.end_x, modelled_goal_kicks.end_y)
    frequencies = counts / len(modelled_goal_kicks)
    return frequencies


def _kick_off_to_field_matrix(actions: pd.DataFrame, team_name: str) -> np.ndarray:
    kick_off_filter = actions['type_name'] == 'kick_off'
    team_filter = actions['team_name'] == team_name
    successful_action_filter = actions['result_name'] == 'success'
    possession_seq_filter = actions['modelled_possession_sequence']
    modelled_kick_offs = actions[kick_off_filter & team_filter & successful_action_filter & possession_seq_filter]
    counts = _count(modelled_kick_offs.end_x, modelled_kick_offs.end_y)
    frequencies = counts / len(modelled_kick_offs)
    return frequencies


def _field_to_move_or_shot_matrix(actions: pd.DataFrame, team_name: str) -> pd.DataFrame:
    # only consider actions of the team under consideration
    team_filter = actions['team_name'] == team_name
    actions = actions[team_filter]

    # filter move and shot actions
    move_filter = actions['type_name'].isin(('pass', 'cross', 'dribble'))
    shot_filter = actions['type_name'] == 'shot'
    modelled_possession_seq_filter = actions['modelled_possession_sequence']
    move_actions = actions[move_filter & modelled_possession_seq_filter]
    shot_actions = actions[shot_filter & modelled_possession_seq_filter]

    # calculate frequencies
    move_matrix = _count(move_actions.start_x, move_actions.start_y)
    shot_matrix = _count(shot_actions.start_x, shot_actions.start_y)
    total_matrix = move_matrix + shot_matrix
    move_freqs = _safe_divide(move_matrix, total_matrix).flatten()
    shot_freqs = _safe_divide(shot_matrix, total_matrix).flatten()
    d = {
        'move': move_freqs,
        'shot': shot_freqs
    }
    result = pd.DataFrame(data=d)
    return result


def _move_transition_matrices(actions: pd.DataFrame, team_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
    # get move actions of the team under consideration
    team_filter = actions['team_name'] == team_name
    move_filter = actions['type_name'].isin(('pass', 'cross', 'dribble'))
    modelled_possession_seq_filter = actions['modelled_possession_sequence']
    move_actions = actions[team_filter & move_filter & modelled_possession_seq_filter]

    # calculate frequencies
    X = pd.DataFrame()
    X["start_cell"] = _get_flat_indexes(move_actions.start_x, move_actions.start_y)
    X["end_cell"] = _get_flat_indexes(move_actions.end_x, move_actions.end_y)
    X["result_name"] = move_actions.result_name
    vc = X.start_cell.value_counts(sort=False)
    start_counts = np.zeros(NB_FIELD_STATES)
    start_counts[vc.index] = vc
    move_transition_matrix = np.zeros((NB_FIELD_STATES, NB_FIELD_STATES))
    move_success_matrix = np.zeros(NB_FIELD_STATES)
    move_not_successful_matrix = np.zeros(NB_FIELD_STATES)
    for i in FIELD_STATES:
        vc2 = X[
            ((X.start_cell == i) & (X.result_name == "success"))
        ].end_cell.value_counts(sort=False)
        move_transition_matrix[i, vc2.index] = vc2 / start_counts[i]
        move_successful_probability = np.sum(move_transition_matrix[i, :])
        move_success_matrix[i] = move_successful_probability
        move_not_successful_probability = 1 - move_successful_probability
        move_not_successful_matrix[i] = move_not_successful_probability
    sums = move_transition_matrix.sum(axis=1, keepdims=1)
    move_transition_matrix = np.divide(move_transition_matrix, sums, out=np.zeros_like(move_transition_matrix),
                                       where=sums != 0)

    d = {
        'success': move_success_matrix,
        'fail': move_not_successful_matrix
    }
    move_success_matrix = pd.DataFrame(data=d)
    return move_success_matrix, move_transition_matrix


def _scoring_prob(actions: pd.DataFrame, team_name: str) -> pd.DataFrame:
    # get shot and goal actions of the team under consideration
    team_filter = actions['team_name'] == team_name
    shot_filter = actions['type_name'] == 'shot'
    modelled_possession_seq_filter = actions['modelled_possession_sequence']
    shot_actions = actions[team_filter & shot_filter & modelled_possession_seq_filter]
    goal_actions = shot_actions[shot_actions.result_name == "success"]

    # calculate frequencies
    shot_matrix = _count(shot_actions.start_x, shot_actions.start_y)
    goal_matrix = _count(goal_actions.start_x, goal_actions.start_y)
    goal_freq = _safe_divide(goal_matrix, shot_matrix).flatten()

    d = {
        'shot_not_successful': 1 - goal_freq,  # todo: check if matrices are right
        'goal': goal_freq
    }
    result = pd.DataFrame(data=d)
    return result
