"""
Function to convert SPADL data to an extended version:
 - Action type id's, result id's, bodypart id's and team id's are resolved to the
   corresponding names.
 - The number of seconds at which an action happened is converted to a timestamp
   for easier interpretability.
 - Actions related to penalty shootouts are removed.
 - Own goals are located and converted to a separate action type.
 - Kick-offs are located and converted to a separate action type.
"""
import pandas as pd
import socceraction.spadl as spadl
from typing import Union


def convert_to_extended_spadl(actions: pd.DataFrame) -> pd.DataFrame:
    actions = _resolve_ids(actions)
    _convert_timeseconds_to_timestamps(actions)
    _remove_penalty_shootouts(actions)
    _convert_own_goals(actions)
    _convert_kick_offs(actions)
    actions = actions[['game_id', 'period_id', 'time_seconds', 'timestamp', 'team_name', 'player_id', 'start_x',
                       'end_x', 'start_y', 'end_y', 'type_name', 'result_name', 'bodypart_name', 'action_id']]
    return actions


def _resolve_ids(actions: pd.DataFrame) -> pd.DataFrame:
    actions = (
        actions.merge(spadl.actiontypes_df(), how="left")
        .merge(spadl.results_df(), how="left")
        .merge(spadl.bodyparts_df(), how="left")
        .reset_index(drop=True)
    )
    return actions


def _convert_timeseconds_to_timestamps(actions: pd.DataFrame) -> None:
    timestamps = actions['time_seconds'].round().apply(pd.to_timedelta, unit='s')
    actions.insert(3, 'timestamp', timestamps, allow_duplicates=True)


def _remove_penalty_shootouts(actions: pd.DataFrame) -> None:
    # locate penalty shootouts
    penalty_shootouts_index = actions[actions['period_id'] == 5].index
    # remove penalty shootouts
    actions.drop(penalty_shootouts_index, inplace=True)
    actions.reset_index()


def _convert_own_goals(actions: pd.DataFrame) -> None:
    # Locate own goals
    own_goal_indices = pd.Series([False] * len(actions), dtype=bool)
    for index, action in actions.iterrows():
        if action['result_name'] == 'owngoal':
            own_goal_indices.at[index] = True
    # convert own goals
    actions.loc[own_goal_indices, 'type_name'] = 'owngoal'
    actions.loc[own_goal_indices, 'result_name'] = 'fail'


# ----- Converting kick-offs ------------------------------------------------- #


def _convert_kick_offs(actions: pd.DataFrame) -> None:
    # locate kick-offs
    kick_offs_after_goal = _locate_first_opponent_actions_after_goals(actions)
    kick_offs_of_each_half = _locate_kickoffs_of_each_half(actions)
    kick_offs = kick_offs_of_each_half | kick_offs_after_goal
    # convert kick-offs to custom action type
    actions.loc[kick_offs, 'type_name'] = 'kick_off'


def _locate_kickoffs_of_each_half(actions: pd.DataFrame) -> pd.Series:
    result = pd.Series([False] * len(actions), dtype=bool)
    # locate starts of each half
    half_starts = pd.Series([False] * len(actions), dtype=bool)
    period_ids = actions['period_id'].unique()
    for period_id in period_ids:
        half_start = actions[actions['period_id'] == period_id].index[0]
        half_starts.at[half_start] = True
    # locate the kick-offs of each half
    for half_start_index in half_starts[half_starts].index:
        kick_off_index = _locate_first_ball_moving_action_after_index(actions, half_start_index - 1)
        potential_kick_off_action = actions.iloc[kick_off_index]
        if _action_can_be_kickoff(potential_kick_off_action):
            result.at[kick_off_index] = True
    return result


def _locate_first_ball_moving_action_after_index(actions: pd.DataFrame, action_index) -> Union[int, None]:
    nb_actions = len(actions)
    for index in range(action_index+1, nb_actions):
        action = actions.iloc[index]
        if _is_ball_moving_action(action):
            return index
    return None


def _locate_first_opponent_ball_moving_action_after_index(actions: pd.DataFrame, goal_index):
    nb_actions = len(actions)
    goal_action = actions.iloc[goal_index]
    is_owngoal = goal_action['type_name'] == 'owngoal'
    goal_action_team = goal_action['team_id']
    for index in range(goal_index+1, nb_actions):
        action = actions.iloc[index]
        if _is_ball_moving_action(action):
            if (is_owngoal and action['team_id'] == goal_action_team) or \
                    (not is_owngoal and action['team_id'] != goal_action_team):
                return index
            else:
                return None
    return None


def _locate_first_opponent_actions_after_goals(actions: pd.DataFrame) -> pd.Series:
    result = pd.Series([False] * len(actions), dtype=bool)
    # locate goals
    goal_indices = pd.Series([False] * len(actions), dtype=bool)
    for index, action in actions.iterrows():
        if _is_goal(action):
            goal_indices.at[index] = True
    # locate the kick-offs after each goal
    for goal_index in goal_indices[goal_indices].index:
        opponent_action_index = _locate_first_opponent_ball_moving_action_after_index(actions, goal_index)
        if opponent_action_index is not None:
            potential_kick_off_action = actions.iloc[opponent_action_index]
            if _action_can_be_kickoff(potential_kick_off_action):
                result.at[opponent_action_index] = True
    return result


def _action_can_be_kickoff(action) -> bool:
    # sanity check whether an action takes place in the zone where kick-offs are
    # taken
    return 50.0 <= action['start_x'] <= 54.0 and \
           32.0 <= action['start_y'] <= 36.0


def _is_ball_moving_action(action) -> bool:
    return action['type_name'] in (
        'pass',
        'cross',
        'throw_in',
        'freekick_crossed',
        'freekick_short',
        'corner_crossed',
        'corner_short',
        'shot',
        'shot_penalty',
        'shot_freekick',
        'dribble',
        'goalkick',
        'kick_off'
    )


def _is_goal(action):
    return action['type_name'] == 'shot' and action['result_name'] == 'success' or \
           action['type_name'] == 'shot_penalty' and action['result_name'] == 'success' or \
           action['type_name'] == 'shot_freekick' and action['result_name'] == 'success' or \
           action['type_name'] == 'owngoal'
