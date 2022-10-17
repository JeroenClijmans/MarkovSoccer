"""
Function to convert SPADL data to an extended version:
 - Action type id's, result id's, bodypart id's and team id's are resolved to the
   corresponding names.
 - The number of seconds at which an action happened is converted to a timestamp
   for easier interpretability.
 - Actions related to penalty shootouts are removed.
 - Own goals are located and converted to a separate action type.
 - Kick-offs are located and converted to a separate action type.
 - Attribute is added related to whether an action is labeled a ball recovery.
 - Attribute is added related to whether an action is part of a modelled possession sequence.
"""
import pandas as pd
import socceraction.spadl as spadl
from typing import Union


# ----- Public --------------------------------------------------------------- #


def convert_to_extended_spadl(actions: pd.DataFrame) -> pd.DataFrame:
    actions = _resolve_ids(actions)
    _convert_timeseconds_to_timestamps(actions)
    _remove_penalty_shootouts(actions)
    _convert_own_goals(actions)
    _convert_kick_offs(actions)
    possession_starts = _locate_possession_starts(actions)
    modelled_possession_starts = _locate_modelled_possession_starts(actions, possession_starts)
    actions['modelled_possession_sequence'] = _locate_possession_seq_actions(actions, modelled_possession_starts)
    actions['ball_recovery'] = _locate_ball_regains(actions, modelled_possession_starts)
    actions = actions[['game_id', 'period_id', 'timestamp', 'team_name', 'player_id', 'start_x',
                       'end_x', 'start_y', 'end_y', 'type_name', 'result_name', 'bodypart_name',
                       'ball_recovery', 'modelled_possession_sequence']]
    return actions


# ----- Private -------------------------------------------------------------- #


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


def _locate_ball_regains(actions: pd.DataFrame, possession_start_indices: pd.Series) -> pd.Series:
    result = pd.Series([False] * len(actions), dtype=bool)
    for index in possession_start_indices[possession_start_indices].index:
        action = actions.iloc[index]
        if action['type_name'] in _BALL_REGAIN_ACTION_TYPES:
            result.at[index] = True
    return result


# ----- Private: Converting kick-offs ---------------------------------------- #


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


# ----- Private: locating ball possession sequences -------------------------- #

# POSSESSION SEQUENCE
# A possession sequence is defined as:
# - an uninterrupted sequence of 3 or more ball moving actions
# - an uninterrupted sequence starting with a set piece / throw-in / goal kick

# BALL REGAIN
# A valid ball regain is defined as a first action of a possession sequence,
# excluding set pieces / throw-ins / goal kicks

# ACTION TABLE
# The following table lists for each action
# - whether it is considered an open play action
# - whether the action is a deliberate ball moving action (DBM)
# - whether the action is considered successful
# - whether an opponent executing this action can interrupt the current possession sequence
#
# |-------------------------------------------------------------------------------------------------------------------|
# | ACTION             | RESULT      | OPEN PLAY         | DBM                | SUCCESSFUL         | DOES INTERRUPT   |
# |-------------------------------------------------------------------------------------------------------------------|
# | Pass               | Success     | yes               | yes                | yes                | yes              |
# |                    | Fail        | yes               | yes                | no                 | yes              |
# |                    | Offside     | yes               | yes                | no                 | yes              |
# |-------------------------------------------------------------------------------------------------------------------|
# | Cross              | Success     | yes               | yes                | yes                | yes              |
# |                    | Fail        | yes               | yes                | no                 | yes              |
# |                    | Offside     | yes               | yes                | no                 | yes              |
# |-------------------------------------------------------------------------------------------------------------------|
# | Throw-in           | Success     | no                | yes                | yes                | yes              |
# |                    | Fail        | no                | yes                | no                 | yes              |
# |-------------------------------------------------------------------------------------------------------------------|
# | Crossed free-kick  | Success     | no                | yes                | yes                | yes              |
# |                    | Fail        | no                | yes                | no                 | yes              |
# |                    | Offside     | no                | yes                | no                 | yes              |
# |-------------------------------------------------------------------------------------------------------------------|
# | Short free-kick    | Success     | no                | yes                | yes                | yes              |
# |                    | Fail        | no                | yes                | no                 | yes              |
# |                    | Offside     | no                | yes                | no                 | yes              |
# |-------------------------------------------------------------------------------------------------------------------|
# | Crossed corner     | Success     | no                | yes                | yes                | yes              |
# |                    | Fail        | no                | yes                | no                 | yes              |
# |-------------------------------------------------------------------------------------------------------------------|
# | Short corner       | Success     | no                | yes                | yes                | yes              |
# |                    | Fail        | no                | yes                | no                 | yes              |
# |-------------------------------------------------------------------------------------------------------------------|
# | Take on            | Success     | yes               | no                 | yes                | yes              |
# |                    | Fail        | yes               | no                 | no                 |  -               |
# |-------------------------------------------------------------------------------------------------------------------|
# | Foul               | (Any)       | no                | no                 | no                 | yes              |
# |-------------------------------------------------------------------------------------------------------------------|
# | Tackle             | Success     | yes               | no                 | yes                | yes              |
# |                    | Fail        | yes               | no                 | no                 | no               |
# |-------------------------------------------------------------------------------------------------------------------|
# | Interception       | Success     | yes               | no                 | yes                | yes              |
# |                    | Fail        | yes               | no                 | no                 | no               |
# |-------------------------------------------------------------------------------------------------------------------|
# | Shot               | Success     | yes               | yes                | yes                | yes              |
# |                    | Fail        | yes               | yes                | no                 | yes              |
# |                    | Own goal    | yes               | yes                | no                 | yes              |
# |-------------------------------------------------------------------------------------------------------------------|
# | Penalty shot       | Success     | no                | yes                | yes                | yes              |
# |                    | Fail        | no                | yes                | no                 | yes              |
# |-------------------------------------------------------------------------------------------------------------------|
# | Free-kick shot     | Success     | no                | yes                | yes                | yes              |
# |                    | Fail        | no                | yes                | no                 | yes              |
# |-------------------------------------------------------------------------------------------------------------------|
# | Keeper save        | Success     | yes               | no                 | yes                | yes              |
# |-------------------------------------------------------------------------------------------------------------------|
# | Keeper claim       | Success     | yes               | no                 | yes                | yes              |
# |                    | Fail        | yes               | no                 | no                 | no               |
# |-------------------------------------------------------------------------------------------------------------------|
# | Keeper punch       | Success     | yes               | no                 | yes                | yes              |
# |-------------------------------------------------------------------------------------------------------------------|
# | Keeper pick-up     | Success     | yes               | no                 | yes                | yes              |
# |-------------------------------------------------------------------------------------------------------------------|
# | Clearance          | Success     | yes               | no                 | yes                | yes              |
# |-------------------------------------------------------------------------------------------------------------------|
# | Bad touch          | Fail        | yes               | no                 | no                 | no               |
# |-------------------------------------------------------------------------------------------------------------------|
# | Dribble            | Success     | yes               | yes                | yes                | yes              |
# |-------------------------------------------------------------------------------------------------------------------|
# | Goal kick          | Success     | no                | yes                | yes                | yes              |
# |-------------------------------------------------------------------------------------------------------------------|
# | Owngoal (*)        | Fail        | yes               | no                 | no                 | no/yes           |
# |-------------------------------------------------------------------------------------------------------------------|
# | Kick-off (*)       | Success     | no                | yes                | yes                | yes              |
# |                    | Fail        | no                | yes                | no                 | yes              |
# |-------------------------------------------------------------------------------------------------------------------|


def _locate_possession_starts(actions: pd.DataFrame) -> pd.Series:
    possession_starts = pd.Series([False] * len(actions), dtype=bool)
    curr_possible_start_index = 0
    move_action_count = 0
    for index, action in actions.iterrows():
        team_in_ball_possession = actions.iloc[curr_possible_start_index]['team_id']
        if _action_can_interrupt_sequence(action, team_in_ball_possession):
            curr_possible_start_index = index
            move_action_count = 0
        if _is_set_piece(action):
            curr_possible_start_index = index
            move_action_count = 0
            possession_starts.at[curr_possible_start_index] = True
        if _is_ball_moving_action(action):
            move_action_count += 1
        if move_action_count == 3:
            possession_starts.at[curr_possible_start_index] = True
    return possession_starts


def _action_can_interrupt_sequence(action, curr_team_in_ball_possession) -> bool:
    return not _is_open_play_action(action) or \
        action['team_id'] != curr_team_in_ball_possession and _is_deliberate_ball_moving_action(action) or \
        action['team_id'] != curr_team_in_ball_possession and _is_successful(action) or \
        action['team_id'] != curr_team_in_ball_possession and _is_own_goal(action)


def _locate_modelled_possession_starts(actions: pd.DataFrame, possession_starts: pd.Series) -> pd.Series:
    result = pd.Series([False] * len(actions), dtype=bool)
    possession_start_indices = possession_starts[possession_starts].index.values
    for possession_start_index in possession_start_indices:
        action = actions.iloc[possession_start_index]
        action_type = action['type_name']
        if action_type in _MODELLED_POSSESSION_START_TYPES and \
                (action_type not in _MODELLED_SET_PIECE_POSSESSION_START_TYPES or action['result_name'] == 'success'):
            result.at[possession_start_index] = True
    return result


def _locate_possession_seq_actions(actions: pd.DataFrame, possession_starts: pd.Series) -> pd.Series:
    actions_part_of_possession = pd.Series([False] * len(actions), dtype=bool)
    for possession_start in possession_starts[possession_starts].index.values:
        curr_action_index = possession_start
        action = actions.iloc[curr_action_index]
        team_in_ball_possession = action['team_id']
        while not _action_can_interrupt_sequence(action, team_in_ball_possession) or curr_action_index == \
                possession_start:
            actions_part_of_possession.at[curr_action_index] = True
            curr_action_index += 1
            if curr_action_index >= len(actions):
                break
            action = actions.iloc[curr_action_index]
    return actions_part_of_possession


# ----- Auxiliary ------------------------------------------------------------ #


_MODELLED_POSSESSION_START_TYPES = (
    'pass',
    'throw_in',
    'goalkick',
    'interception',
    'tackle',
    'kick_off',
    'clearance',
    'take_on',
    'dribble',
    'shot',
    'cross',
    'freekick_short'
)


_MODELLED_SET_PIECE_POSSESSION_START_TYPES = (
    'throw_in',
    'freekick_short',
    'goalkick',
    'kick_off'
)


_BALL_REGAIN_ACTION_TYPES = (
    'pass',
    'cross',
    'take_on',
    'tackle',
    'interception',
    'shot',
    'clearance',
    'dribble',
)


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


def _is_open_play_action(action) -> bool:
    return action['type_name'] in (
        'pass',
        'cross',
        'take_on',
        'tackle',
        'interception',
        'shot',
        'keeper_save',
        'keeper_claim',
        'keeper_pick_up',
        'keeper_punch',
        'clearance',
        'bad_touch',
        'dribble',
        'owngoal'
    )


def _is_deliberate_ball_moving_action(action) -> bool:
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


def _is_successful(action) -> bool:
    return action['result_name'] == 'success'


def _is_own_goal(action) -> bool:
    return action['type_name'] == 'owngoal'


def _is_set_piece(action) -> bool:
    return action['type_name'] in (
        'throw_in',
        'freekick_crossed',
        'freekick_short',
        'corner_crossed',
        'corner_short',
        'shot_penalty',
        'shot_freekick',
        'goalkick',
        'kick_off'
    )
