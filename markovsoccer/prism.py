"""
Class to facilitate converting a team model to a model that can be used by the
PRISM model checker.
"""

import abc
import decimal
import os
from collections import OrderedDict
from decimal import Decimal
from math import floor

from markovsoccer.config import *


# ----- PRISM variables ------------------------------------------------------ #


class PrismModuleVariable:

    def __init__(self, name: str):
        self.name = name

    def get_name(self):
        return self.name

    @abc.abstractmethod
    def to_code(self) -> str:
        pass


class PrismModuleIntegerVariable(PrismModuleVariable):

    def __init__(self, name: str, lower_bound: int, upper_bound: int, init: int):
        super().__init__(name)
        if lower_bound > upper_bound:
            raise ValueError("Lower bound ({}) is greater than upper bound ({})".format(lower_bound, upper_bound))
        if not lower_bound <= init <= upper_bound:
            raise ValueError("Init is not within the given bounds")
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.init = init

    def to_code(self) -> str:
        return "{} : [{}..{}] init {};".format(self.name, self.lower_bound, self.upper_bound, self.init)


# ----- PRISM module commands ------------------------------------------------ #


def round_down(nb: float, decimals: int) -> Decimal:
    """
    Code inspired by: https://stackoverflow.com/questions/41383787/round-down-to-2-decimal-in-python
    """
    with decimal.localcontext() as ctx:
        d = decimal.Decimal(nb)
        ctx.rounding = decimal.ROUND_DOWN
        return round(d, decimals)


def _convert_float_probabilities_to_decimals(probs: dict, decimals: int) -> dict:
    result = probs.copy()
    temp = OrderedDict(sorted(probs.items(), key=lambda x: round_down(x[1], decimals) - Decimal(
        x[1])))  # sorted on highest truncate error
    total_truncate_error = 1 - sum([round_down(probs[x], decimals) for x in probs.keys()])
    count_needs_to_be_rounded_up = int(total_truncate_error * 10 ** decimals)
    for key, val in temp.items():
        if count_needs_to_be_rounded_up > 0:
            unit = Decimal(1) / Decimal(10 ** decimals)
            result[key] = round_down(temp[key], decimals) + unit
            count_needs_to_be_rounded_up -= 1
        else:
            result[key] = round_down(temp[key], decimals)
    return result


def _filter_probabilities(probs: dict) -> dict:
    result = dict()
    for key, val in probs.items():
        if val > Decimal(0.000000000000000001):
            result[key] = val
    return result


def _preprocess_probabilities(probs: dict) -> dict:
    decimal_probs = _convert_float_probabilities_to_decimals(probs, 18)
    return _filter_probabilities(decimal_probs)


class PrismModuleCommand:

    def __init__(self, name: str, req: str, effect: dict):
        self.name = name
        self.req = req
        self.effect = _preprocess_probabilities(effect)

    def to_code(self) -> str:
        if len(self.effect) == 0:
            return ""

        probabilities_code = ""
        for outcome in self.effect.keys():
            outcome_probability = self.effect[outcome]
            if probabilities_code == "":
                probabilities_code += "{:.18f}:{}".format(outcome_probability, outcome)
            else:
                probabilities_code += " + {:.18f}:{}".format(outcome_probability, outcome)
        return "[{}] {} -> {};".format(self.name, self.req, probabilities_code)


# ----- PRISM module --------------------------------------------------------- #


class PrismModule:

    def __init__(self, name: str):
        self.name = name
        self.variables = []
        self.commands = []

    def get_name(self):
        return self.name

    def add_integer_variable(self, name: str, lower_bound: int, upper_bound: int,
                             init: int) -> PrismModuleIntegerVariable:
        if name in [v.get_name() for v in self.variables]:
            raise ValueError("Variable with name {} already exists".format(name))
        new_variable = PrismModuleIntegerVariable(name, lower_bound, upper_bound, init)
        self.variables.append(new_variable)
        return new_variable

    def add_command(self, name: str, req: str, effect: dict) -> PrismModuleCommand:
        """
        Add a new command to this PRISM module
        :param name: name
        :param req: requirements for the command to be allowed
        :param effect: effects of the command. This has to be a dictionary of {effect: probability} pairs, in which
               the effect is a string directly in the PRISM language. Effect probabilities must sum to 1.
        :return: new command
        """
        new_command = PrismModuleCommand(name, req, effect)
        self.commands.append(new_command)
        return new_command

    def to_code(self) -> str:
        code = ""
        code += "module {}\n".format(self.get_name())
        for var in self.variables:
            code += "\n\t"
            code += var.to_code()
        code += "\n"
        for command in self.commands:
            code += "\n\t"
            code += command.to_code()
        code += "\n\nendmodule"
        return code

    def _add_transitions(self, team_model: 'TeamModel'):
        """
        Adds transitions from a team model to the module.
        :param team_model: the team model
        """
        for state in range(NB_STATES):
            transition_probabilities = {}
            for state2 in range(NB_STATES):
                transition_prob = team_model.transition_matrix[state, state2]
                transition_probabilities[f"(state'={str(state2)})"] = transition_prob
            req = f"state={str(state)}"
            self.add_command("random", req, transition_probabilities)


# ----- PRISM rewards -------------------------------------------------------- #


class PrismRewardRule:
    def __init__(self, req: str, reward: str):
        self.req = req
        self.reward = reward

    def to_code(self) -> str:
        return "{} : {};".format(self.req, self.reward)


class PrismReward:

    def __init__(self, name: str):
        self.name = name
        self.rules = []

    def get_name(self):
        return self.name

    def add_rule(self, req: str, reward: str) -> PrismRewardRule:
        """
        Add a new reward rule to this PRISM reward
        :param req: state requirements for the reward to be given (directly in PRISM language)
        :param reward: reward expression (directly in PRISM language)
        :return: new reward rule
        """
        new_rule = PrismRewardRule(req, reward)
        self.rules.append(new_rule)
        return new_rule

    def to_code(self) -> str:
        code = 'rewards "{}"'.format(self.name)
        for rule in self.rules:
            code += "\n\t" + rule.to_code()
        code += "\nendrewards"
        return code


# ----- PRISM labels --------------------------------------------------------- #


class PrismLabel:

    def __init__(self, name: str, state_req: str):
        self.name = name
        self.state_req = state_req

    def get_name(self):
        return self.name

    def to_code(self) -> str:
        return 'label "{}" = {};'.format(self.name, self.state_req)


# ----- PRISM model ---------------------------------------------------------- #


class PrismModel:
    allowedTypes = frozenset({"dtmc"})

    def __init__(self, model_type: str):
        if model_type not in PrismModel.allowedTypes:
            raise ValueError("{} is not a valid type argument, please choose one of the following: 'dtmc'")
        self.model_type = model_type
        self.modules = []
        self.rewards = []
        self.labels = []

    # ----- modification ----------------------------------------------------- #

    def add_module(self, name: str) -> PrismModule:
        if name in [m.get_name() for m in self.modules]:
            raise ValueError("Module with name {} already exists".format(name))

        new_module = PrismModule(name)
        self.modules.append(new_module)
        return new_module

    def find_module(self, name: str) -> PrismModule:
        for index in range(0, len(self.modules)):
            if self.modules[index].get_name() == name:
                return self.modules[index]
        raise ValueError("Module with name {} not found".format(name))

    def add_reward(self, name: str) -> PrismReward:
        if name in [r.get_name() for r in self.rewards]:
            raise ValueError("Reward with name {} already exists".format(name))
        new_reward = PrismReward(name)
        self.rewards.append(new_reward)
        return new_reward

    def add_label(self, name: str, state_req: str) -> PrismLabel:
        """
        Add a new label to the model.
        :param name: name of the label
        :param state_req: requirement for a state to attain the label. Must be an expression in PRISM language.
        :return: new PRISM label
        """
        if name in [l.get_name() for l in self.labels]:
            raise ValueError("Label with name {} already exists".format(name))
        new_label = PrismLabel(name, state_req)
        self.labels.append(new_label)
        return new_label

    # ----- code ------------------------------------------------------------- #

    def to_code(self) -> str:
        code = ""
        code += "//generated using PRISM converter\n\n{}".format(self.model_type)
        for module in self.modules:
            code += "\n\n" + module.to_code()
        for reward in self.rewards:
            code += "\n\n" + reward.to_code()
        code += "\n"
        for label in self.labels:
            code += "\n" + label.to_code()
        return code

    def write_to_file(self, path: str):
        f = open(path, "w+")
        f.write(self.to_code())
        f.write("\n")
        f.close()

    # ----- convert team model to PRISM model -------------------------------- #

    @staticmethod
    def construct_from(team_model: 'TeamModel'):
        # initialization
        prism_model = PrismModel("dtmc")
        module = prism_model.add_module("model")

        # define states
        module.add_integer_variable("state", 0, NB_STATES - 1, INITIAL_STATE)

        # add transitions
        module._add_transitions(team_model)

        # add labels
        prism_model._add_init_label()
        prism_model._add_start_labels()
        prism_model._add_absorbing_state_labels()
        prism_model._add_half_labels()
        prism_model._add_third_labels()
        prism_model._add_penalty_box_label()

        # add rewards
        prism_model._add_nb_actions_reward()
        prism_model._add_left_center_right_rewards()

        return prism_model

    def _add_init_label(self):
        init_cell_req = "state={}".format(str(INITIAL_STATE))
        self.add_label("initial_state", init_cell_req)

    def _add_start_labels(self):
        req1 = "state={}".format(str(BALL_REGAIN_STATE))
        self.add_label("ball_regain", req1)
        req2 = "state={}".format(str(THROW_IN_STATE))
        self.add_label("throw_in", req2)
        req3 = "state={}".format(str(SHORT_FREE_KICK_STATE))
        self.add_label("freekick_short", req3)
        req4 = "state={}".format(str(GOALKICK_STATE))
        self.add_label("goalkick", req4)
        req5 = "state={}".format(str(KICKOFF_STATE))
        self.add_label("kick_off", req5)

    def _add_absorbing_state_labels(self):
        move_not_successful_state_req = "state={}".format(str(MOVE_NOT_SUCCESSFUL_STATE))
        self.add_label("move_not_successful", move_not_successful_state_req)
        shot_not_successful_state_req = "state={}".format(str(SHOT_NOT_SUCCESSFUL_STATE))
        self.add_label("shot_not_successful", shot_not_successful_state_req)
        goal_req = "state={}".format(str(GOAL_STATE))
        self.add_label("goal", goal_req)
        shot_req = "state={} | state={}".format(str(SHOT_NOT_SUCCESSFUL_STATE), str(GOAL_STATE))
        self.add_label("shot_taken", shot_req)

    def _add_half_labels(self):
        own_half_req = ""
        opponent_half_req = ""
        for state in OWN_HALF_STATES:
            if len(own_half_req) == 0:
                own_half_req = "state={}".format(state)
            else:
                own_half_req += " | state={}".format(state)
        for state in OPPONENT_HALF_STATES:
            if len(opponent_half_req) == 0:
                opponent_half_req = "state={}".format(state)
            else:
                opponent_half_req += " | state={}".format(state)
        self.add_label("own_half", own_half_req)
        self.add_label("opponent_half", opponent_half_req)

    def _add_third_labels(self):
        defensive_third_req = ""
        middle_third_req = ""
        offensive_third_req = ""
        for cell in FIELD_STATES:
            (cell_y, cell_x) = _get_cell_indices(cell)
            if cell_x < LENGTH // 3:
                if len(defensive_third_req) == 0:
                    defensive_third_req = "state={}".format(cell)
                else:
                    defensive_third_req += " | state={}".format(cell)
            elif cell_x >= LENGTH - LENGTH // 3:
                if len(offensive_third_req) == 0:
                    offensive_third_req = "state={}".format(cell)
                else:
                    offensive_third_req += " | state={}".format(cell)
            else:
                if len(middle_third_req) == 0:
                    middle_third_req = "state={}".format(cell)
                else:
                    middle_third_req += " | state={}".format(cell)
        self.add_label("defensive_third", defensive_third_req)
        self.add_label("offensive_third", offensive_third_req)
        self.add_label("middle_third", middle_third_req)

    def _add_penalty_box_label(self):
        penalty_box_req = ""
        for cell in FIELD_STATES:
            (cell_y, cell_x) = _get_cell_indices(cell)
            if 0.204 * WIDTH <= cell_y < floor(0.796 * WIDTH) and \
                    cell_x > 0.814 * LENGTH:
                if (len(penalty_box_req)) == 0:
                    penalty_box_req = "state={}".format(cell)
                else:
                    penalty_box_req += " | state={}".format(cell)
        self.add_label("penalty_box", penalty_box_req)

    def _add_nb_actions_reward(self):
        number_actions_reward = self.add_reward('number_actions')
        number_actions_conditional = self._states_to_req(FIELD_STATES)
        number_actions_reward.add_rule(number_actions_conditional, '1')

    def _add_left_center_right_rewards(self):
        left_reward = self.add_reward('left')
        left_conditional = self._states_to_req(LEFT_STATES)
        left_reward.add_rule(left_conditional, '1')
        right_reward = self.add_reward('right')
        right_conditional = self._states_to_req(RIGHT_STATES)
        right_reward.add_rule(right_conditional, '1')
        center_reward = self.add_reward('center')
        center_conditional = self._states_to_req(CENTER_STATES)
        center_reward.add_rule(center_conditional, '1')

    @staticmethod
    def _states_to_req(states):
        req = ""
        for state in states:
            req = f"state={state}" if (req == "") else f"{req} | state={state}"
        return req


# ----- auxiliary ------------------------------------------------------------ #


def _get_cell_indices(flat_cell_index: int) -> (int, int):
    x = flat_cell_index % LENGTH  # length index
    y = flat_cell_index // LENGTH  # width index
    return y, x
