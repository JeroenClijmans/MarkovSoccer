import matplotlib.pyplot as plt
import matplotsoccer.fns as fns
import numpy as np
from abc import ABC, abstractmethod
from markovsoccer.prism import PrismModel
from markovsoccer.team_model import TeamModel
from markovsoccer.config import *
from scipy.interpolate import interp2d  # type: ignore
from markovsoccer.util import convert_dictionary_to_heatmap, heatmap_interpolated_visualization


class Feature(ABC):

    @staticmethod
    @abstractmethod
    def calculate(team_model: TeamModel) -> any:
        pass


class SideUsage(Feature):

    def __init__(self):
        return

    @staticmethod
    def calculate(team_model: TeamModel) -> dict[str, float]:
        left = team_model.expected_number_visits_in(fromm=INITIAL_STATE, states=LEFT_STATES)
        center = team_model.expected_number_visits_in(fromm=INITIAL_STATE, states=CENTER_STATES)
        right = team_model.expected_number_visits_in(fromm=INITIAL_STATE, states=RIGHT_STATES)
        total = left + center + right
        d = {
            'left': left / total,
            'right': right / total,
            'center': center / total
        }
        return d

    @staticmethod
    def visualize(team_model: TeamModel):
        zone_usages = SideUsage.calculate(team_model)
        left_val, right_val, center_val = zone_usages['left'], zone_usages['right'], zone_usages['center']
        avgs = dict()
        for i in range(0, 48):
            avgs[i] = left_val
        for i in range(48, 144):
            avgs[i] = center_val
        for i in range(144, 192):
            avgs[i] = right_val
        matrix = convert_dictionary_to_heatmap(avgs, 12, 16)
        ax = heatmap_interpolated_visualization(matrix, show=False, cbar=False, norm_min=0.1, norm_max=0.6)

        cfg = fns.spadl_config
        cell_width = (cfg["width"] - cfg["origin_x"]) / WIDTH
        cell_length = (cfg["length"] - cfg["origin_y"]) / LENGTH
        base_y = cfg["origin_y"]
        base_x = cfg["origin_x"]
        left_x = base_x + 8 * cell_length
        left_y = base_y + 10.5 * cell_width
        center_x = base_x + 8 * cell_length
        center_y = base_y + 6 * cell_width
        right_x = base_x + 8 * cell_length
        right_y = base_y + 1.5 * cell_width
        ax.text(left_x, left_y, '{:.1%} (L)'.format(left_val), horizontalalignment='center',
                verticalalignment='center',
                zorder=10000, fontsize=18)
        ax.text(center_x, center_y, '{:.1%} (C)'.format(center_val), horizontalalignment='center',
                verticalalignment='center',
                zorder=10000, fontsize=18)
        ax.text(right_x, right_y, '{:.1%} (R)'.format(right_val), horizontalalignment='center',
                verticalalignment='center',
                zorder=10000, fontsize=18)
        plt.axis("scaled")
        plt.show()


class SideUsageShot(Feature):

    def __init__(self):
        return

    @staticmethod
    def calculate(team_model: TeamModel) -> dict[str, float]:
        team_model_shots = team_model.construct_model_if_absorbed_in(SHOT_STATES)
        return SideUsage.calculate(team_model_shots)

    @staticmethod
    def visualize(team_model: TeamModel):
        team_model_shots = team_model.construct_model_if_absorbed_in(SHOT_STATES)
        return SideUsage.visualize(team_model_shots)


class InwardsOutwardsPreference(Feature):
    SEQUENCE_LENGTH = 2
    N_SEQUENCES = 200

    def __init__(self):
        return

    @staticmethod
    def calculate(team_model: TeamModel) -> dict[str, float]:
        most_likely_subsequences = team_model.get_most_likely_subsequences(
            InwardsOutwardsPreference.SEQUENCE_LENGTH,
            InwardsOutwardsPreference.N_SEQUENCES
        )
        to_inside = 0
        to_outside = 0
        for seq in most_likely_subsequences:
            if seq.points_inwards():
                to_inside += 1
            elif seq.points_outwards():
                to_outside += 1
        percentage_inwards = to_inside / InwardsOutwardsPreference.N_SEQUENCES
        percentage_outwards = to_outside / InwardsOutwardsPreference.N_SEQUENCES
        return {
            "inwards": percentage_inwards,
            "outwards": percentage_outwards
        }

    @staticmethod
    def visualize_most_likely_subsequences(team_model: TeamModel, n: int):
        most_likely_subsequences = team_model.get_most_likely_subsequences(
            InwardsOutwardsPreference.SEQUENCE_LENGTH,
            n
        )
        for seq in most_likely_subsequences:
            seq.visualize()


class SpeedOfPlay(Feature):

    def __init__(self):
        return

    @staticmethod
    def calculate(team_model: TeamModel) -> float:
        """
        Calculate the speed of play, quantified as the average number of actions
        of sequences ending in a shot.
        :param team_model: The team model
        :return: Average number of actions of sequences ending in a shot
        """
        team_model_shots = team_model.construct_model_if_absorbed_in(SHOT_STATES)
        return team_model_shots.average_number_actions(INITIAL_STATE)


class LongBalls(Feature):

    def __init__(self):
        return

    @staticmethod
    def calculate(team_model: TeamModel) -> float:
        """
        Calculate the likelihood to use long ball, quantified as the probability
        of directly moving the ball from the own half to the offensive third in
        one action.
        :param team_model: The team model
        :return: Probability of directly moving the ball from the own half to
        the offensive third in one action.
        """
        # calculate using the PRISM model checker
        result = 0
        normalization_constant = 0
        for state in OWN_HALF_STATES:
            weight = team_model.expected_number_visits_in(INITIAL_STATE, {state})
            normalization_constant += weight
            result += weight * team_model.probability_of_moving_to(state, OFFENSIVE_THIRD_STATES)
        return result / normalization_constant


class LongGoalKicks(Feature):

    def __init__(self):
        return

    @staticmethod
    def calculate(team_model: TeamModel) -> float:
        """
        Calculate the likelihood of a long goal kick, quantified as the
        probability of kicking a goal kick to the opponent half.
        :param team_model: The team model
        :return: Probability of kicking a goal kick to the opponent half.
        """
        return team_model.probability_of_moving_to(GOALKICK_STATE, OPPONENT_HALF_STATES)


class SuccessfulCounterattackProbability(Feature):

    def __init__(self):
        return

    @staticmethod
    def calculate(team_model: TeamModel) -> float:
        """
        Calculate the probability of launching a successful counterattack,
        quantified as the probability of arriving at a shot within eight actions
        after recovering the ball in the own half.
        :param team_model: The team model
        :return: Probability of arriving at a shot within eight actions after
        recovering the ball in the own half.
        """
        # construct a heatmap of the probability of arriving at a shot within
        # eight actions for each state in the own half
        prism_model = PrismModel.construct_from(team_model)
        heatmap = prism_model.property_heatmap('filter(printall, P=? [ F<=8 "shot_taken" ], "own_half")', WIDTH,
                                               LENGTH)
        heatmap = heatmap[:, :LENGTH // 2]  # only consider the own half
        # weight each state in the own half by the probability of recovering the
        # ball there
        row = team_model.transition_matrix[BALL_REGAIN_STATE, :NB_FIELD_STATES]
        weights = row.reshape((WIDTH, LENGTH))
        weights = weights[:, :LENGTH // 2]  # only consider the own half
        weights = weights / np.sum(weights)
        result = np.sum(np.multiply(heatmap, weights))
        return result


class AbilityToCreateShootingOpportunities(Feature):

    def __init__(self):
        return

    @staticmethod
    def calculate(team_model: TeamModel) -> any:
        shot_states = list(map(lambda s: s - NB_TRANSIENT_STATES, list(SHOT_STATES)))
        probabilities_of_absorption = team_model.B[:NB_FIELD_STATES, np.array(shot_states)]
        probabilities_of_shot = np.sum(probabilities_of_absorption, axis=1)
        probabilities_of_shot_heatmap = probabilities_of_shot.reshape((WIDTH, LENGTH))
        # weight each state in the own half by its relative usage
        weights = team_model.fundamental_matrix[INITIAL_STATE, :NB_FIELD_STATES].reshape((WIDTH, LENGTH))
        weights[:, LENGTH//2:] = 0
        weights = weights / np.sum(weights)
        return np.sum(np.multiply(weights, probabilities_of_shot_heatmap))
