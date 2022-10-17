"""
Representation and inspection of discrete-time Markov chains.
"""
from typing import List

import matplotlib.pyplot as plt
import matplotsoccer.fns as fns
import numpy as np
from markovsoccer.config import *
from markovsoccer.util import heatmap_visualization, get_cell_indices_from_flat


class DTMC:

    # ----- initialization --------------------------------------------------- #

    def __init__(self,
                 transition_matrix: np.ndarray,
                 initial_state: int,
                 start_absorbing_states: int = None):
        """
        Initialization

        :param transition_matrix: 2D-matrix representing the transition probabilities from each state to each other
        state. The transition matrix has to be ordered, i.e., transient states must have lower indices than
        absorbing states.
        :param initial_state: initial state
        :param start_absorbing_states: Index of the first absorbing state in the transition matrix. A value of None
        means that there are only transient states.
        """
        self.transition_matrix = transition_matrix
        self.initial_state = initial_state
        self.start_absorbing_states = start_absorbing_states
        self.size = transition_matrix.shape[0]

        # R is the matrix containing the transition probabilities from transient to absorbing states
        self.R = self.transition_matrix[:start_absorbing_states, start_absorbing_states:]

        # Q is the matrix containing the transition probabilities from transient to transient states
        self.Q = self.transition_matrix[:start_absorbing_states, :start_absorbing_states]

        # fundamental matrix
        self.fundamental_matrix = DTMC._calc_fundamental_matrix(self.Q)

        # B_ij is the probability of eventually ending in absorbing state j if starting in transient state i
        self.B = np.matmul(self.fundamental_matrix, self.R)

        # t_i is the average number of actions in i before ending in an absorbing state
        self.t = np.dot(self.fundamental_matrix, np.ones(NB_TRANSIENT_STATES))

    @staticmethod
    def _calc_fundamental_matrix(transition_matrix: np.ndarray):
        try:
            return DTMC._calc_fundamental_matrix_precise_solution(transition_matrix)
        except np.linalg.LinAlgError:
            return DTMC._calc_fundamental_matrix_abridged(transition_matrix)

    @staticmethod
    def _calc_fundamental_matrix_precise_solution(transition_matrix: np.ndarray):
        return np.linalg.inv(np.identity(NB_TRANSIENT_STATES) - transition_matrix)

    @staticmethod
    def _calc_fundamental_matrix_abridged(transition_matrix: np.ndarray):
        # if the matrix is not invertible, then an abridged version of the fundamental matrix formula can be used
        abridge_after = 50
        fundamental_matrix_abridged = np.zeros((NB_TRANSIENT_STATES, NB_TRANSIENT_STATES))
        term = transition_matrix.copy()
        for _ in range(abridge_after):
            fundamental_matrix_abridged = fundamental_matrix_abridged + term
            term = np.matmul(term, transition_matrix)
        return fundamental_matrix_abridged

    # ----- Model checking --------------------------------------------------- #

    def expected_number_visits_in(self, fromm: int, states: set):
        """
        Compute the average number of visits to a particular set of states when starting from a given state.
        :param fromm: The state to start from.
        :param states: The states to count.
        :return: Average number of visits to the given set of states when starting from the given state.
        """
        result = 0
        for state in states:
            result += self.fundamental_matrix[fromm, state]
        return result

    def probability_of_moving_to(self, fromm: int, states: set):
        """
        Compute the probability of transitioning from a particular state to any of the states in the given set.
        :param fromm: The state to start from.
        :param states: The states to transition to.
        :return: The probability of transitioning from the given state to any of the states in the given set.
        """
        result = 0
        for state in states:
            result += self.transition_matrix[fromm, state]
        return result

    def get_most_likely_subsequences(self, sequence_length: int, n=None):
        sequences = list()
        # weight each start state by its usage
        normalization_constant = 0
        for state in FIELD_STATES:
            normalization_constant += self.expected_number_visits_in(self.initial_state, {state})
        for state in FIELD_STATES:
            probability = self.expected_number_visits_in(self.initial_state, {state}) / normalization_constant
            sequences.append(TransitionSequence((state,), probability))
        # calculate subsequence probabilities
        for _ in range(sequence_length):
            new_sequences = list()
            for sequence in sequences:
                new_sequences += sequence.extensions(self.transition_matrix, NB_FIELD_STATES)
            sequences = new_sequences
        sequences = sorted(sequences, key=lambda x: x.probability, reverse=True)
        if n is None:
            return sequences
        else:
            return sequences[:n]

    # ----- Related DTMC's --------------------------------------------------- #

    def construct_model_if_absorbed_in(self, absorbing_states: set) -> "DTMC":
        """
        Construct the DTMC which only generates the sequences of this DTMC which end in the given set of absorbing
        states.
        :param absorbing_states: The states to condition on
        :return: DTMC conditioned on the given set of absorbing states.
        """
        nb_transient_states = self.start_absorbing_states
        nb_absorbing_states = self.size - self.start_absorbing_states
        R_new = self._construct_R_if_absorbed_in(absorbing_states)
        Q_new = self._construct_Q_if_absorbed_in(absorbing_states)
        transition_matrix_new = np.zeros((self.size, self.size))
        transition_matrix_new[:self.start_absorbing_states, :self.start_absorbing_states] = Q_new
        transition_matrix_new[:self.start_absorbing_states, self.start_absorbing_states:] = R_new
        transition_matrix_new[self.start_absorbing_states:, :self.start_absorbing_states] = \
            np.zeros((nb_absorbing_states, nb_transient_states))
        transition_matrix_new[self.start_absorbing_states:, self.start_absorbing_states:] = np.identity(nb_absorbing_states)
        return DTMC(transition_matrix_new, initial_state=self.initial_state, start_absorbing_states=self.start_absorbing_states)

    def _construct_R_if_absorbed_in(self, absorbing_states: set):
        nb_transient_states = self.start_absorbing_states
        nb_absorbing_states = self.size - self.start_absorbing_states
        R_new = np.zeros((nb_transient_states, nb_absorbing_states))
        for transient_state in range(nb_transient_states):
            divisor = 0
            for absorbing_state in absorbing_states:
                absorbing_state_index = absorbing_state - nb_transient_states
                divisor += self.B[transient_state, absorbing_state_index]
            for absorbing_state in absorbing_states:
                absorbing_state_index = absorbing_state - nb_transient_states
                R_new[transient_state, absorbing_state_index] = self.R[transient_state, absorbing_state_index] / divisor
        return R_new

    def _construct_Q_if_absorbed_in(self, absorbing_states: set):
        Db = self._construct_Db_if_absorbed_in(absorbing_states)
        Db_inv = np.linalg.pinv(Db)
        Q_new = np.matmul(np.matmul(Db_inv, self.Q), Db)
        return Q_new

    def _construct_Db_if_absorbed_in(self, absorbing_states: set):
        nb_transient_states = self.start_absorbing_states
        b_values = np.zeros(nb_transient_states)
        for absorbing_state in absorbing_states:
            absorbing_state_index = absorbing_state - nb_transient_states
            b_values = np.add(b_values, self.B[:, absorbing_state_index])
        return np.diag(b_values)


# ----- transition sequences ------------------------------------------------- #


class TransitionSequence:
    def __init__(self, start_sequence, start_probability):
        self.sequence = start_sequence
        self.probability = start_probability

    def __len__(self):
        return len(self.sequence)

    def __str__(self):
        return "Seq" + str(self.sequence)

    def get_state(self, n: int):
        return self.sequence[n]

    def get_first_state(self):
        return self.sequence[0]

    def get_last_state(self):
        return self.sequence[len(self)-1]

    def points_inwards(self):
        fx, _ = get_cell_indices_from_flat(self.get_first_state())
        ex, _ = get_cell_indices_from_flat(self.get_last_state())
        center = (WIDTH-1) / 2  # minus 1 because the maximal index is WIDTH-1 (indices start from 0 instead of 1)
        return abs(ex - center) < abs(fx - center)  # end cell is closer to center than start cell

    def points_outwards(self):
        fx, _ = get_cell_indices_from_flat(self.get_first_state())
        ex, _ = get_cell_indices_from_flat(self.get_last_state())
        center = (WIDTH - 1) / 2  # minus 1 because the maximal index is WIDTH-1 (indices start from 0 instead of 1)
        return abs(ex - center) > abs(fx - center)  # end cell is further from center than start cell

    def extensions(self, transition_matrix, nb_states: int) -> List['TransitionSequence']:
        result = list()
        for state in FIELD_STATES:
            if state == self.get_last_state():
                continue
            new_sequence = self.sequence + (state,)
            new_probability = self.probability * transition_matrix[self.get_last_state(), state]
            if new_probability > 0.000001:
                result.append(
                    TransitionSequence(new_sequence, new_probability)
                )
        return result

    def visualize(self):
        return TransitionSequence._sequence_heatmap(self)

    @staticmethod
    def _sequence_heatmap(sequence):
        matrix = np.zeros((WIDTH, LENGTH))
        for i in range(len(sequence)):
            state = sequence.get_state(i)
            row = state // LENGTH
            col = state % LENGTH
            matrix[row, col] = 0.8
        ax = heatmap_visualization(matrix, show=False, ax=None, figsize=None, alpha=1, cmap="hot",
                                   linecolor="white", cbar=False, norm_min=0, norm_max=1)
        cfg = fns.spadl_config

        for i in range(len(sequence) - 1):
            state = sequence.get_state(i)
            state2 = sequence.get_state(i + 1)
            row1 = (WIDTH - 1) - state // LENGTH
            col1 = state % LENGTH
            row2 = (WIDTH - 1) - state2 // LENGTH
            col2 = state2 % LENGTH
            base_y = cfg["origin_y"] + cfg["width"] / (2 * WIDTH)
            base_x = cfg["origin_x"] + cfg["length"] / (2 * LENGTH)
            cell_width = (cfg["width"] - cfg["origin_x"]) / WIDTH
            cell_length = (cfg["length"] - cfg["origin_y"]) / LENGTH
            start_x = base_x + col1 * cell_length
            start_y = base_y + row1 * cell_width
            end_x = base_x + col2 * cell_length
            end_y = base_y + row2 * cell_width
            length_x = end_x - start_x
            length_y = end_y - start_y
            ax.arrow(start_x, start_y, length_x, length_y, length_includes_head=True, width=0.7, zorder=10000)
        plt.axis("scaled")
        plt.show()
