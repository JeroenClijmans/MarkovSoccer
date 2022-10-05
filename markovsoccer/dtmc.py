"""
Representation and inspection of discrete-time Markov chains.
"""
import numpy as np
from markovsoccer.config import *


class DTMC:

    # ----- initialization --------------------------------------------------- #

    def __init__(self,
                 transition_matrix: np.ndarray,
                 start_absorbing_states: int = None):
        """
        Initialization

        :param transition_matrix: 2D-matrix representing the transition probabilities from each state to each other
        state. The transition matrix has to be ordered, i.e., transition states must have lower indices than
        absorbing states.
        :param start_absorbing_states: Index of the first absorbing state in the transition matrix. A value of None
        means that there are only transient states.
        """
        self.transition_matrix = transition_matrix
        self.start_absorbing_states = start_absorbing_states
        self.size = transition_matrix.shape[0]  # todo: check if works

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

    def average_number_visits_in(self, fromm: int, states: set):
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

    # ----- Related DTMC's --------------------------------------------------- #

    def construct_dtmc_if_absorbed_in(self, absorbing_states: set):
        nb_transient_states = self.start_absorbing_states
        nb_absorbing_states = self.size - self.start_absorbing_states  # todo: check if correct
        R_new = self._construct_R_if_absorbed_in(absorbing_states)
        Q_new = self._construct_Q_if_absorbed_in(absorbing_states)
        transition_matrix_new = np.zeros((self.size, self.size))
        transition_matrix_new[:self.start_absorbing_states, :self.start_absorbing_states] = Q_new
        transition_matrix_new[:self.start_absorbing_states, self.start_absorbing_states:] = R_new
        transition_matrix_new[self.start_absorbing_states:, :self.start_absorbing_states] = \
            np.zeros((nb_absorbing_states, nb_transient_states))
        transition_matrix_new[self.start_absorbing_states:, self.start_absorbing_states:] = np.identity(nb_absorbing_states)
        return DTMC(transition_matrix_new, start_absorbing_states=self.start_absorbing_states)

    def _construct_R_if_absorbed_in(self, absorbing_states: set):
        nb_transient_states = self.start_absorbing_states
        nb_absorbing_states = self.size - self.start_absorbing_states  # todo: check if correct
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
