import numpy as np

# Acciones del juego de suma limitada:
ACTIONS = (0, 1, 2, 3, 4, 5)
THRESHOLD = 5  # Umbral de suma.

class Game():

    def __init__(self, actions: tuple[int, ...] = ACTIONS, threshold: int = THRESHOLD):
        """
        Represents the limited-sum game.

        Parameters:
            - actions (list[int]): list of possible actions (default: [0,1,2,3,4,5])
            - threshold (int): sum threshold beyond which both get 0 (default: 5)
        """
        self.actions = actions
        self.threshold = threshold

    @property
    def payoff_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Payoff matrix of the game.

        Returns:
            - 6x6 np array of the matrix
        """
        size = len(self.actions)
        matrix_p1 = np.zeros((size, size), dtype=int)
        matrix_p2 = np.zeros((size, size), dtype=int)
        for i, a_1 in enumerate(self.actions):
            for j, a_2 in enumerate(self.actions):
                payoff_p1, payoff_p2 = self.evaluate_result(a_1, a_2)
                matrix_p1[i, j] = payoff_p1
                matrix_p2[i, j] = payoff_p2
        return matrix_p1, matrix_p2

    def evaluate_result(self, a_1: int, a_2: int) -> tuple[int, int]:
        """
        Given two actions, returns the payoffs of the two players.

        Parameters:
            - a_1 (int): action of player 1 (0 to 5)
            - a_2 (int): action of player 2 (0 to 5)

        Returns:
            - tuple of two ints, being the first and second values the payoff
            for the first and second player, respectively.
        """
        return (a_1, a_2) if (a_1 + a_2) <= self.threshold else (0, 0)