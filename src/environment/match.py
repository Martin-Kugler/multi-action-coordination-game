import numpy as np
from .player import Player

class Match():

    def __init__(self, player_1: Player,
                       player_2: Player,
                       n_rounds: int = 100,
                       error: float = 0.0):
        """
        Match class to represent an iterative limited-sum game

        Parameters:
            - player_1 (Player): first player of the match
            - player_2 (Player): second player of the match
            - n_rounds (int = 100): number of rounds in the match
            - error (float = 0.0): error probability (on a 0-1 scale).
        """

        assert n_rounds > 0, "'n_rounds' should be greater than 0"

        self.player_1 = player_1
        self.player_2 = player_2
        self.n_rounds = n_rounds
        self.error = error

        self.score = [0, 0]  # this variable will store the final result of
                                 # the match, once the 'play()' function has
                                 # been called. The two values of the tuple
                                 # correspond to the points scored by the first
                                 # and second player, respectively.

    def complementary(self, n: int) -> int:
        '''Returns the complementary of an action.
        For example, in case of being in a 0-5 actions,
        1 will return 4 or 3 will return 2 '''
        return max(self.player_1.game.actions) - n

    def play(self, do_print: bool = False) -> None:
        """
        Main call of the class. Play the match.
        Stores the final result in 'self.score'

        Parameters
            - do_print (bool = False): if True, should print the ongoing
            results at the end of each round (i.e. print round number, last
            actions of both players and ongoing score).
        """
        # Clean the histories:
        self.player_1.clean_history()
        self.player_2.clean_history()

        for _ in range(self.n_rounds):

            # Receive the respective strategies:
            strategy_p1 = self.player_1.strategy(self.player_2)
            strategy_p2 = self.player_2.strategy(self.player_1)

            # Apply the possible error by calling the complementary function:
            if np.random.rand() < self.error:
                strategy_p1 = self.complementary(strategy_p1)
                if do_print:
                    print(f'Error of {self.player_1.name}!\n')
            if np.random.rand() < self.error:
                strategy_p2 = self.complementary(strategy_p2)
                if do_print:
                    print(f'Error of {self.player_2.name}!\n')

            # Final score:
            score = self.player_1.compute_scores(self.player_2, strategy_p1, strategy_p2)

            # Update the score:
            self.score[0] += score[0]
            self.score[1] += score[1]

            # Print if indicated:
            if do_print:
                print(f'    Round {_}: {strategy_p1} vs {strategy_p2} --> {score}, score: {self.score}\n')