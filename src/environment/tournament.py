import matplotlib.pyplot as plt
import itertools
from .player import Player
from .match import Match

class Tournament():

    def __init__(self, players: tuple[Player, ...],
                       n_rounds: int = 100,
                       error: float = 0.0,
                       repetitions: int = 2):
        """
        All-against-all tournament

        Parameters:
            - players (tuple[Player, ...]): tuple of players that will play the tournament
            - n_rounds (int = 100): number of rounds in each match
            - error (float = 0.0): error probability (in base 1)
            - repetitions (int = 2): number of matches each player plays against each other player
        """

        self.players = players
        self.n_rounds = n_rounds
        self.error = error
        self.repetitions = repetitions

        # This is a key variable of the class. It is intended to store the
        # ongoing ranking of the tournament. It is a dictionary whose keys are
        # the players in the tournament, and its corresponding values are the
        # points obtained in their interactions with each other. In the end, to
        # see the winner, it will be enough to sort this dictionary by the
        # values.
        self.ranking = {player: 0 for player in self.players}  # initial values

    def sort_ranking(self) -> None:
        """Sort the ranking by the value (score)"""
        self.ranking = dict(sorted(self.ranking.items(), key=lambda x: x[1]))

    def play(self) -> None:
        """
        Main call of the class. It must simulate the championship and update
        the variable 'self.ranking' with the accumulated points obtained by
        each player in their interactions.
        """
        for p1, p2 in itertools.combinations(self.players, 2): # Iterate throught all unique pairs of players.
            for _ in range(self.repetitions): # In order to prevent random result due to possible errors, it will be made a determined number of repetitions of all matches.

                # Clean both histories:
                p1.clean_history()
                p2.clean_history()

                # Play the match:
                match = Match(p1, p2, self.n_rounds, self.error)
                match.play()

                # Update ranking:
                self.ranking[p1] += match.score[0]
                self.ranking[p2] += match.score[1]

        # Sort ranking at the end
        self.sort_ranking()

    def plot_results(self):
        """
        Plots a bar chart of the final ranking. On the x-axis should appear
        the names of the sorted ranking of players participating in the
        tournament. On the y-axis the points obtained.
        """
        x = [player.name for player in self.ranking.keys()]
        y = list(self.ranking.values())

        plt.figure(figsize=(8, 4))
        plt.bar(x, y)

        plt.xlabel("Players")
        plt.ylabel("Total Score")
        plt.title("Tournament Results")

        # Methods for esthetic purposes:
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.show()