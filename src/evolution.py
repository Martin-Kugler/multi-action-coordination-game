import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from .player import Player
from .tournament import Tournament

class Evolution():

    def __init__(self, players: tuple[Player, ...],
                       n_rounds: int = 100,
                       error: float = 0.0,
                       repetitions: int = 2,
                       generations: int = 100,
                       reproductivity: float = 0.05,
                       initial_population: tuple[int, ...] | int = 100):
        """
        Evolutionary tournament

        Parameters:
            - players (tuple[Player, ...]): tuple of players that will play the
         tournament
            - n_rounds (int = 100): number of rounds in each game
            - error (float = 0.0): error probability (in base 1)
            - repetitions (int = 2): number of games each player plays against
         the rest
            - generations (int = 100): number of generations to simulate
            - reproductivity (float = 0.05): ratio (base 1) of worst players
         that will be removed and substituted by the top ones in the natural
         selection process carried out at the end of each generation
            - initial_population (tuple[int, ...] | int = 100): list of
         individuals representing each players (same index as 'players' tuple)
         OR total population size (int).
        """

        self.players = players
        self.n_rounds = n_rounds
        self.error = error
        self.repetitions = repetitions
        self.generations = generations
        self.reproductivity = reproductivity

        if isinstance(initial_population, int):
            self.initial_population = [math.floor(initial_population
                                       / len(self.players))
                                       for _ in range(len(self.players))]
        else:
            self.initial_population = initial_population

        self.total_population = sum(self.initial_population)
        self.repr_int = int(self.total_population * self.reproductivity)

        self.ranking = {copy.deepcopy(player): 0 for i, player in
                        enumerate(self.players)
                        for _ in range(self.initial_population[i])}

    def natural_selection(self, result_tournament: dict[Player, int]) -> dict[Player, int]:
        """
        Kill the worst guys, reproduce the top ones. Takes the ranking once a
        face-to-face tournament has been played and returns another ranking,
        with the evolutionary changes applied

        Parameters:
            - result_tournament: the 'tournament.ranking' kind of sorted dict.

        Results:
            - Same kind of dict ranking as the input, but with the evolutionary
         dynamics applied
        """
        # Delete the last players:
        for player in list(result_tournament.keys())[:self.repr_int]:
            del result_tournament[player]

        # Append the new players:
        for player in list(result_tournament.keys())[-self.repr_int:]:
            result_tournament[copy.deepcopy(player)] = 0

        return result_tournament

    def count_strategies(self) -> dict[str, int]:
        """
        Counts the number of played alive of each strategy, based on the
        initial list of players. Should be computed analyzing the
        'self.ranking' variable. Useful for the results plot/print (not needed
        for the tournament itself)

        Results:
            - A dict, containing as keys the name of the players and as
         values the number of individuals they have now alive in the tournament
        """
        counts = {}
        for player in self.ranking.keys():
            name = player.name
            counts[name] = counts.get(name, 0) + 1
        return counts

    def play(self, do_print: bool = False):
        """
        Main call of the class. Performs the computations to simulate the
        evolutionary tournament.

        Parameters
            - do_print (bool = False): if True, should print the ongoing
         results at the end of each generation (i.e. print generation number,
         and number of individuals playing each strategy).
        """
        # Variable which shows the evolution of the number of individuals
        # of each player during each generation:

        count_evolution = {player.name: [val] for player, val in
                             zip(self.players, self.initial_population)}

        for generation in range(1, self.generations + 1):
            # Create and play each tournament, receiving the respective ranking
            # and updating it with natural_selection:
            tournament = Tournament(tuple(self.ranking.keys()), self.n_rounds, self.error, self.repetitions)
            tournament.play()
            self.ranking = self.natural_selection(tournament.ranking)

            # Append the new counts of individuals in the count_evolution dict
            # based on the values recieved by count_strategies():
            counts = self.count_strategies()
            for player_name in count_evolution.keys():
                count_evolution[player_name].append(counts.get(player_name, 0))

            if do_print:
                print(f'    Generation number {generation}: {counts}.\n')

        return count_evolution

    def stackplot(self, count_evolution: dict[str, list]) -> None:
        """
        Plots a 'stackplot' of the evolution of the tournament

        Parameters:
            - count_evolution (dict[Player, list]): a dictionary containing as
         keys the name of the strategies of the different players of the
         tournament. Each value is a list, where the 'i'-th position of that
         list indicates the number of individuals that player has at the end of
         the 'i'-th generation
         """

        COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']

        for i, name in enumerate(count_evolution.keys()):
            plt.plot([], [], label=name, color= COLORS[(i) % len(COLORS)])

        plt.stackplot(list(range(self.generations + 1)),
                      np.array(list(count_evolution.values())), colors=COLORS)

        plt.legend()
        plt.show()

