from abc import ABC, abstractmethod
import numpy as np
from .game import Game

from typing import Self, Sequence

class Player(ABC):

    @abstractmethod
    def __init__(self, game: Game, name: str = ""):
        """
        Abstract class that represents a generic player

        Parameters:
            - name (str): the name of the strategy
            - game (Game): the game that this player will play
        """

        self.name = name
        self.game = game

        self.history  = []  # This is the main variable of this class. It is
                            # intended to store all the history of actions
                            # performed by this player.
                            # Example: [0, 1, 2, 3] <- So far, the
                            # interaction lasts four rounds. In the first one,
                            # this player chose 0. In the second, 1. Etc.

    @abstractmethod
    def strategy(self, opponent: Self) -> int:
        """
        Main call of the class. Gives the action for the following round of the
        interaction, based on the history

        Parameters:
            - opponent (Player): is another instance of Player.

        Results:
            - An integer representing the action (0 to 5)
        """
        pass

    def compute_scores(self, opponent, action_self: int | None = None,
                       action_opponent: int | None = None) -> tuple[float, float]:
        """
        Compute the scores for a given opponent and introduces the respective
        actions into the respective histories of both players.

        Parameters:
            - opponent (Player): is another instance of Player.

        Results:
            - A tuple of two floats, where the first value is the current
            player's payoff, and the second value is the opponent's payoff.
        """
        # Get the actions of each player according to their strategies if not given:
        if action_self is None:
            action_self = self.strategy(opponent)
        if action_opponent is None:
            action_opponent = opponent.strategy(self)

        # Update their histories:
        self.history.append(action_self)
        opponent.history.append(action_opponent)

        # Return the score:
        return self.game.evaluate_result(action_self, action_opponent)

    def clean_history(self):
        """Resets the history of the current player"""
        self.history = []


# The basic strategies for the limited-sum game are represented below:

class Always0(Player):

    def __init__(self, game: Game, name: str = "always_0"):
        """Always chooses 0"""
        super().__init__(game, name)

    def strategy(self, opponent: Player) -> int:
        """Always chooses 0"""
        return 0


class Always3(Player):

    def __init__(self, game: Game, name: str = "always_3"):
        """Always chooses 3"""
        super().__init__(game, name)

    def strategy(self, opponent: Player) -> int:
        """Always chooses 3"""
        return 3


class UniformRandom(Player):

    def __init__(self, game: Game, name: str = "uniform_random"):
        """Chooses uniformly at random"""
        super().__init__(game, name)

    def strategy(self, opponent: Player) -> int:
        """Chooses uniformly at random"""
        return np.random.choice(self.game.actions)


class Focal5(Player):

    def __init__(self, game: Game, name: str = "focal_5"):
        """Tries to coordinate on i+j=threshold"""
        super().__init__(game, name)

    def strategy(self, opponent: Player) -> int:
        """First round: threshold // 2, then adapts based on opponent trying to maximize the
        chances of establishing a 5-way split in each round by returning the difference
        between the threshold and the last number returned by the opponent"""
        return self.game.threshold // 2 if opponent.history == [] else (self.game.threshold - opponent.history[-1])


class TitForTat(Player):

    def __init__(self, game: Game, name: str = "tit_for_tat"):
        """Tit-for-tat adapted to the JCMA."""
        super().__init__(game, name)

    def strategy(self, opponent: Player) -> int:
        """Similar to Focal5, but reactive with opponent's actions above threshold // 2,
        in which case he will mirror the opponent's action."""
        if opponent.history == []:
            return self.game.threshold // 2
        else:
            return opponent.history[-1] if opponent.history[-1] > (self.game.threshold // 2) \
                else (self.game.threshold - opponent.history[-1])