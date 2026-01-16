from limited_sum.game import Game
from limited_sum.dqn_agent import DQNAgent
from limited_sum.sherlock_holmes import SherlockHolmes
from limited_sum.other_players import *
from limited_sum.player import *
from limited_sum.tournament import Tournament

# Aquí probamos un torneo individual:

game = Game()

# Lista de jugadores: 
players =[ 
    Always0(game), Always3(game), Focal5(game), TitForTat(game),
    GrimTrigger(game), Pavlov(game), Bully(game), Alternator(game),
    CastigadorInfernal(game),
    SherlockHolmes(game), UniformRandom(game), TitForTwoTats(game),
    TwoTitsForTat(game), SoftMajority(game),
    AdaptivePlayer(game), Joss(game), Handshake(game)
]

dqn_agent = DQNAgent(game, "dqn_agent")

all_players = (*players, dqn_agent)

tournament = Tournament(all_players, n_rounds=100, error=0.01, repetitions=1)
tournament.play()
tournament.plot_results()