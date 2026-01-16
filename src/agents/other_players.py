from .player import Player
from .game import Game
import random
import numpy as np


# Estrategia del Profesor
class CastigadorInfernal(Player):
    """
    Adaptive strategy for the limited-sum game that balances coordination and self-protection.

    Strategy:
    - Starts trying to coordinate on i+j=5 (efficient outcome)
    - Monitors opponent's cooperation patterns and adapts accordingly
    - Uses graduated punishment for greedy behavior
    - Attempts forgiveness and cooperation recovery
    - Adjusts strategy based on opponent's consistency
    """

    def __init__(self, game: Game, name: str = ""):
        super().__init__(game, name)
        self.cooperation_score = 0  # Track opponent's cooperative behavior
        self.punishment_mode = False
        self.punishment_rounds = 0

    def strategy(self, opponent: Player) -> int:
        """
        Adaptive strategy with cooperation tracking and graduated response
        """
        # First round: start with 2 (middle ground)
        if not self.history:
            return 2

        last_opponent = opponent.history[-1]

        # Update cooperation tracking
        if last_opponent <= 3:
            self.cooperation_score += 1
        else:
            self.cooperation_score -= 2

        # Analyze opponent's recent pattern (last 5 rounds)
        recent_rounds = min(5, len(opponent.history))
        recent_actions = opponent.history[-recent_rounds:]
        avg_recent = sum(recent_actions) / len(recent_actions)

        # Simplified punishment mechanism
        if self.punishment_mode:
            self.punishment_rounds += 1
            # Simple punishment: play 0 for 2 rounds, then try to recover
            if self.punishment_rounds <= 2:
                return 0
            else:
                # Reset and try to recover cooperation
                self.punishment_mode = False
                self.punishment_rounds = 0
                return 2

        # Detect consistently greedy behavior
        if last_opponent > 3 and avg_recent > 3.5:
            self.punishment_mode = True
            self.punishment_rounds = 0
            return 0

        # Normal coordination attempt
        if last_opponent <= 3:
            # Try to maintain sum=5
            return max(0, min(5, 5 - last_opponent))

        # Default fallback
        return 2
    
# Añadimos nuevos tipos de jugadores básicos generados por Gemini para poder
# entrenar mejor nuestro modelo:

class GrimTrigger(Player):
    """Cooperativo hasta que le fallan una vez, luego juega 0 para siempre."""
    def __init__(self, game, name="Grim"):
        super().__init__(game, name)
        self.triggered = False
    def strategy(self, opponent):
        if not opponent.history: return 3
        if opponent.history[-1] + self.history[-1] > 5: self.triggered = True
        return 3 if not self.triggered else 0

class Pavlov(Player):
    """Win-Stay, Lose-Shift: si ganó puntos, repite. Si colapsó, cambia."""
    def __init__(self, game, name="Pavlov"):
        super().__init__(game, name)
    def strategy(self, opponent):
        if not self.history: return 2
        last_sum = self.history[-1] + opponent.history[-1]
        if last_sum <= 5: return self.history[-1] # Stay
        return random.choice([0, 1, 2]) # Shift to safer

class Bully(Player):
    """Agresivo: pide 4 o 5 para forzar al rival a bajar a 1 o 0."""
    def __init__(self, game, name="Bully"):
        super().__init__(game, name)
    def strategy(self, opponent):
        return 4 if random.random() > 0.1 else 5

class Alternator(Player):
    """Cambia entre 2 y 3 para testear la adaptabilidad del rival."""
    def __init__(self, game, name="Alternator"):
        super().__init__(game, name)
    def strategy(self, opponent):
        return 3 if len(self.history) % 2 == 0 else 2

class Prober(Player):
    """Empieza con [2, 3, 2]. Si el rival es blando, pide 4 o 5. Si no, coopera."""
    def __init__(self, game, name="Prober"):
        super().__init__(game, name)
    def strategy(self, opponent):
        if len(self.history) < 3: return [2, 3, 2][len(self.history)]
        # Si el rival siempre ha jugado <= 2, le exploto
        if all(h <= 2 for h in opponent.history): return 4
        return 3 - (opponent.history[-1] % 2) # Cooperación inestable

class TitForTwoTats(Player):
    """Más perdonador que TFT: solo baja a 0 si el rival colapsó dos veces seguidas."""
    def __init__(self, game, name="T2T"):
        super().__init__(game, name)
    def strategy(self, opponent):
        if len(opponent.history) < 2: return 2
        # Si las dos últimas rondas del rival causaron colapso (> 5 sumado a lo mío)
        if (opponent.history[-1] + self.history[-1] > 5) and \
           (opponent.history[-2] + self.history[-2] > 5):
            return 0
        return 3 - (opponent.history[-1] % 2) # Intenta coordinar en 2 o 3

class TwoTitsForTat(Player):
    """Doble castigo: si el rival causa un colapso, juega 0 durante dos rondas."""
    def __init__(self, game, name="2TFT"):
        super().__init__(game, name)
        self.punish_rounds = 0
    def strategy(self, opponent):
        if self.punish_rounds > 0:
            self.punish_rounds -= 1
            return 0
        if opponent.history and (opponent.history[-1] + self.history[-1] > 5):
            self.punish_rounds = 2
            return 0
        return 2 # Base segura

class SoftMajority(Player):
    """Juega la acción más frecuente del rival, siempre que no sea agresiva (>3)."""
    def __init__(self, game, name="SoftMaj"):
        super().__init__(game, name)
    def strategy(self, opponent):
        if not opponent.history: return 2
        counts = np.bincount(opponent.history, minlength=6)
        most_frequent = np.argmax(counts)
        return most_frequent if most_frequent <= 3 else 2

class AdaptivePlayer(Player):
    """Calcula qué acción le ha dado más puntos contra este rival en el pasado."""
    def __init__(self, game, name="Adaptive"):
        super().__init__(game, name)
        self.performance = {a: 0 for a in range(6)}
    def strategy(self, opponent):
        if len(self.history) > 0:
            last_a = self.history[-1]
            last_r = last_a if (last_a + opponent.history[-1] <= 5) else 0
            self.performance[last_a] += last_r

        # Elige la acción con mejor acumulado histórico
        return max(self.performance, key=self.performance.get)

class Joss(Player):
    """Tit-For-Tat 'pícaro': normalmente copia, pero un 10% de las veces pide 5 de la nada."""
    def __init__(self, game, name="Joss"):
        super().__init__(game, name)
    def strategy(self, opponent):
        if not opponent.history: return 2
        if random.random() < 0.10: return 5 # Traición aleatoria
        return 5 - opponent.history[-1] if opponent.history[-1] <= 5 else 0

class Handshake(Player):
    """Solo coopera si el rival juega una secuencia específica (ej: 1, 2). Si no, juega 0."""
    def __init__(self, game, name="Handshake"):
        super().__init__(game, name)
        self.recognized = False
    def strategy(self, opponent):
        if len(opponent.history) == 2:
            if opponent.history[0] == 1 and opponent.history[1] == 2:
                self.recognized = True
        if len(opponent.history) < 2: return [1, 2][len(opponent.history)]
        return 3 if self.recognized else 0
