from .player import Player
from .game import Game

class SherlockHolmes(Player):
    """
    Estrategia basado en el personaje de Sherlock Holmes como detective.

    Ideas iniciales:

    1: Ser estable y no entrar en guerras largas, pierden mucha media por ronda.
    2: Ser perdonadores: no asumir mala fe por un único movimiento, puede ser P_error.
    3: Poder defendernos de rivales greedy sin perder contra cooperativos.
    4: Tener un “plan” contra TitForTat, porque en nuestras pruebas era la estrategia top.

    Exploración y resultados observados:

    - Primera aproximación (Idea1): intentamos elegir acción por valor esperado aplicando una fórmula
      de inercia: a * P(opp <= 5-a), usando el historial del rival. Funcionaba “bien” contra random/always,
      pero nos dimos cuenta de dos problemas:
        - Reaccionaba demasiado a rachas cortas (con ruido) y entraba en ciclos malos.
        - Contra estrategias greedy (tipo Focal5) a veces colapsábamos repetidamente.

    - Ajustes tras pruebas:
      - Base estable: empezamos con 2 y, si hay colapso, reseteamos a 2.
         (Con P_error, un colapso puede venir de un flip; castigar fuerte empeora).
      - “Castigo corto” a greedy repetido:
         Hemos visto que nuestra estrategia rendía mal contra perfiles tipo Focal5 (pedían 5 a menudo).
         Por eso añadimos el bloque de detección de greedy (2 de las últimas 3 jugadas >=4) y
         hacemos solo una ronda de castigo (0). Luego volvemos a 2 para no quedarnos atascados.
      - “Anti-TFT”:
         En el torneo normalizado estilo campeonato, TitForTat suele salir arriba por su estabilidad.
         Si detectamos que el rival copia nuestra jugada anterior, usamos un patrón 2/3 alterno.
         Esto evita colapsos y suele dar una ventaja pequeña pero constante frente a TFT.

    - Como resultado global con nuestro evaluador estilo campeonato concluimos que
      nuestra estrategia queda en el bloque alto y es competitiva contra las estrategias de las que
      disponíamos, aunque TitForTat domina por su estabilidad estructural en este formato.

    Resumen del comportamiento:

    - Inicio: juega 2.
    - Si hubo colapso: vuelve a 2 (reset).
    - Si detecta greedy repetido: castiga 1 ronda con 0 y vuelve a 2.
    - Si detecta TitForTat: alterna 2/3.
    - Si no: elige acción por valor esperado de ventana, con ligera inercia para evitar saltos bruscos.

    """

    def __init__(self, game: Game, name: str = "sherlock_holmes") -> None:
        super().__init__(game, name)
        self.ventana: int = 30 # Número de rondas pasadas que analizará.
        self.castigo: int = 0 # Rondas que debe durar un castigo activo.
        self.turno: int = 0 # Para alternar 2/3 contra TFT.
        self.ultima: int = 2 # Acción anterior para inercia suave y así evitar saltos.

    def _ultimos(self, lista: list[int], n: int) -> list[int]:
        """
        Devuelve los últimos n elementos o toda la lista si es más corta.
        """
        return lista[-n:] if len(lista) > n else lista[:]

    def strategy(self, opponent: Player) -> int:
        """
        Decide la acción usando historial propio y del oponente.
        """

        umbral = self.game.threshold
        slf = self.history
        opp = opponent.history


        # 1) LÓGICA DE INICIO Y RESETEO:

        # Primera ronda: señal estable.
        if not slf:
            self.castigo = 0
            self.turno = 0
            self.ultima = 2
            return 2

        # Si colapsamos la ronda anterior: reset robusto al ruido.
        if slf[-1] + opp[-1] > umbral:
            self.castigo = 0
            self.turno = 0
            self.ultima = 2
            return 2

        # Castigo corto si está activo:
        if self.castigo > 0:
            self.castigo -= 1
            self.ultima = 0
            return 0


        # 2) ANÁLISIS DE ESTADÍSTICO Y DETECCIÓN DE PATRONES:

        # Ventana de observación del rival:
        window = self._ultimos(opp, self.ventana)
        n = len(window)

        # Conteo de acciones del rival en ventana:
        conteo = [0] * 6
        for a in window:
            conteo[a] += 1

        # Casos (tipos de estrategias básicas) en función de patrones:

        if conteo[0] / n >= 0.8:
            # -> Always0: explotamos con 5.
            self.ultima = 5
            return 5

        if conteo[3] / n >= 0.7:
            # -> Always3: mejor respuesta estable es 2
            self.ultima = 2
            return 2

        # Anti-greedy: detecta si en 2 de las últimas 3 jugó >=4 (tipo Focal5).
        # Lo añadimos porque vimos que sin esto nuestra media bajaba contra Focal5:
        if len(opp) >= 3 and sum(1 for x in opp[-3:] if x >= 4) >= 2:
            self.castigo = 1 # Se implanta castigo para frenar a estrategias que intentan imponer un reparto desigual.
            self.ultima = 0
            return 0

        # Detectar TitForTat (copia la acción anterior del oponente) y usar patrón 2/3:
        k = min(len(slf), len(opp)) - 1
        k = min(k, self.ventana)

        if k >= 12:
            copias = 0
            for i in range(-k, 0):
                if opp[i] == slf[i - 1]:
                    copias += 1

            if copias / k >= 0.68: # Si hay coincidencia mayor al 68% (número semi-arbitrario), se asume
                                   # que el rival está copiando nuestras acciones.
                a = 2 if (self.turno % 2 == 0) else 3
                self.turno += 1
                self.ultima = a
                return a


        # 3) OPTIMIZACIÓN POR VALOR ESPERADO:

        # Si no hay patrón: optimizar la acción (a) según fórmula -> a * P(opp <= 5-a) con inercia.
        # Acumulado para P(opp <= i):
        acum = [0] * 6
        s = 0
        for i in range(6):
            s += conteo[i]
            acum[i] = s

        mejor_a = 2
        mejor_val = -1.0

        for a in range(6):
            lim = umbral - a
            prob_ok = (acum[lim] / n) if lim >= 0 else 0.0

            # Valor esperado básico + inercia:
            val = a * prob_ok - 0.05 * abs(a - self.ultima) # Este último componente es la
                                                            # 'inercia' para evitar cambiar de
                                                            # estrategia demasiado rápido, con
                                                            # un coeficiente de inercia (0.05) que
                                                            # calibra cuán reacio es al cambio.

            # Frenamos a 4/5 si no vemos bastante masa en 0/1 (reduce colapsos contra castigadores):
            if a >= 4 and (conteo[0] + conteo[1]) / n < 0.55:
                val -= 0.35

            if val > mejor_val:
                mejor_val = val
                mejor_a = a

        self.ultima = mejor_a
        return mejor_a
