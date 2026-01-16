import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import os
from .player import Player


class ReplayBuffer():

    def __init__(self, capacity: int = 10000):
        """
        Estructura de memoria cíclica para la DQN.

        Parámetros:
            - capacity: El número máximo de experiencias que guardaremos, incializado
            de forma predeterminada en 10000.
        """
        # Usamos deque (double-ended queue, una cola doblemente enlazada dada el curso pasado)
        # porque al llegar al límite 'capacity'
        # borra automáticamente el elemento más antiguo (FIFO: First In, First Out).
        self.buffer = deque(maxlen=capacity)


    def push(self, state, action, reward, next_state, done):
        """
        Guarda una nueva transición en la memoria en forma de tupla.
        """
        # Guardamos la tupla (s, a, r, s', done).
        self.buffer.append((state, action, reward, next_state, done))


    def sample(self, batch_size: int):
        """
        Extrae una muestra aleatoria para el entrenamiento.

        Nota: Aquí es donde rompemos esa correlación temporal
        que tratábamos de evitar (que se obsesione con las acciones recientes)
        al mezclar experiencias de diferentes momentos de la partida.
        """
        # Seleccionamos 'batch_size' experiencias al azar, siendo un batch un determinado número de muestras:
        batch = random.sample(self.buffer, batch_size)

        # Separamos los componentes para que la Red Neuronal los procese en bloque (en forma de tensores,
        # que son las estructuras básicas con las que funciona toda Red Neuronal):
        states, actions, rewards, next_states, dones = zip(*batch)

        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.uint8))

    def __len__(self):
        return len(self.buffer)
    

class DQNTrainer():

    def __init__(self, policy_net, target_net, learning_rate=0.001, gamma=0.95):

        # IMPORTANTE: Establecemos la GPU como entorno de ejecución en caso de estar disponible.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = policy_net
        self.target_net = target_net
        self.gamma = gamma
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate) # Establecemos un factor de aprendizaje
                                                                                    # inicial que se irá modificando.
        self.criterion = nn.MSELoss() # Error Cuadrático Medio.

        # Nota: el Adam Optimizer es un algoritmo particular del descenso de gradiente
        # que calibra automáticamente el factor de aprendizaje (learning rate) de forma eficiente.


    def train_step(self, replay_buffer, batch_size):

        if len(replay_buffer) < batch_size:
            return # No entrenamos hasta tener suficientes recuerdos.


        # 1) Extraemos una muestra aleatoria de la memoria:
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Convertimos a tensores de PyTorch y los enviamos a la GPU en caso de estar disponible:
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)


        # 2) Cálculo de la Predicción Actual.

        # Usamos la red que está aprendiendo (policy_net):
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Nota: Esta línea de código lo que hace es un "forward pass" de la red política con los
        # estados dados por la muestra de la memoria (el Replay Buffer), lo cual devuelve una
        # matriz con todos los estados junto con las predicciones de cada acción (los valores Q), y finalmente
        # reúne (gather) únicamente las acciones que realmente se tomaron en el pasado (actions),
        # que son las que nos interesan para comparar lo que la Red pensaba que sería la
        # mejor acción frente al objetivo real calculado por la Ecuación de Bellman.


        # 3) Cálculo del Objetivo (Bellman).

        # Usamos la red congelada (target_net) para ver el futuro con estabilidad:
        with torch.no_grad(): # Esto lo que hace es no entrenar esta red, pues ésta se queda "congelada".

            # A continuación, propagamos (forward_pass) por la red los nuevos estados (next_states)
            # y escogemos los máximos valores de cada uno:
            max_next_q_values = self.target_net(next_states).max(1)[0]

            # Finalmente, aplicamos la Ecuación de Bellman:
            expected_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones)) # Si la partida terminó (done),
                                                                                # el futuro vale 0 (no tiene nada más que calcular).


        # 4) Cálculo del TD Error y optimización.

        # El "loss" es la diferencia al cuadrado entre la realidad y la expectativa
        loss = self.criterion(current_q_values, expected_q_values.unsqueeze(1))

        # Backpropagation: Ajustamos los pesos de la red según el error.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()



# Primero generamos la arquitectura de la Red Neuronal:

class QNetwork(nn.Module): # Hereda de la clase base de PyTorch para redes neuronales.

    def __init__(self, input_dim, output_dim):

        super().__init__()


        # DEFINICIÓN DEL PERCEPTRÓN MULTICAPA (MLP):

        # Entrada: 12 neuronas (ventana de 5 rondas para 2 jugadores + 2 neuronas de contexto).
        self.fc1 = nn.Linear(input_dim, 64) # Capa oculta 1.
        self.fc2 = nn.Linear(64, 64) # Capa oculta 2. Elegimos 64 neuronas para que tenga
                                      # capacidad suficiente para analizar patrones complejos
                                      # pero no tantas como para generar overfitting.

        # Salida: 6 neuronas (valores Q para acciones 0-5).
        self.fc3 = nn.Linear(64, output_dim)


    def forward(self, x):
        '''
        Propagación por la Red Neuronal dadas unas entradas.
        '''

        # Aplicamos activación ReLU en capas ocultas para detectar
        # patrones complejos (aplicamos no-linealidad):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # Salida lineal para predecir la recompensa total esperada (Valor Q):
        return self.fc3(x)



class DQNAgent(Player):

    def __init__(self, game, name="DQN_Master", window_size=5):

        super().__init__(game, name)

        # IMPORTANTE: Establecemos la GPU como entorno de ejecución en caso de estar disponible.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1) CONFIGURACIÓN DE DIMENSIONES:
        self.window_size = window_size # El número de estados recientes del historial que se guardarán.
        self.n_actions = len(game.actions) # 6 acciones (0 a 5).
        self.input_dim = (window_size * 2) + 2   # 12 entradas (5 nuestras, 5 del rival y 2 de contexto).


        # 2) INICIALIZACIÓN DE REDES:

        # Red Política: La que decide y aprende en cada paso (la enviamos a la GPU si está disponible).
        self.policy_net = QNetwork(self.input_dim, self.n_actions).to(self.device)

        # Red Objetivo: La "congelada" para dar estabilidad al aprendizaje (Bellman) (la enviamos a la GPU si está disponible).
        self.target_net = QNetwork(self.input_dim, self.n_actions).to(self.device)

        # Extraemos los pesos de la Red Política (state_dict()) y lo introducimos a la Red Objetivo.
        # Esto permite que ambas redes comiencen en el mismo punto de partida.
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Establecemos la Red Objetivo en "modo evaluación" (es decir, que no se va a entrenar):
        self.target_net.eval()


        # 3) INICIALIZACIÓN DEL BÚFFER (LA MEMORIA):
        self.memory = ReplayBuffer(capacity=10000)


        # 4) INICIALIZACIÓN DEL ENTRENADOR:

        # Importante: Reutilizamos la clase DQNTrainer. Le pasamos las redes y el descuento:
        self.trainer = DQNTrainer(
            policy_net=self.policy_net,
            target_net=self.target_net,
            learning_rate=0.001,
            gamma=0.8
        )


        # 5) HIPERPARÁMETROS DE COMPORTAMIENTO:

        # Debemos establecer cómo se comporta la red en cuanto al archiconocido dilema
        # de exploración-explotación, para lo cual usaremos la técnica de Épsilon-Greedy:

        self.epsilon = 1.0           # Exploración inicial (100% azar).
        self.epsilon_min = 0.05      # Mínimo azar permitido, para evitar que la Red sea predecible.
        self.epsilon_decay = 0.995   # Factor de aprendizaje: cuánto dejamos de explorar en cada paso.

        self.batch_size = 64         # Cuántos recuerdos repasamos en cada entrenamiento.
        self.target_update_freq = 100 # Cada cuántas rondas sincronizamos la Red Objetivo
                                      # (para que no se mantenga todo el rato "congelado").
        self.steps_done = 0          # Contador para poder saber cuándo actualizar la Red Objetivo.

        # Variable que guardará el estado de la ronda anterior para poder aprender:
        self.last_state = None

        # Variables que determinan cuándo entrenar la red (para mejorar rendimiento):
        self.train_freq = 5
        self.steps_since_train = 0 # Contador.

        # Intentar cargar el modelo ya entrenado si el archivo existe.
        # Buscar el archivo en el mismo directorio donde está dqn_agent.py:
        model_path = os.path.join(os.path.dirname(__file__), 'modelo.pth')
        if os.path.exists(model_path):
            self.policy_net.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.epsilon = 0.05 # Dado que ya está entrenado, casi no necesita azar.
            print(f"{self.name}: Modelo cargado.")
        else:
            print(f"{self.name}: No se encontró modelo. Iniciando desde cero.")



    def _get_state(self, opponent):
        """
        PASO 1: OBSERVACIÓN.
        Convierte el historial de acciones en un vector de entrada (el estado).
        """

        # 1) Obtenemos las últimas acciones del historial:
        my_hist = self.history[-self.window_size:]
        op_hist = opponent.history[-self.window_size:]

        state = []
        for i in range(self.window_size):
            if i < len(my_hist):
                # Normalizamos (0-5 -> 0.0-1.0) para que la red neuronal sea estable:
                state.append(my_hist[-(i+1)] / 5.0)
                state.append(op_hist[-(i+1)] / 5.0)
            else:
                # Si la partida acaba de empezar, rellenamos con 0 (padding):
                state.append(0.0)
                state.append(0.0)

        # 2) Obtenemos el contexto del rival (media y desviación):
        if len(opponent.history) > 0:
            op_mean = np.mean(opponent.history) / 5.0
            op_std = np.std(opponent.history) / 5.0
        else:
            op_mean, op_std = 0.0, 0.0

        state.append(op_mean)
        state.append(op_std)

        return np.array(state, dtype=np.float32)


    def strategy(self, opponent) -> int:
        """
        PASO 2 Y 3: PREDICCIÓN Y ACTUACIÓN.
        Decidimos qué acción escoger (0-5) usando Épsilon-Greedy.

        Nota: Aquí es donde realizamos el truco de hacer que la Red aprenda sobre
        la ronda anterior justo antes de realizar la acción. Es por ello que se llama
        a la función "update_knowledge"
        """

        # FASE DE ENTRENAMIENTO:

        # Si ya hemos jugado una ronda y tenemos el estado anterior guardado:
        if len(self.history) > 0 and self.last_state is not None:

            # 1) Recuperamos qué pasó en la última ronda:
            last_action = self.history[-1]
            last_opp_action = opponent.history[-1]

            # 2) Calculamos la recompensa que recibimos en esa ronda:
            reward = float(last_action) if (last_action + last_opp_action <= 5) else 0.0

            # 3) El "estado siguiente" es el estado actual antes de decidir la nueva acción:
            current_state = self._get_state(opponent)

            # 4) Ejecutamos el aprendizaje llamando a "update_knowledge":
            self.update_knowledge(self.last_state, last_action, reward, current_state, False)


        # FASE DE PREDICCIÓN Y ACTUACIÓN:

        # Obtenemos el estado actual y lo guardamos para aprender de él en la siguiente ronda:
        state_np = self._get_state(opponent)
        self.last_state = state_np

        # Convertimos a Tensor de PyTorch (formato: 1 fila, 10 columnas):
        state_t = torch.FloatTensor(state_np).unsqueeze(0).to(self.device) # Lo enviamos a la GPU si está disponible.

        # Mecanismo de Exploración (azar) vs Explotación (conocimiento):
        if np.random.rand() < self.epsilon:

            # Explorar - Probar un número al azar para descubrir nuevas tácticas y
            # evitar óptimos locales o que la Red sea predecible.
            return np.random.choice(self.game.actions)

        # Explotar - Usar la Red para elegir la mejor opción:
        with torch.no_grad(): # Desactivamos el motor de cálculo de gradientes de PyTorch pues
                              # no se pretende actualizar la Red (así se evita consumo de RAM).

            # Propagación de los estados en forma de tensor por la Red (forward pass):
            q_values = self.policy_net(state_t)

            # Elegimos la acción (0-5, que coincide con el índice) que tenga el valor Q más alto:
            return torch.argmax(q_values).item()


    def update_knowledge(self, state, action, reward, next_state, done):
        """
        PASO 4, 5 Y 6: RECEPCIÓN DE RECOMPENSA, CÁLCULO DEL OBJETIVO Y ENTRENAMIENTO.
        Se llama antes de cada ronda para que el agente aprenda de lo que pasó
        según la ronda pasada antes de realizar su acción.
        """

        # 1) Guardamos la experiencia en la memoria y aumentamos el número de rondas sin entrenar:
        self.memory.push(state, action, reward, next_state, done)
        self.steps_since_train += 1

        # 2) Si tenemos suficientes recuerdos y la llevamos sin entrenar 5 rondas o más, entrenamos la Red:
        if len(self.memory) > self.batch_size and self.steps_since_train >= self.train_freq:

            # Usamos el entrenador (dado por la clase DQNTrainer) para ajustar los pesos de la Red Política:
            self.trainer.train_step(self.memory, self.batch_size)
            self.steps_since_train = 0 # Resetear contador.

            # Reducimos el Épsilon (la Red deja de ser tan "curioso" o "explorativo"):
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 3) Sincronización - Cada x pasos, actualizamos la Red Objetivo:
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            # Copiamos los pesos aprendidos a la red "congelada":
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
