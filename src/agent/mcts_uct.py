import math
import random
import numpy as np
import keras
import os
import tensorflow as tf
from game import othello
from agent.model import model as mod

c = 1 / math.sqrt(2)  # Constante de exploración para UCT (balance entre exploración y explotación)

# Cargo el modelo (Red Neuronal)
base_route = os.path.dirname(os.path.abspath(__file__))
model_route = os.path.join(base_route, "model/othello_model.keras")
model = keras.models.load_model(model_route)

class MCTSNode:
    def __init__(self, state, player, parent=None, action=None):
        self.state = state  # Estado del tablero en el nodo
        self.player = player  # Jugador que moverá desde este estado
        self.parent = parent
        self.action = action  # La acción que llevó a este nodo desde el padre. Es un vector, representando la casilla a la que se mueve
        self.children = []
        self.visits = 0
        self.total_reward = 0 # Suma de 0s, 1s y -1s, según gane o pierda jugador en los distintos nodos
        self.not_explored = None # Posibles movimientos aún no explorados

    def is_terminal(self):   # Nodo terminal: El juego ha terminado en este estado
        return othello.is_game_finished(self.state)

    def expand(self):
        if self.not_explored is None:
            # Inicializa la lista de movimientos válidos para explorar
            self.not_explored = othello.valid_movements(self.state, self.player)
            random.shuffle(self.not_explored)  # Aleatoriza para diversidad en la expansión
        
        # Saco un movimiento(acción) para expandir y creo el nodo hijo correspondiente
        action = self.not_explored.pop() # No da error ya que si ya no quedan movimientos por explorar, no se llama a expand
        next_state = np.copy(self.state)
        othello.apply_movement(next_state, action[0], action[1], self.player)
        next_player = 3 - self.player  # Cambio el turno al siguiente jugador
        child = MCTSNode(next_state, next_player, parent=self, action=action)
        self.children.append(child)
        return child

    def is_totally_expanded(self): # Compruebo si todos los movimientos posibles ya fueron explorados
        if self.not_explored is None:
            # Inicializa la lista de movimientos válidos si no está inicializada
            self.not_explored = othello.valid_movements(self.state, self.player)
        return len(self.not_explored) == 0

    def best_child(self, c, noise_std=0.01, training=False):

        def ucb1(n): # n objeto de tipo MCTSNode
            if n.visits == 0:
                return float('inf')  # Prioriza hijos no visitados para explorarlos
            
            base_score = (n.total_reward / n.visits) + c * math.sqrt(2 * math.log(self.visits) / n.visits) # Fórmula UCB1 que combina recompensa media y exploración controlada por c
            noise = random.gauss(0, noise_std) # Ruido gaussiano para que haya algo de aleatoriedad en las simulaciones de agente con red vs agente con red
            return base_score + noise

        return max(self.children, key=ucb1)  # Devuelve el hijo que maximiza la ecuacion ucb

    def backup(self, reward): # Algoritmo de retropropagación
        node = self
        current_reward = reward
        while node is not None:
            node.visits += 1
            node.total_reward += current_reward # Acumulo recompensas
            current_reward = -current_reward # Cambio de perspectiva de jugador para reflejar la recompensa en el adversario
            node = node.parent

def tree_policy(node, c):
    while not node.is_terminal():
        movs = othello.valid_movements(node.state, node.player)
        if not movs:  # No hay movimientos, pasar turno
            # Buscar si ya hay hijo que representa pase de turno (action == None)
            pass_turn_child = next((child for child in node.children if child.action is None), None) # Devuelve primer elemento que cumple la condicion de que la accion sea nula
            if pass_turn_child is not None:
                node = pass_turn_child # Paso directamente este nodo para no crear varios nodos exactamente iguales
                continue
            else:
                child = MCTSNode(np.copy(node.state), 3 - node.player, parent=node, action=None)
                node.children.append(child)
                return child

        if not node.is_totally_expanded():
            child = node.expand()
            return child
        else:
            best = node.best_child(c)
            node = best
    return node

def default_policy(state, player): # Siempre recibe el root player, que es para el que hay que calcular la recompensa
    input = mod.convert_board_state(state, player) # Convierte el estado del tablero al formato que espera la red neuronal, desde la perspectiva del jugador raiz
    input = np.expand_dims(input, axis=0)  # Añade una dimensión extra para simular un batch de tamaño 1 (necesario para la red neuronal)
    input_tensor = tf.convert_to_tensor(input, dtype=tf.float32)
    prediction = model(input_tensor, training = False)[0][0].numpy()  # Obtengo la predicción a partir del tensor, y convierto a un valor de numpy
    return prediction


def default_policy_old(state, root_player, node_player): # Default policy antigua, pillando movimientos random. Se le pasa como parámetros root_player, el jugador para el que se estima la reward, y node_player, el jugador activo
    board = np.copy(state)
    actual_player = node_player
    skipped_turns = 0

    # Simulación con acciones aleatorias hasta terminar la partida
    while not othello.is_board_full(board) and skipped_turns < 2:
        movs = othello.valid_movements(board, actual_player)
        if not movs:
            skipped_turns += 1
            actual_player = 3 - actual_player
            continue
        skipped_turns = 0
        mov = random.choice(movs)
        othello.apply_movement(board, mov[0], mov[1], actual_player)
        actual_player =  3 - actual_player

    winner = othello.get_winner(board)
    if winner == 0:
        return 0  # Empate
    return 1 if root_player == winner else -1  # +1 si gana el jugador original, -1 si pierde

def mcts_uct(state, player, iterations=1000, neural=True, training=False):  # Algoritmo principal MCTS con UCT # iterations: número de simulaciones para mejorar la decisión
    # Asumimos que va a haber movimientos válidos en la raiz, ya que si no no se llama a la función en un primer lugar
    root = MCTSNode(np.copy(state), player)

    for _ in range(iterations):
        node = tree_policy(root, c)  # Selección y expansión del nodo
        if(neural):
            reward = default_policy(node.state, root.player)  # Simulación con red neuronal como default policy
        else:
            reward = default_policy_old(node.state, root.player, node.player)  # Simulación con default policy propia de mcts uct
        node.backup(reward)  # Retropropagación
    if (neural):
        best_node = root.best_child(0, training) # Selecciona el hijo con mejor recompensa media (c=0, solo explotación). Alternativa: Devolver hijo con más visitas.
    else:
        best_node = root.best_child(0) # Si no se usa la red neuronal nunca se va a introducir ruido
    return best_node.action
