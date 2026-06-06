import math
import random
import numpy as np
import keras
import os
import tensorflow as tf
from game import othello
from agent.model import model as mod

# Exploración UCT: constante usada en la fórmula UCB
EXPLORATION_CONSTANT = 1 / math.sqrt(2)

# Cargar modelo de política/valor (red neuronal)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model/othello_model.keras")
model = keras.models.load_model(MODEL_PATH)


class MonteCarloNode:
    """Nodo del árbol MCTS que representa un estado de tablero.

    Atributos:
    - state: copia del tablero (np.ndarray)
    - player: jugador que debe mover en este nodo
    - parent: nodo padre (None para la raíz)
    - action: acción (x,y) que llevó a este nodo desde el padre (None para pase)
    - children: lista de nodos hijos
    - visits: número de visitas a este nodo
    - total_reward: suma de recompensas observadas (desde la perspectiva del nodo)
    - untried_actions: movimientos aún no explorados desde este nodo
    """

    def __init__(self, state, player, parent=None, action=None):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0
        self.untried_actions = None

    def is_terminal(self):
        """True si el estado es terminal (partida finalizada)."""
        return othello.is_game_finished(self.state)

    def expand(self):
        """Expande un hijo tomando una acción no explorada y devolviendo el nuevo nodo hijo."""
        if self.untried_actions is None:
            self.untried_actions = othello.valid_movements(self.state, self.player)
            random.shuffle(self.untried_actions)

        action = self.untried_actions.pop()
        next_state = np.copy(self.state)
        othello.apply_movement(next_state, action[0], action[1], self.player)
        next_player = 3 - self.player
        child = MonteCarloNode(next_state, next_player, parent=self, action=action)
        self.children.append(child)
        return child

    def fully_expanded(self):
        """Devuelve True si no quedan acciones por explorar desde este nodo."""
        if self.untried_actions is None:
            self.untried_actions = othello.valid_movements(self.state, self.player)
        return len(self.untried_actions) == 0

    def best_child(self, exploration_param, noise_std=0.01, training=False):
        """Selecciona el mejor hijo según UCB1; si `training` añade ruido para diversidad."""

        def ucb_score(child):
            if child.visits == 0:
                return float('inf')
            exploitation = child.total_reward / child.visits
            exploration = exploration_param * math.sqrt(2 * math.log(self.visits) / child.visits)
            score = exploitation + exploration
            if training:
                score += random.gauss(0, noise_std)
            return score

        return max(self.children, key=ucb_score)

    def backup(self, reward):
        """Propaga la recompensa hacia la raíz, alternando perspectiva en cada nivel."""
        node = self
        current_reward = reward
        while node is not None:
            node.visits += 1
            node.total_reward += current_reward
            current_reward = -current_reward
            node = node.parent


def select_and_expand(node, exploration_constant, training=False):
    """Baja por el árbol hasta encontrar un nodo no terminal para expandir.

    - Si no hay movimientos en un nodo se crea/usa un hijo que representa el pase de turno.
    """
    while not node.is_terminal():
        moves = othello.valid_movements(node.state, node.player)
        if not moves:
            # Buscar hijo que represente el pase de turno
            pass_child = next((c for c in node.children if c.action is None), None)
            if pass_child is not None:
                node = pass_child
                continue
            else:
                child = MonteCarloNode(np.copy(node.state), 3 - node.player, parent=node, action=None)
                node.children.append(child)
                return child

        if not node.fully_expanded():
            return node.expand()
        node = node.best_child(exploration_constant, training=training)
    return node


def neural_evaluate(state, root_player):
    """Evalúa `state` con la red neuronal desde la perspectiva de `root_player`.

    Devuelve un valor escalar en [-1,1] (o según la normalización usada en el modelo).
    """
    arr = mod.convert_board_state(state, root_player)
    arr = np.expand_dims(arr, axis=0)
    input_tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
    prediction = model(input_tensor, training=False)[0][0].numpy()
    return prediction


def rollout_random(state, root_player, node_player):
    """Simulación aleatoria (política antigua): juega hasta el final con movimientos aleatorios.

    `root_player` es el jugador para el que queremos la recompensa; `node_player` es quien mueve en `state`.
    """
    board = np.copy(state)
    current = node_player
    skipped = 0

    while not othello.is_board_full(board) and skipped < 2:
        moves = othello.valid_movements(board, current)
        if not moves:
            skipped += 1
            current = 3 - current
            continue
        skipped = 0
        mv = random.choice(moves)
        othello.apply_movement(board, mv[0], mv[1], current)
        current = 3 - current

    winner = othello.get_winner(board)
    if winner == 0:
        return 0
    return 1 if root_player == winner else -1


def run_mcts(root_state, root_player, iterations=1000, use_neural=True, training=False):
    """Ejecuta MCTS+UCT y devuelve la acción (x,y) elegida.

    - `use_neural`: si True usa la red neuronal como política de evaluación.
    - `training`: si True añade ruido durante la selección para diversificar exploración.
    """
    root = MonteCarloNode(np.copy(root_state), root_player)

    for _ in range(iterations):
        node = select_and_expand(root, EXPLORATION_CONSTANT, training=training)
        if use_neural:
            reward = neural_evaluate(node.state, root.player)
        else:
            reward = rollout_random(node.state, root.player, node.player)
        node.backup(reward)

    # Elegir el mejor hijo para jugar: c=0 para solo explotación
    best = root.best_child(0, training=training) if use_neural else root.best_child(0)
    return best.action


# Wrappers/aliases para mantener compatibilidad
MCTSNode = MonteCarloNode
tree_policy = select_and_expand
default_policy = neural_evaluate
default_policy_old = rollout_random
mcts_uct = run_mcts
EXPLORATION_C = EXPLORATION_CONSTANT
def mcts_uct(state, player, iterations=1000, neural=True, training=False):
    """Compatibilidad: wrapper con la firma antigua.

    Llama a `run_mcts` mapeando los parámetros antiguos a los nuevos.
    """
    return run_mcts(state, player, iterations=iterations, use_neural=neural, training=training)
