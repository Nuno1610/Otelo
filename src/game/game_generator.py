import numpy as np
import os
import random
import pandas as pd
from multiprocessing import Pool
from game import othello
from agent import mcts_uct

# Constantes (mantener alias en español para compatibilidad)
WHITE = 1
BLACK = 2
BLANCO = WHITE
NEGRO = BLACK


def assign_reward(player, winner):
    """Asigna recompensa desde la perspectiva de `player`.

    Devuelve 1 si `player` ganó, -1 si perdió, 0 en empate.
    """
    if winner == 0:
        return 0
    if player == WHITE:
        return 1 if winner == WHITE else -1
    return 1 if winner == BLACK else -1


def simulate_self_play(iterations=1000):
    """Simula una partida donde ambos jugadores usan MCTS (devuelve lista de ejemplos).

    Cada elemento de la lista es tupla `(state, active_player, reward)`.
    """
    board = othello.create_board()
    current = BLACK  # comienza siempre el negro
    history = []
    skipped_turns = 0

    while not othello.is_board_full(board) and skipped_turns < 2:
        moves = othello.valid_movements(board, current)
        if not moves:
            skipped_turns += 1
            current = 3 - current
            continue
        skipped_turns = 0

        move = mcts_uct.mcts_uct(state=board, player=current, iterations=iterations, training=True)

        othello.apply_movement(board, move[0], move[1], current)
        snapshot = np.copy(board)
        current = 3 - current
        history.append((snapshot, current))

    winner = othello.decide_winner(board)

    examples = []
    for state, active_player in history:
        reward = assign_reward(active_player, winner)
        examples.append((state, active_player, reward))

    return examples


def simulate_vs_random(iterations=1000):
    """Simula una partida: agente MCTS vs jugador aleatorio.

    Devuelve `(examples, agent_won_flag)` donde `agent_won_flag` es 1 si el agente ganó.
    """
    board = othello.create_board()
    history = []
    skipped_turns = 0

    current_player = BLACK
    agent_color = random.choice([WHITE, BLACK])

    while not othello.is_board_full(board) and skipped_turns < 2:
        moves = othello.valid_movements(board, current_player)
        if not moves:
            skipped_turns += 1
            current_player = 3 - current_player
            continue
        skipped_turns = 0

        if current_player == agent_color:
            move = mcts_uct.mcts_uct(state=board, player=current_player, iterations=iterations, neural=True)
        else:
            move = random.choice(moves)

        othello.apply_movement(board, move[0], move[1], current_player)
        snapshot = np.copy(board)
        current_player = 3 - current_player
        history.append((snapshot, current_player))

    winner = othello.decide_winner(board)

    agent_won = 0
    if winner == agent_color:
        print("Gana el agente")
        agent_won = 1
    elif winner == 0:
        print("Empate")
    else:
        print("Pierde el agente")

    examples = [(state, active_player, assign_reward(active_player, winner)) for state, active_player in history]
    return examples, agent_won


def simulate_vs_previous(iterations=1000):
    """Simula una partida: agente con red neuronal vs agente con política anterior.

    Devuelve `(examples, agent_won_flag)`.
    """
    board = othello.create_board()
    history = []
    skipped_turns = 0

    current_player = BLACK
    neural_agent = random.choice([WHITE, BLACK])

    while not othello.is_board_full(board) and skipped_turns < 2:
        moves = othello.valid_movements(board, current_player)
        if not moves:
            skipped_turns += 1
            current_player = 3 - current_player
            continue
        skipped_turns = 0

        if current_player == neural_agent:
            move = mcts_uct.mcts_uct(state=board, player=current_player, iterations=iterations, neural=True)
        else:
            move = mcts_uct.mcts_uct(state=board, player=current_player, iterations=iterations, neural=False)

        othello.apply_movement(board, move[0], move[1], current_player)
        snapshot = np.copy(board)
        current_player = 3 - current_player
        history.append((snapshot, current_player))

    winner = othello.decide_winner(board)

    agent_won = 0
    if winner == neural_agent:
        print("Gana el agente con red neuronal")
        agent_won = 1
    elif winner == 0:
        print("Empate")
    else:
        print("Pierde el agente con red neuronal")

    examples = [(state, active_player, assign_reward(active_player, winner)) for state, active_player in history]
    return examples, agent_won


def generate_games_parallel(simulation_function, num_games=500, iterations=1000, processes=4):
    """Genera datos ejecutando `simulation_function` en paralelo `num_games` veces.

    - Si la función devuelve (data, agent_won), también retorna el total de victorias.
    - Si la función devuelve solo data, retorna la lista combinada.
    """
    args = [(iterations,) for _ in range(num_games)]
    with Pool(processes=processes) as pool:
        results = pool.starmap(simulation_function, args)

    data = []
    # Detección basada en el tipo de retorno: si cada resultado es tupla (data, flag)
    if simulation_function in (simulate_vs_previous, simulate_vs_random):
        victories = 0
        for game_data, agent_won in results:
            data.extend(game_data)
            victories += agent_won
        return data, victories

    for game_data in results:
        data.extend(game_data)
    return data


if __name__ == "__main__":
    processes = max(1, os.cpu_count() - 1)

    while True:
        try:
            num_games, iterations = map(int, input("Número de partidas y número de iteraciones (separados por espacio): ").split())
            break
        except Exception:
            print("Entrada inválida.")

    while True:
        try:
            choice = int(input("1) agente vs sí mismo  2) agente vs aleatorio  3) agente NN vs agente anterior: "))
            if choice in (1, 2, 3):
                simulation_function = simulate_self_play if choice == 1 else simulate_vs_random if choice == 2 else simulate_vs_previous
                break
            print("Introduce 1, 2 o 3")
        except Exception:
            print("Entrada inválida.")

    while True:
        try:
            mode = int(input("1) sobrescribir datos  2) extender conjunto de datos: "))
            if mode in (1, 2):
                break
            print("Introduce 1 o 2")
        except Exception:
            print("Entrada inválida.")

    if simulation_function in (simulate_vs_previous, simulate_vs_random):
        data, victories = generate_games_parallel(simulation_function, num_games=num_games, iterations=iterations, processes=processes)
    else:
        data = generate_games_parallel(simulation_function, num_games=num_games, iterations=iterations, processes=processes)

    base_route = os.path.dirname(os.path.abspath(__file__))
    data_route = os.path.join(base_route, "..", "data")
    os.makedirs(data_route, exist_ok=True)
    file_route = os.path.join(data_route, "training_data.pkl")

    df = pd.DataFrame(data, columns=['state', 'player', 'result'])
    if os.path.exists(file_route) and mode == 2:
        old_df = pd.read_pickle(file_route)
        final_df = pd.concat([old_df, df], ignore_index=True)
    else:
        final_df = df

    final_df.to_pickle(file_route)

    if simulation_function == simulate_vs_random:
        print(f"Victorias del agente: {victories} de {num_games} partidas jugadas")
        print(f"Porcentaje de victorias: {(victories / num_games) * 100}%")
    elif simulation_function == simulate_vs_previous:
        print(f"Victorias del agente con red neuronal: {victories} de {num_games} partidas jugadas")
        print(f"Porcentaje de victorias: {(victories / num_games) * 100}%")

    print(f"Guardados {len(data)} ejemplos para entrenamiento.")


# Wrappers para compatibilidad con los nombres anteriores en español
asignar_recompensa = assign_reward
simulate_agent_vs_agent = simulate_self_play
simulate_agent_vs_random = simulate_vs_random
simulate_agent_vs_old = simulate_vs_previous
generate_data_parallel = generate_games_parallel