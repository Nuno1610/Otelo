import numpy as np
import os
import random
import pandas as pd
from multiprocessing import Pool
from game import othello
from agent import mcts_uct

BLANCO = 1
NEGRO = 2

def simulate_agent_vs_agent(iterations=1000):
    board = othello.create_board()
    player = NEGRO # Empieza el negro siempre
    states = []
    skipped_turns = 0

    while not othello.is_board_full(board) and skipped_turns < 2:

        movs = othello.valid_movements(board, player)
        if not movs: # Si no hay movimientos validos, se skipea el turno, no se puede aplicar el algoritmo
            skipped_turns += 1
            player = 3 - player
            continue
        skipped_turns = 0

        mov = mcts_uct.mcts_uct(state=board, player=player, iterations=iterations, training=True) # Aplica algoritmo mcts con uct para encontrar movimiento óptimo

        othello.apply_movement(board, mov[0], mov[1], player)
        actual_state = np.copy(board)
        player = 3 - player
        states.append((actual_state, player))

    winner = othello.decide_winner(board)

    results = []
    for state, active_player in states: # Por cada estado almacenado guardo el resultado de la partida para el jugador activo
        if winner == 0: # Si hay empate
            result = 0
        elif active_player == winner:
            result = 1
        else:
            result = -1
        results.append((state, active_player, result))

    return results

def simulate_agent_vs_random(iterations=1000): # Simula partida en la que el agente entrenado juega contra un agente que pilla movimientos random
    board = othello.create_board()
    states = []
    skipped_turns = 0

    current_player = NEGRO
    agent = random.choice([BLANCO,NEGRO])

    while not othello.is_board_full(board) and skipped_turns < 2:

        movs = othello.valid_movements(board, current_player)
        if not movs: # Si no hay movimientos validos, se skipea el turno, no se puede aplicar el algoritmo
            skipped_turns += 1
            current_player = 3 - current_player
            continue
        skipped_turns = 0

        if current_player == agent:
            mov = mcts_uct.mcts_uct(state=board, player=current_player, iterations=iterations, neural=True) # Aplica algoritmo mcts con uct para encontrar movimiento óptimo
        else:
            mov = random.choice(movs)

        othello.apply_movement(board, mov[0], mov[1], current_player)
        actual_state = np.copy(board)
        current_player = 3 - current_player
        states.append((actual_state, current_player))

    winner = othello.decide_winner(board)

    agent_won = 0

    if winner == agent:
        print("Gana el agente")
        agent_won = 1
    elif winner == 0:
        print("Empate")
    else:
        print("Pierde el agente")

    results = []
    for state, active_player in states: # Por cada estado almacenado guardo el resultado de la partida para el jugador activo
        if winner == 0: # Si hay empate
            result = 0
        elif active_player == winner:
            result = 1
        else:
            result = -1
        results.append((state, active_player, result))

    return results, agent_won

def simulate_agent_vs_old(iterations=1000): # Simula partida en la que el agente entrenado juega contra el mismo agente con la política anterior
    board = othello.create_board()
    states = []
    skipped_turns = 0

    current_player = NEGRO
    neural_agent = random.choice([BLANCO,NEGRO])

    while not othello.is_board_full(board) and skipped_turns < 2:

        movs = othello.valid_movements(board, current_player)
        if not movs: # Si no hay movimientos validos, se skipea el turno, no se puede aplicar el algoritmo
            skipped_turns += 1
            current_player = 3 - current_player
            continue
        skipped_turns = 0

        if current_player == neural_agent:
            mov = mcts_uct.mcts_uct(state=board, player=current_player, iterations=iterations, neural=True) # Aplica algoritmo mcts con red neuronal como política
        else:
            mov = mcts_uct.mcts_uct(state=board, player=current_player, iterations=iterations, neural=False)

        othello.apply_movement(board, mov[0], mov[1], current_player)
        actual_state = np.copy(board)
        current_player = 3 - current_player
        states.append((actual_state, current_player))

    winner = othello.decide_winner(board)

    agent_won = 0

    if winner == neural_agent:
        print("Gana el agente con red neuronal")
        agent_won = 1
    elif winner == 0:
        print("Empate")
    else:
        print("Pierde el agente con red neuronal")

    results = []
    for state, active_player in states: # Por cada estado almacenado guardo el resultado de la partida para el jugador activo
        if winner == 0: # Si hay empate
            result = 0
        elif active_player == winner:
            result = 1
        else:
            result = -1
        results.append((state, active_player, result))

    return results, agent_won

def generate_data_parallel(simulation_function, num_games=500, iterations=1000, processes=4): #Simulación de varias partidas paralelamente, para reducir tiempo de espera
    args = [(iterations,) for _ in range(num_games)] # Creamos una lista de argumentos, uno por cada juego. El argumento siempre es el mismo, las iteraciones
    with Pool(processes=processes) as pool: # Pool de procesos, cada uno simulando una partida. Al salir del bloque with Python se encarga de liberar recursos
        results = pool.starmap(simulation_function, args) # Cada simulate_game recibe un argumento de la lista, que es el mismo realmente
    data = []
    if(simulation_function == simulate_agent_vs_old or simulation_function == simulate_agent_vs_random):
        victories = 0 # Contador de victorias del agente, sea el normal o el que usa la neurona dependiendo del caso
        for game_data, agent_won in results: 
            data.extend(game_data)  # Unimos todos los datos de las diferentes partidas simuladas en una única lista
            victories += agent_won # Sumo al contador de victorias para estadisticas
        return data, victories
    else:
        for game_data in results: 
            data.extend(game_data)  # Unimos todos los datos de las diferentes partidas simuladas en una única lista
        return data

if __name__ == "__main__":
    processes = os.cpu_count() - 1 # Usa todos los núcleos disponibles menos uno, para no saturar

    while True:
        try:
            num_games, iterations = map(int,input("Introduce el número de partidas a simular y el número de iteraciones del algoritmo mcts_uct (separados por un espacio): ").split())
            break
        except:
            print("Entrada inválida.")
    
    while True:
        try:
            choice = int(input("Escoge entre simular datos con el agente jugando contra sí mismo (1), el agente jugando contra un oponente que escoge movimientos aleatorios (2) o el agente con red neuronal contra uno sin red (3): "))
            if choice in [1,2,3]:
                simulation_function = simulate_agent_vs_agent if choice == 1 else simulate_agent_vs_random if choice == 2 else simulate_agent_vs_old
                break
            else:
                print("Introduce un número entre 1 y 3")
        except:
            print("Entrada inválida.")

    while True:
        try:
            mode = int(input("Introduce 1 si quieres sobreescribir cualquier dato antiguo o 2 si quieres extender el conjunto de datos: "))
            if mode in [1,2]:
                break
            else:
                print("Introduce un número entre 1 y 3")
        except:
            print("Entrada inválida.")
    
    if(simulation_function == simulate_agent_vs_old or simulation_function == simulate_agent_vs_random):
        data, victories = generate_data_parallel(simulation_function,num_games=num_games, iterations=iterations, processes=processes)
    else:
        data = generate_data_parallel(simulation_function,num_games=num_games, iterations=iterations, processes=processes)

    base_route = os.path.dirname(os.path.abspath(__file__)) # Obtiene la ruta absoluta del script
    data_route = os.path.join(base_route, "..", "data") # Obtiene la ruta donde guardar el fichero a partir de la anterior
    os.makedirs(data_route, exist_ok=True)
    file_route = os.path.join(data_route, "training_data.pkl")

    df = pd.DataFrame(data, columns=['state', 'player', 'result'])
    if os.path.exists(file_route) and mode == 2: # Si ya existe el fichero y estamos no estamos en modo sobreescritura, concatenamos datos
        old_df = pd.read_pickle(file_route)
        final_df = pd.concat([old_df, df], ignore_index=True)
    else:
        final_df = df
    
    final_df.to_pickle(file_route)

    # Mostramos estadísticas para análisis en caso de simular partidas del agente vs random o agente con red neuronal vs agente con politica antigua
    if(simulation_function == simulate_agent_vs_random):
        print(f"Victorias del agente: {victories} de {num_games} partidas jugadas")
        print(f"Porcentaje de victorias: {(victories / num_games) * 100}%")
    elif(simulation_function == simulate_agent_vs_old):
        print(f"Victorias del agente con red neuronal: {victories} de {num_games} partidas jugadas")
        print(f"Porcentaje de victorias: {(victories / num_games) * 100}%")

    print(f"Guardados {len(data)} ejemplos para entrenamiento.")