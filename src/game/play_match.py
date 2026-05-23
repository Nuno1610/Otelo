from game.othello import create_board, is_board_full, show_board, valid_movements, apply_movement, decide_winner
from agent.mcts_uct import mcts_uct

# Constantes para representar el estado de cada casilla en el tablero
EMPTY = 0   # Casilla vacía
WHITE = 1   # Ficha blanca
BLACK = 2   # Ficha negra

def play(user, iterations, neural=False): # Bucle principal para jugar una partida
    board = create_board()  # Crea el tablero inicial con las 4 fichas en el centro
    print("¡Comienza la partida!")
    
    agent = 3 - user  # El agente es el color opuesto
    current_player = BLACK  # Empieza el jugador negro según las reglas
    skipped_turns = 0
    while not is_board_full(board) and skipped_turns < 2:  # Bucle principal del juego hasta que se acabe
        print("\nTurno de:", "BLANCAS" if current_player == WHITE else "NEGRAS")
        show_board(board)  # Muestra el tablero actual

        valid_moves = valid_movements(board, current_player)

        if not valid_moves:
            print("No hay movimientos válidos. Se pasa el turno.")
            current_player = 3 - current_player
            skipped_turns += 1
            continue
        else:
            skipped_turns = 0 # Reinicia contador si hay movimientos posibles

        if current_player == user:
            # Turno del jugador humano
            print("Movimientos válidos:", valid_moves)
            while True:
                try:
                    x, y = map(int, input("Introduce fila y columna (con un espacio entre ambos): ").split())  # Split por cualquier número de espacios, por si acaso
                    if (x, y) in valid_moves:
                        apply_movement(board, x, y, user)
                        break
                    else:
                        print("Movimiento inválido. Inténtalo de nuevo.")
                except:
                        print("Entrada incorrecta. Inténtalo de nuevo.")
        else: # Turno del agente (elige un movimiento aplicando mcts con uct)
            mov = mcts_uct(board, current_player, iterations = iterations, neural=neural) # La política usada depende del parámetro neural
            print(f"El agente mueve ficha a: {mov[0]} {mov[1]}")
            apply_movement(board, mov[0], mov[1], agent)

        current_player = 3 - current_player  # Cambia de turno al otro jugador

    show_board(board)
    ganador = decide_winner(board)
    if ganador == user:
        print("¡Has ganado!")
    elif ganador == agent:
        print("Derrota")
    else:
        print("¡Empate!")

if __name__ == "__main__":
    while True: # Elección de color por parte del usuario
        try:
            user = int(input("Elige tu color: 1 (Blancas) o 2 (Negras - Empiezas tú): "))
            if user in [WHITE, BLACK]:
                break
            else:
                print("Por favor, introduce 1 o 2.")
        except:
            print("Entrada inválida.")

    while True: # Selección de tipo de agente
        try:
            choice = int(input("Escoge tipo de oponente: Agente sin red neuronal(1) o Agente con red neuronal(2): "))
            if choice in [1, 2]:
                neural = False if choice == 1 else True
                break
            else:
                print("Introduce un número entre 1 y 3.")
        except:
            print("Entrada inválida.")

    while True: # Selección de dificultad de juego (numero de iteraciones del algoritmo MCTS)
        try:
            iterations = int(input("Elige la dificultadad del juego (nº de iteraciones del algoritmo, a más iteraciones, mayor tiempo de espera): "))
            break
        except:
            print("Entrada inválida.")
    play(user, iterations, neural)