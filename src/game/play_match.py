from game.othello import (
    create_board,
    is_board_full,
    show_board,
    valid_movements,
    apply_movement,
    decide_winner,
)
from agent.mcts_uct import mcts_uct

# Valores para el estado de cada casilla
EMPTY = 0
WHITE = 1
BLACK = 2


def run_match(human_player, mcts_iterations, use_neural=False):
    """Ejecuta una partida entre un jugador humano y el agente MCTS.

    Parámetros:
    - human_player: `WHITE` o `BLACK` indicando el color del humano.
    - mcts_iterations: número de iteraciones que usa el MCTS.
    - use_neural: si True, usa la política basada en red neuronal.
    """
    board = create_board()
    print("¡Comienza la partida!")

    opponent = 3 - human_player
    current = BLACK  # según las reglas, comienza el jugador negro
    skipped_turns = 0

    while not is_board_full(board) and skipped_turns < 2:
        print("\nTurno de:", "BLANCAS" if current == WHITE else "NEGRAS")
        show_board(board)

        moves = valid_movements(board, current)
        if not moves:
            print("No hay movimientos válidos. Se pasa el turno.")
            current = 3 - current
            skipped_turns += 1
            continue

        skipped_turns = 0

        if current == human_player:
            # Turno del humano: solicitar coordenadas hasta recibir un movimiento válido
            print("Movimientos válidos:", moves)
            while True:
                try:
                    raw = input("Introduce fila y columna (separadas por espacio): ")
                    x_str, y_str = raw.strip().split()
                    x, y = int(x_str), int(y_str)
                except ValueError:
                    print("Entrada incorrecta. Introduce dos números separados por espacio.")
                    continue

                if (x, y) in moves:
                    apply_movement(board, x, y, human_player)
                    break
                else:
                    print("Movimiento inválido. Inténtalo de nuevo.")
        else:
            # Turno del agente: elegir movimiento mediante MCTS+UCT
            move = mcts_uct(board, current, iterations=mcts_iterations, neural=use_neural)
            # `move` debe ser una tupla (fila, columna)
            print(f"El agente mueve ficha a: {move[0]} {move[1]}")
            apply_movement(board, move[0], move[1], opponent)

        current = 3 - current

    show_board(board)
    winner = decide_winner(board)
    if winner == human_player:
        print("¡Has ganado!")
    elif winner == opponent:
        print("Derrota")
    else:
        print("¡Empate!")


# Mantener compatibilidad: `play` sigue siendo accesible como alias
play = run_match


if __name__ == "__main__":
    # Elección de color por parte del usuario
    while True:
        try:
            choice = int(input("Elige tu color: 1 (Blancas) o 2 (Negras - Empiezas tú): "))
            if choice in (WHITE, BLACK):
                human = choice
                break
            print("Por favor, introduce 1 o 2.")
        except ValueError:
            print("Entrada inválida.")

    # Selección de tipo de agente
    while True:
        try:
            sel = int(input("Escoge oponente: 1) sin red neuronal  2) con red neuronal: "))
            if sel in (1, 2):
                use_neural_flag = (sel == 2)
                break
            print("Introduce 1 o 2.")
        except ValueError:
            print("Entrada inválida.")

    # Selección de número de iteraciones para MCTS (dificultad)
    while True:
        try:
            iters = int(input("Elige número de iteraciones para MCTS (entero positivo): "))
            break
        except ValueError:
            print("Entrada inválida.")

    run_match(human, iters, use_neural=use_neural_flag)