import numpy as np

# Tablero
BOARD_SIZE = 8

# Estados de casilla
EMPTY = 0
WHITE = 1
BLACK = 2

# Direcciones de exploración (8 vecinos)
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def initialize_board():
    """Crea y devuelve un tablero 8x8 con la posición inicial.

    Devuelve un `np.ndarray` de enteros con los estados `EMPTY`, `WHITE`, `BLACK`.
    """
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    # Posición inicial en el centro
    board[3, 3], board[4, 4] = WHITE, WHITE
    board[3, 4], board[4, 3] = BLACK, BLACK
    return board


def display_board(board):
    """Imprime el tablero en consola con símbolos para cada estado."""
    symbols = {EMPTY: '.', WHITE: 'B', BLACK: 'N'}
    header = "  " + " ".join(map(str, range(BOARD_SIZE)))
    print(header)
    for i in range(BOARD_SIZE):
        row = [symbols[int(board[i, j])] for j in range(BOARD_SIZE)]
        print(f"{i} " + " ".join(row))


def is_inside(x, y):
    """Devuelve True si (x, y) está dentro del tablero."""
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def legal_moves(board, player):
    """Devuelve la lista de movimientos válidos (tuplas (x, y)) para `player`."""
    moves = []
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if board[x, y] != EMPTY:
                continue
            if captured_discs(board, x, y, player):
                moves.append((x, y))
    return moves


def apply_move(board, x, y, player):
    """Coloca la pieza del `player` en (x, y) y voltea las piezas capturadas."""
    board[x, y] = player
    for fx, fy in captured_discs(board, x, y, player):
        board[fx, fy] = player


def captured_discs(board, x, y, player):
    """Calcula y devuelve la lista de casillas del oponente que quedarían capturadas.

    Recorre las 8 direcciones y acumula fichas enemigas que quedan flanqueadas.
    """
    opponent = 3 - player
    captured = []

    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        path = []
        while is_inside(nx, ny) and board[nx, ny] == opponent:
            path.append((nx, ny))
            nx += dx
            ny += dy
        if path and is_inside(nx, ny) and board[nx, ny] == player:
            captured.extend(path)

    return captured


def count_pieces(board):
    """Devuelve una tupla (num_blancas, num_negras)."""
    return np.count_nonzero(board == WHITE), np.count_nonzero(board == BLACK)


def board_full(board):
    """True si no hay casillas vacías."""
    return np.all(board != EMPTY)


def game_finished(board):
    """True si el juego ha terminado: tablero lleno o ninguno tiene movimientos."""
    if board_full(board):
        return True
    return len(legal_moves(board, WHITE)) == 0 and len(legal_moves(board, BLACK)) == 0


def evaluate_winner(board):
    """Cuenta piezas, muestra el recuento y devuelve el ganador.

    Devuelve `WHITE`, `BLACK` o 0 en caso de empate.
    """
    num_white, num_black = count_pieces(board)
    print(f"Fichas blancas: {num_white}, fichas negras: {num_black}")
    if num_white == num_black:
        return 0
    return WHITE if num_white > num_black else BLACK


def winner_without_print(board):
    """Como `evaluate_winner` pero sin imprimir el recuento (útil para simulaciones)."""
    num_white, num_black = count_pieces(board)
    if num_white == num_black:
        return 0
    return WHITE if num_white > num_black else BLACK


# Wrappers para mantener compatibilidad con código existente
# (se mantienen los nombres originales mientras internamente usamos los nuevos nombres).
create_board = initialize_board
show_board = display_board
inside_board = is_inside
valid_movements = legal_moves
apply_movement = apply_move
get_captured_discs = captured_discs
count_discs = count_pieces
is_board_full = board_full
is_game_finished = game_finished
decide_winner = evaluate_winner
get_winner = winner_without_print