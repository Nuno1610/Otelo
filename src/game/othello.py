import numpy as np

# Constantes para representar el estado de cada casilla en el tablero
EMPTY = 0   # Casilla vacía
WHITE = 1   # Ficha blanca
BLACK = 2   # Ficha negra

# Direcciones para explorar alrededor de una casilla (8 direcciones: diagonales, verticales y horizontales)
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def create_board(): # Inicializa tablero con las 4 fichas iniciales en el centro
    board = np.zeros((8, 8), dtype=int)
    board[3, 3], board[4, 4] = WHITE, WHITE
    board[3, 4], board[4, 3] = BLACK, BLACK
    return board

def show_board(board): # Muestra el tablero en consola usando símbolos para representar fichas y casillas vacías
    symbols = ['.', 'B', 'N']  # Otra opción: ●: Negro | ○: Blanco | .: Libre
    print("  " + " ".join(map(str, range(8))))  # Encabezado de las columnas con números 0-7
    for i in range(8):
        row = [symbols[board[i, j]] for j in range(8)]  # Mapea números a símbolos para cada fila
        print(f"{i} " + " ".join(row))  # Muestra el número de fila y los símbolos correspondientes

def inside_board(x, y): # Verifica que las coordenadas (x, y) estén dentro de los límites del tablero 8x8
    return 0 <= x < 8 and 0 <= y < 8

def valid_movements(board, player):
    movs = []
    for x in range(8):
        for y in range(8):
            if board[x, y] != EMPTY:
                continue  # Sólo se puede mover hacia casillas vacías
            if get_captured_discs(board, x, y, player):  # Si hay fichas que capturar al mover ahí
                movs.append((x, y))  # Añade esa casilla a movimientos válidos
    return movs

def apply_movement(board, x, y, player): # Aplica el movimiento del jugador hacia la posición (x, y) y cambia las fichas capturadas
    board[x, y] = player
    for fx, fy in get_captured_discs(board, x, y, player):
        board[fx, fy] = player  # Cambia fichas atrapadas del oponente a las del jugador

def get_captured_discs(board, x, y, player): # Devuelve la lista de fichas del oponente que serían capturadas si el jugador coloca ficha en (x, y)
    opponent = 3 - player  # Si jugador es blanco(1), oponente es negro(2) y viceversa
    captured = []

    for dx, dy in DIRECTIONS:  # Para cada dirección alrededor de la casilla
        nx, ny = x + dx, y + dy
        path = []  # Lista temporal para guardar fichas capturables en esta dirección
        while inside_board(nx, ny) and board[nx, ny] == opponent:
            path.append((nx, ny))  # Añade ficha del oponente al camino
            nx += dx
            ny += dy
        # Si tras una o más fichas del oponente, encontramos una ficha del jugador, se capturan las del path
        if path and inside_board(nx, ny) and board[nx, ny] == player:
            captured.extend(path)

    return captured

def count_discs(board):
    return np.count_nonzero(board == WHITE), np.count_nonzero(board == BLACK)

def is_board_full(board):
    return np.all(board != 0)  # Si hay alguna casilla con 0, significa que está libre

def is_game_finished(board):
    if is_board_full(board):  # Termina si el tablero está lleno
        return True  # Evita calcular movimientos válidos si no quedan casillas libres
    return len(valid_movements(board, 1)) == 0 and len(valid_movements(board, 2)) == 0 # Termina si ninguno de los dos jugadores tiene movimientos válidos

def decide_winner(board): # Decide el ganador contando las fichas y mostrando el resultado por pantalla
    num_white, num_black = count_discs(board)  # Gana el jugador con más fichas
    print(f"Fichas blancas: {num_white}, fichas negras: {num_black}")
    if num_white == num_black:
        return 0  # Empate
    return WHITE if num_white > num_black else BLACK

def get_winner(board): # Igual que decide_winner pero sin imprimir, útil para las simulaciones
    num_white, num_black = count_discs(board)
    if num_white == num_black:
        return 0
    return WHITE if num_white > num_black else BLACK