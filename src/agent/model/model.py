import os
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def create_model(input_shape=(8, 8, 4)): # Tablero de 8 x 8, con 3 canales representando las situaciones de las casillas, y 1 representando el jugador actual
    model = Sequential() # Red secuencial = pila de capas donde la salida de una es la entrada de la siguiente

    # Usamos relu en las capas ocultas por simplicidad y eficacia: Solo compara con 0, llevando a un entrenamiento más rápido

    # Capas convolucionales
    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu', input_shape=input_shape)) # Capa con 128 filtros 3x3 para detectar patrones espaciales simples.
    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu')) # Capa con 128 filtros 3x3 para patrones más complejos, tomando como base la primera

    # Capas densas
    model.add(Flatten()) # Convierte salida tridimensional del Conv2D a un vector unidimensional
    model.add(Dense(128, activation='relu')) # Capa de 128 neuronas para comenzar a entender el tablero
    model.add(Dense(64, activation='relu')) # Capa de 64 neuronas, que aprende una representación comprimida y significativa del estado del tablero
    model.add(Dense(1, activation='tanh'))  # Capa de salida. Devuelve valor en el rango [-1, 1]

    model.compile(optimizer=Adam(), loss='mse', metrics=['mae']) # Define entrenamiento del modelo, mse para penalizar errores grandes y mae para mostrar resultados
    return model

def convert_board_state(board, player): # Convierte la matriz que representa el tablero para su procesamiento en la red neuronal
    if player == 1: # jugador BLANCO
        player_pieces = (board == 1).astype(np.float32)
        opponent_pieces = (board == 2).astype(np.float32)
        player_flag = 1.0
    else: # jugador NEGRO
        player_pieces = (board == 2).astype(np.float32)
        opponent_pieces = (board == 1).astype(np.float32)
        player_flag = 0.0
    
    empty = (board == 0).astype(np.float32)
    player_channel = np.full(board.shape, player_flag, dtype=np.float32)
    
    # Creación de cuatro canales: 1. Casillas del jugador para el que se calcula la probabilidad de victoria, marcadas a 1 | 2. Casillas del oponente, marcadas a 2 | 3. Casillas vacías | 4. Identificador del jugador, 0 negro 1 blanco
    return np.stack([player_pieces, opponent_pieces, empty, player_channel], axis=-1)

if __name__ == "__main__":
    # Establecimiento de ruta de lectura del fichero
    base_route = os.path.dirname(os.path.abspath(__file__))
    data_route = os.path.join(base_route, "../..", "data")
    file_route = os.path.join(data_route, "training_data.pkl")

    df = pd.read_pickle(file_route)  # nuevo archivo limpio con pandas

    states = df['state'].to_list() # Representación del tablero tras una jugada
    players = df['player'].to_list() # Jugador activo en cada estado
    results = df['result'].to_list() # Resultado de la partida para el jugador activo. Es lo que la red debe predecir según los estados

    board_states = np.array([convert_board_state(tablero, player) for tablero, player in zip(states, players)])
    outcomes = np.array(results).reshape(-1, 1) # Cambia la forma del array para que tenga una columna(1) y tantas filas como sea necesario(-1)

    # División entrenamiento/validación
    attributes_train, attributes_eval, goal_train, goal_eval = train_test_split(board_states, outcomes, test_size=.2, random_state=12) # random_state establece la misma semilla siempre, para asegurar reproducibilidad

    # Crear modelo
    model = create_model()

    # Entrenamiento
    history = model.fit(
        attributes_train, goal_train,
        epochs=100, batch_size=256, # 100 épocas, con minilotes de 256. Cuanda haya mneos datos, mejor usar 50 épocas con minilotes de 128.
    )

    #Evaluación
    mse, mae = model.evaluate(attributes_eval, goal_eval, batch_size=128)
    print(f"Pérdida (MSE) en validación: {mse:.4f}")
    print(f"Error absoluto medio (MAE) en validación: {mae:.4f}")

    # Guardar modelo
    model_route = os.path.join(base_route, "othello_model.keras")
    model.save(model_route)

    print("Entrenamiento terminado y modelo guardado.")

    variance = np.var(goal_eval)
    r2 = 1 - (mse / variance)

    print(f"Coeficiente de determinación: {r2:.4f}")