import os
import pickle

if __name__ == "__main__":
    
    base_route = os.path.dirname(os.path.abspath(__file__)) # Obtener la ruta absoluta del script

    data_route = os.path.join(base_route, "training_data.pkl")

    # Abro el fichero con los datos y lo cargo
    with open(data_route, 'rb') as f:
        datos = pickle.load(f)

    print(datos)