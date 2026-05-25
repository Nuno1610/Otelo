import os
import pickle
import sys

import numpy.core.numeric as numpy_numeric

sys.modules.setdefault("numpy._core.numeric", numpy_numeric)

if __name__ == "__main__":
    
    base_route = os.path.dirname(os.path.abspath(__file__)) # Obtener la ruta absoluta del script

    data_route = os.path.join(base_route, "training_data.pkl")

    # Abro el fichero con los datos y lo cargo
    with open(data_route, 'rb') as f:
        datos = pickle.load(f)

    print(datos)