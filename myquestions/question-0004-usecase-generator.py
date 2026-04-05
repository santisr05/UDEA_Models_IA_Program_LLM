import numpy as np
from sklearn.feature_selection import VarianceThreshold

def generar_caso_de_uso_seleccionar_sensores():
    n_rows = 10
    n_cols = 5
    # Creamos una matriz donde algunas columnas tengan varianza casi nula
    X = np.random.rand(n_rows, n_cols)
    X[:, 0] = 0.5 # Columna con varianza 0
    X[:, 2] = 0.5 + np.random.normal(0, 0.01, n_rows) # Varianza muy baja
    
    umbral = 0.1
    
    # Lógica de cálculo para el output
    selector = VarianceThreshold(threshold=umbral)
    expected_output = selector.fit_transform(X)
    
    return {"X": X, "umbral": umbral}, expected_output