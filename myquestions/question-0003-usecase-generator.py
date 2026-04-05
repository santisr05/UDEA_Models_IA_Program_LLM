import numpy as np
from sklearn.metrics import f1_score

def generar_caso_de_uso_calcular_f1_ponderado():
    n_samples = np.random.randint(20, 50)
    # Generamos etiquetas para 3 clases (0, 1, 2)
    y_true = np.random.randint(0, 3, n_samples)
    # Simulamos predicciones con algo de ruido/error
    y_pred = y_true.copy()
    mask = np.random.random(n_samples) < 0.3 # 30% de error aleatorio
    y_pred[mask] = np.random.randint(0, 3, mask.sum())
    
    # Lógica de cálculo para el output
    expected_output = float(f1_score(y_true, y_pred, average='weighted'))
    
    return {"y_true": y_true, "y_pred": y_pred}, expected_output