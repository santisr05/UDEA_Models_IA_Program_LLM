import pandas as pd
import numpy as np

def generar_caso_de_uso_encode_frecuencia():
    ciudades_pool = ['Medellín', 'Bogotá', 'Cali', 'Barranquilla']
    n_samples = np.random.randint(50, 100)
    # Generamos una distribución aleatoria de ciudades
    data = np.random.choice(ciudades_pool, size=n_samples)
    df = pd.DataFrame({'ciudad': data})
    
    # Lógica de cálculo para el output
    counts = df['ciudad'].value_counts(normalize=True)
    expected_output = df['ciudad'].map(counts).values
    
    return {"df": df}, expected_output