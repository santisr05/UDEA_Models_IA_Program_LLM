import pandas as pd
import numpy as np

def generar_caso_de_uso_filtrar_outliers_iqr():
    n_samples = np.random.randint(10, 20)
    # Generamos datos normales y añadimos un par de outliers extremos
    precios = np.random.normal(500, 50, n_samples)
    precios = np.append(precios, [10, 2000]) # Outliers garantizados
    np.random.shuffle(precios)
    
    df = pd.DataFrame({'precio': precios})
    factor = 1.5
    
    # Lógica de cálculo para el output
    q1 = df['precio'].quantile(0.25)
    q3 = df['precio'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    expected_output = df[(df['precio'] >= lower) & (df['precio'] <= upper)]['precio'].values
    
    return {"df": df, "factor": factor}, expected_output