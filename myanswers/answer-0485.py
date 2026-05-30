import pandas as pd

def calcular_alertas_creciente(df, umbral):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['station_id', 'timestamp'])
    df['incremento'] = df.groupby('station_id')['water_level'].diff()
    df['alerta'] = df['incremento'] > umbral
    return df