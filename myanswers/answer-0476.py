from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def evaluar_modelo(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    
    return mse