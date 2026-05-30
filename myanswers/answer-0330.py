import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

def select_and_evaluate(df, target_col, k=3):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Seleccionar las k mejores características
    selector = SelectKBest(f_regression, k=k)
    selector.fit(X, y)
    best_features = list(X.columns[selector.get_support()])
    
    # Pipeline con Ridge
    pipeline = Pipeline([
        ('selector', SelectKBest(f_regression, k=k)),
        ('model', Ridge())
    ])
    
    # Validación cruzada 5 folds
    cv_results = cross_validate(pipeline, X, y, cv=5,
                                scoring=['neg_root_mean_squared_error', 'r2'])
    
    return {
        'best_features': best_features,
        'cv_rmse': float(-cv_results['test_neg_root_mean_squared_error'].mean()),
        'cv_r2': float(cv_results['test_r2'].mean())
    }