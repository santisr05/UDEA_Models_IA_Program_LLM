from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

def select_and_evaluate(df, target_col, k=3):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    selector = SelectKBest(f_regression, k=k)
    selector.fit(X, y)
    best_features = list(X.columns[selector.get_support()])
    
    pipeline = Pipeline([
        ('selector', SelectKBest(f_regression, k=k)),
        ('model', Ridge())
    ])
    
    cv_results = cross_validate(pipeline, X, y, cv=5,
                                scoring=['neg_root_mean_squared_error', 'r2'])
    
    return k