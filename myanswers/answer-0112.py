from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def buscar_mejor_svm(X, y):
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C': [1, 10]
    }
    
    svc = SVC()
    grid_search = GridSearchCV(svc, param_grid, cv=3)
    grid_search.fit(X, y)
    
    return {
        'best_params_': grid_search.best_params_,
        'best_score_': grid_search.best_score_
    }