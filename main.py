from utils.load_data import load_data
from utils.basic_preprocess import basic_preprocess
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import RandomizedSearchCV  
from sklearn.svm import SVC
import numpy as np

COMBINATIONS_N = 250

[X_data, Y_data] = basic_preprocess(*load_data())
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.25, shuffle=True, random_state=42)

<<<<<<< HEAD

param_grid = {  
    'n_estimators': [10, 50, 100, 200, 500],  # Número de árboles en el bosque  
    'max_depth': [None, 10, 20, 30, 40, 50],  # Profundidad máxima del árbol  
    'min_samples_split': [2, 5, 10, 20],      # Número mínimo de muestras necesarias para dividir un nodo  
    'min_samples_leaf': [1, 2, 4, 10],        # Número mínimo de muestras necesarias en un nodo hoja  
    'max_features': ['auto', 'sqrt', 'log2'], # Número de características a considerar en cada división  
    'bootstrap': [True, False],                # Si se utiliza el muestreo con reemplazo al construir árboles  
    'criterion': ['gini', 'entropy'],          # Función de evaluación de la calidad de la división  
    'class_weight': [None, 'balanced'],        # Pesos asociados con clases (en caso de clasificación desequilibrada)  
}
random_search = RandomizedSearchCV(estimator=RandomForestClassifier(),   
                                    param_distributions=param_grid,   
                                    n_iter=COMBINATIONS_N,               # Número de combinaciones aleatorias a probar  
                                    cv=5,                     # Validación cruzada de 5 pliegues  
                                    verbose=1,               # Para imprimir información durante la búsqueda  
                                    random_state=42,         # Para reproducibilidad  
                                    n_jobs=-1)               # Usar todos los núcleos disponibles
random_search.fit(X_train, Y_train)
rf_alg = RandomForestClassifier(**random_search.best_params_)
rf_alg.fit(X_train, Y_train)
joblib.dump(rf_alg, "random_forest.joblib")

print("Random forest entrenado !!!")



=======
rf_model = joblib.load("./models/random_forest.joblib")

print(rf_model.get_params())
>>>>>>> dc9b0c8 (Modificacion de readme)

param_dist = {  
    'C': np.logspace(-3, 3, 7),                 # Regularización (C) en una escala logarítmica  
    'kernel': ['linear', 'rbf', 'poly'],         # Tipos de kernel: lineal, RBF y polinómico  
    'degree': [2, 3, 4],                        # Grado del polinomio (solo para kernel polinómico)  
    'gamma': ['scale', 'auto']                  # Parametro gamma para RBF y polinómico  
}  
random_search = RandomizedSearchCV(estimator=SVC()  ,   
                                   param_distributions=param_dist,   
                                   n_iter=COMBINATIONS_N,               # Número de combinaciones aleatorias a probar  
                                   cv=5,                     # Validación cruzada de 5 pliegues  
                                   verbose=1,               # Para imprimir información durante la búsqueda  
                                   random_state=42,         # Para reproducibilidad  
                                   n_jobs=-1)               # Usar todos los núcleos disponibles  
random_search.fit(X_train, Y_train)
svc_alg = SVC(**random_search.best_params_)
svc_alg.fit(X_train, Y_train)
joblib.dump(svc_alg, "svc.joblib")
