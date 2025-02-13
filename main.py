from utils.load_data import load_data
from sklearn.linear_model import LogisticRegression
from utils.basic_preprocess import basic_preprocess
from sklearn.linear_model import LogisticRegression  
import joblib

"""
    Mejores parámetros para Regresión Logística: {'C': 10, 'max_iter': 500, 'penalty': 'l1', 'solver': 'liblinear', 'tol': 0.0001}
"""

[X_data, Y_data] = basic_preprocess(*load_data())
alg = LogisticRegression(C=10, max_iter=500, penalty="l1", solver="liblinear", tol=0.0001)
alg.fit(X_data, Y_data)
joblib.dump(alg, "regresion_logistica.joblib")
