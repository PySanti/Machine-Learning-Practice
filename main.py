from utils.load_data import load_data
from sklearn.linear_model import LogisticRegression
from utils.basic_preprocess import basic_preprocess
from sklearn.linear_model import LogisticRegression  
import joblib
from sklearn.model_selection import train_test_split
from utils.precision import precision

[X_data, Y_data] = basic_preprocess(*load_data())
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.25, shuffle=True, random_state=42)

alg = joblib.load("regresion_logistica.joblib")
precision(Y_test, alg.predict(X_test), "test")

