from utils.load_data import load_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from utils.precision import precision
from utils.basic_preprocess import basic_preprocess

[X_data, Y_data] = basic_preprocess(*load_data())

X_train, X_test, Y_train, Y_test = train_test_split(X_data.copy(), Y_data.copy(), test_size=0.25, shuffle=True, random_state=42)


print("Random forest \n")
alg = RandomForestClassifier(n_estimators=20,min_samples_split=300)
alg.fit(X_train, Y_train)
precision(Y_train, alg.predict(X_train), label="train")
precision(Y_test, alg.predict(X_test), label="test")

print("\n")

print("Regresion logistica \n")
alg = LogisticRegression(solver="newton-cholesky")
alg.fit(X_train, Y_train)
precision(Y_train, alg.predict(X_train), label="train")
precision(Y_test, alg.predict(X_test), label="test")

print("\n")
