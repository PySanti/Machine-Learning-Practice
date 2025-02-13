from sklearn.metrics import (f1_score, confusion_matrix)

def precision(Y_test, Y_pred, label):
    """
    Retorna las metricas de precision acordadas
    """
    print("~~~~~~~ {label}")
    print(f"Precision positiva : {f1_score(Y_test, Y_pred)}")
    print(f"Precision negativa : {f1_score(Y_test, Y_pred, pos_label=0)}")
    print(confusion_matrix(Y_test, Y_pred))
