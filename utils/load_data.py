import pandas as pd
import numpy as np
import pandas as pd

def load_data():
    """
        Cargara el dataset
    """
    df = pd.concat([
        pd.read_csv("./data/adult.data", delimiter=","),
        pd.read_csv("./data/adult.test", delimiter=",")
    ], axis=0)

    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace("?", np.nan, inplace=True)
    return [df.drop("income", axis=1), df["income"].map({"<=50K" : 0, ">50K" : 1})] 
