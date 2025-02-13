from sklearn.pipeline import Pipeline
from preprocess.nan_fixer import FillNanCatColWithMode
from preprocess.encoding import CustomOneHotEncoding
from imblearn.over_sampling import SMOTE

def basic_preprocess(X_data, Y_data):
    """
        Recibe la data despues de haber sido cargada y aplica
        procedimientos de preprocesamiento y over sampling (SMOTE)
    """
    basic_transform_pipeline = Pipeline([
        ("fill_cat_nan", FillNanCatColWithMode()),
        ("encoding", CustomOneHotEncoding()),
    ])
    X_data= basic_transform_pipeline.fit_transform(X_data)
    X_data, Y_data = SMOTE(random_state=42).fit_resample(X_data, Y_data)
    return [X_data, Y_data]


