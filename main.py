from utils.load_data import load_data
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from preprocess.nan_fixer import FillNanCatColWithMode
from preprocess.encoding import CustomOneHotEncoding
from preprocess.scaler import CustomScaler

[X_data, Y_data] = load_data()

basic_transform_pipeline = Pipeline([
    ("fill_cat_nan", FillNanCatColWithMode()),
    ("encoding", CustomOneHotEncoding()),
    ("scaler", CustomScaler(X_data.select_dtypes(exclude=["object"]).columns.tolist()))
])

X_data= basic_transform_pipeline.fit_transform(X_data)

X_train, X_test, Y_train, Y_test = train_test_split(X_data.copy(), Y_data.copy(), test_size=0.25, shuffle=True, random_state=42)
