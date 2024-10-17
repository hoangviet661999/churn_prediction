import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def read_dataset(data_dir: str, logger: logging.Logger) -> pd.DataFrame | None:
    try:
        data = pd.read_csv(data_dir)
    except FileNotFoundError:
        logger.error(FileNotFoundError(f"No such file or directory: {data_dir}."))
        return None
    if data.shape[1] != 11:
        raise ValueError(
            f"Expected data has shape (None, 11) but got shape(None, {data.shape[1]})"
        )

    return data


def process_data(
    X: pd.DataFrame, training: bool = True, data_pipeline: Pipeline = None
) -> None:
    """
    Processing data to feed in Machine Learning model.

    Parameters:
        X (pd.Dataframe): Features in data needed to transform
        training (bool): Indicator if training mode or validation/inference mode
        data_pipeline (sklearn.pipeline.Pipeline):
            Pipeline to transform data consists of StandardScaler for numeric features
            and OneHotEncoder for categorical features, only used if training=false

    Returns:
        X (np.array): Processed data
        data_pipeline (sklearn.pipeline.Pipeline):
            Data pipeline if training is true else passed through.
    """

    numeric_features = list(X.select_dtypes(include=["int", "float"]).columns)
    discrete_features = ["Tenure", "NumOfProducts", "HasCrCard", "IsActiveMember"]
    continuous_features = list(set(numeric_features) - set(discrete_features))
    categorical_features = list(X.select_dtypes(include=["object"]).columns)

    if training:
        numeric_transformer = Pipeline([("scaler", StandardScaler())])

        categorical_transformer = Pipeline(
            [("onehot", OneHotEncoder(handle_unknown="error"))]
        )

        data_prep = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, continuous_features),
                ("categorical", categorical_transformer, categorical_features),
            ],
            remainder="passthrough",
        )

        data_pipeline = Pipeline([("data_prep", data_prep)])
        X = data_pipeline.fit_transform(X)

    else:
        X = data_pipeline.transform(X)

    return X, data_pipeline
