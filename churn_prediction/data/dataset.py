import logging

import pandas as pd
import wandb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def read_dataset(data_dir: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Read dataset from csv file.

    Parameters:
        data_dir(str): csv file path
        logger(logging.Logger): logging process

    Returns:
        data(pd.Dataframe): readable dataset from path
    """
    assert data_dir.endswith("csv"), "We only support csv files for now"
    try:
        data = pd.read_csv(data_dir)
    except FileNotFoundError as e:
        logger.error(FileNotFoundError(f"No such file or directory: {data_dir}."))
        raise e

    return data


def process_data(
    X: pd.DataFrame, training: bool = True, data_pipeline: Pipeline = None
) -> tuple[pd.DataFrame, Pipeline]:
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


def cleaning_data(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning dataset before EDA.

    Parameters:
        dataset(pd.DataFrame): data needed to clean

    Returns:
        cleaned_dataset(pd.Dataframe): cleaned dataset
    """

    features = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cleaned_dataset = dataset[features]

    return cleaned_dataset


def save_data_locally(dataset: pd.DataFrame, output_path: str) -> None:
    """
    Save the dataset to a file.

    Parameters:
        dataset : dataset need to save.
        path (str or Path): Path to save the dataset.
    """
    dataset.to_csv(output_path, index=False)


def save_data_wandb(output_path: str, run) -> None:
    """
    Save the dataset to wandb

    Parameters:
        output_path(str): path of dataset need to save.
        run: wandb run.
    """
    artifact = wandb.Artifact(
        name="cleaned_data",
        type="dataset",
        description="Cleaned data ready for EDA stage",
    )
    artifact.add_file(output_path)
    run.log_artifact(artifact)
