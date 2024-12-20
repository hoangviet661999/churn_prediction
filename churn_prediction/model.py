from pathlib import Path

import joblib
import numpy as np
import wandb
from sklearn.metrics import f1_score, precision_score, recall_score


def train_model(model, X: np.array, y: np.array):
    """
    Trains a machine learning model and returns it.

        Parameters:
            X_train (np.array): Training data.
            y_train (np.array): Labels.

        Returns:
            model: Trained machine learning model.
    """
    model = model.fit(X, y)
    return model


def eval_model(model, X: np.array, y: np.array) -> tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall, and F1.

        Parameters:
            y (np.array): Known labels, binarized.
            preds (np.array): Predicted labels, binarized.

        Returns:
            precision : float
            recall : float
            f1 : float
    """

    y_pred = model.predict(X)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    return precision, recall, f1


def save_model_locally(model, path: str | Path) -> None:
    """
    Save the trained model to a file.

    Parameters:
        model : Trained machine learning model.
        path (str or Path): Path to save the model.
    """

    joblib.dump(model, path)


def save_model_wandb(path: str | Path, run) -> None:
    """
    Save the trained model to wandb.

    Parameters:
        path (str or Path) : path of model needed to save.
        run: wandb run
    """
    artifact = wandb.Artifact(
        name="rf_model",
        type="model",
        description="Random Forest model for Churn Prediction",
    )
    artifact.add_file(path)
    run.log_artifact(artifact)


def load_model(path: str | Path):
    """
    Load a trained model from a file.

        Parameters:
            path(str or Path): Path to the model.

        Returns:
            model: Trained machine learning model.
    """

    model = joblib.load(path)
    return model


def inference(model, X: np.array) -> int:
    """
    Prediction churn_prediction.

        Parameters:
            model: machine learning model to make prediction.
            X (pd.Dataframe): user need to be predicted.

        Returns:
            y(int) : Churn or not
    """
    y = model.predict(X)
    return y
