import logging

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from churn_prediction.dataset import process_data
from churn_prediction.model import eval_model, train_model

logger = logging.getLogger(__name__)


@hydra.main(config_path="./configs", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    logger.info(f"Configuration: \n {OmegaConf.to_yaml(cfg)}")

    dataset_params = cfg["Dataset"]
    assert dataset_params.data_dir.endswith("csv"), "We only support csv files for now"
    try:
        dataset = pd.read_csv(dataset_params.data_dir)
    except FileNotFoundError:
        logger.error(
            FileNotFoundError(f"No such file or directory: {dataset_params.data_dir}.")
        )
        return

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
    ]
    X = dataset[features]
    y = np.array(dataset["Exited"])
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    X_train, data_pipeline = process_data(X_train, training=True)
    X_val, _ = process_data(X_val, training=False, data_pipeline=data_pipeline)
    logger.info(f"Train dataloader has shape {X_train.shape}")
    logger.info(f"Validation dataloader has shape {X_val.shape}")

    logreg_params = cfg["LogisticRegression"]
    logreg = hydra.utils.instantiate(logreg_params)

    logger.info("Start training...")
    logreg = train_model(logreg, X_train, y_train)
    logger.info("Training done!!!")

    precision, recall, f1 = eval_model(logreg, X_val, y_val)
    logger.info(f"\n{precision =}\n{recall =}\n{f1 =}")


if __name__ == "__main__":
    train()
