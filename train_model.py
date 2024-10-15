import logging
import os

import hydra
import pandas as pd
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from churn_prediction.dataset import process_data
from churn_prediction.model import eval_model, save_model, train_model
from churn_prediction.visualize import plot_feature_importances

logger = logging.getLogger(__name__)


@hydra.main(config_path="./configs", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    logger.info(f"Configuration: \n {OmegaConf.to_yaml(cfg)}")

    run = wandb.init(
        project="mlops for bank churn", config=OmegaConf.to_container(cfg, resolve=True)
    )

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
    y = dataset["Exited"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    X_train, data_pipeline = process_data(X_train, training=True)
    X_val, _ = process_data(X_val, training=False, data_pipeline=data_pipeline)
    logger.info(f"Train dataloader has shape {X_train.shape}")
    logger.info(f"Validation dataloader has shape {X_val.shape}")

    if "LogisticRegression" in cfg:
        logreg_params = cfg["LogisticRegression"]
        logreg = hydra.utils.instantiate(logreg_params)

        logger.info("Start training Logistic Regression...")
        logreg = train_model(logreg, X_train, y_train)
        logger.info("Training done!!!")

        precision, recall, f1 = eval_model(logreg, X_val, y_val)
        logger.info(f"Logistic Regression: \n{precision =}\n{recall =}\n{f1 =}")

        wandb.summary["logreg_precision"] = precision
        wandb.summary["logreg_recall"] = recall
        wandb.summary["logreg_f1"] = f1

    if "RandomForest" in cfg:
        rf_params = cfg["RandomForest"]
        rf = hydra.utils.instantiate(rf_params)

        logger.info("Start training Random Forest...")
        rf = train_model(rf, X_train, y_train)
        logger.info("Training done!!!")

        precision, recall, f1 = eval_model(rf, X_val, y_val)
        logger.info(f"Random Forest: \n{precision =}\n{recall =}\n{f1 =}")

        wandb.summary["rf_precision"] = precision
        wandb.summary["rf_recall"] = recall
        wandb.summary["rf_f1"] = f1

        img = plot_feature_importances(data_pipeline, rf)
        wandb.log({"fea_imp_img": wandb.Image(img)})

        save_model(rf, os.path.join(HydraConfig.get().run.dir, "rf.pth"))
        artifact = wandb.Artifact(
            name="rf_model",
            type="model",
            description="Random Forest model for Churn Prediction",
            metadata={"precision": precision, "recall": recall, "f1": f1},
        )
        artifact.add_file(os.path.join(HydraConfig.get().run.dir, "rf.pth"))
        run.log_artifact(artifact)


if __name__ == "__main__":
    train()
