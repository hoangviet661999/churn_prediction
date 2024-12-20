import logging
import os

import hydra
import wandb
from churn_prediction.data.dataset import process_data, read_dataset
from churn_prediction.model import (
    eval_model,
    save_model_locally,
    save_model_wandb,
    train_model,
)
from churn_prediction.utils.visualize import plot_feature_importances
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@hydra.main(config_path="./configs", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    logger.info(f"Configuration: \n {OmegaConf.to_yaml(cfg)}")

    run = wandb.init(
        project="mlops for bank churn",
        config=OmegaConf.to_container(cfg, resolve=True),
        job_type="train",
    )

    dataset_params = cfg["Dataset"]
    data_dir = run.use_artifact(dataset_params.data_dir, type="dataset").file(
        dataset_params.file_name
    )
    X = read_dataset(data_dir, logger)

    if X.shape[1] != 11:
        raise ValueError(
            f"Expected data has shape (None, 11) but got shape(None, {X.shape[1]})"
        )

    y = X.pop("Exited")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    X_train, data_pipeline = process_data(X_train, training=True)
    X_val, _ = process_data(X_val, training=False, data_pipeline=data_pipeline)
    logger.info(f"Train dataloader has shape {X_train.shape}")
    logger.info(f"Validation dataloader has shape {X_val.shape}")

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

    save_model_locally(rf, os.path.join(HydraConfig.get().run.dir, "rf.pth"))
    logger.info("Start saving model to wandb ...")
    save_model_wandb(os.path.join(HydraConfig.get().run.dir, "rf.pth"), run)
    logger.info("Saving done!!!")


if __name__ == "__main__":
    train()
