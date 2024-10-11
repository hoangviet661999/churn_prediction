import pandas as pd
from churn_prediction.dataset import process_data
from churn_prediction.model import train_model, eval_model
from sklearn.model_selection import train_test_split
import numpy as np
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from churn_prediction.utils.logging import get_logger

@hydra.main(config_path="./configs", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    print(f"configuration: \n {OmegaConf.to_yaml(cfg)}")

    dataset_params = cfg["Dataset"]
    try:
        dataset = pd.read_csv(dataset_params.data_dir)
    except FileNotFoundError:
        raise FileNotFoundError(f"No such file or directory: {dataset_params.data_dir}, please check the path!!!")
    
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

    logreg_params = cfg["LogisticRegression"]
    logreg = hydra.utils.instantiate(logreg_params)
    logreg = train_model(logreg, X_train, y_train)
    precision, recall, f1 = eval_model(logreg, X_val, y_val)
    print(f"{precision =}\n{recall =}\n{f1 =}")


if __name__ == "__main__":
    train()
