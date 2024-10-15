import argparse
import logging

import pandas as pd
import wandb

logger = logging.getLogger(__name__)


def cleaning_data(input_path: str, output_path: str) -> None:
    """
    Cleaning dataset before EDA.

    Parameters:
        input_path(str): Path to data file under csv format.
        output_path(str): Path to save data after cleaning.

    Returns:
        None: return None
    """

    run = wandb.init(project="mlops for bank churn", job_type="cleaning")

    assert input_path.endswith("csv"), "We only support csv files for now"
    try:
        dataset = pd.read_csv(input_path)
    except FileNotFoundError:
        logger.error(FileNotFoundError(f"No such file or directory: {input_path}."))
        return

    logger.info("Start cleaning dataset ...")
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
    dataset = dataset[features]
    logger.info("Cleaning dataset done!!!")

    logger.info("Start saving data to local and wandb ...")
    dataset.to_csv(output_path, index=False)
    artifact = wandb.Artifact(
        name="cleaned_data",
        type="dataset",
        description="Cleaned data ready for EDA stage",
    )
    artifact.add_file(output_path)
    run.log_artifact(artifact)
    logger.info("Saving done!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic cleaning data")

    parser.add_argument(
        "--input_path",
        type=str,
        default="data/train.csv",
        required=True,
        help="Path to input path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/cleaned_train.csv",
        required=True,
        help="Path to output path",
    )

    args = parser.parse_args()

    cleaning_data(args.input_path, args.output_path)
