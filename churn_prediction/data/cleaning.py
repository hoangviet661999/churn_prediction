import argparse
import logging

import wandb

from .dataset import (cleaning_data, read_dataset, save_data_locally,
                      save_data_wandb)

logger = logging.getLogger(__name__)


def clean(input_path, output_path):
    data = read_dataset(input_path, logger)

    logger.info("Start cleaning data...")
    cleaned_data = cleaning_data(data)
    logger.info("Cleaning done!!!")

    save_data_locally(cleaned_data, output_path)

    logger.info("Start saving data to wandb ...")
    run = wandb.init(project="mlops for bank churn", job_type="cleaning")
    save_data_wandb(output_path, run)
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

    clean(args.input_path, args.output_path)
