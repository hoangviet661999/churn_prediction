import argparse
import os
import sys

import wandb
from dataset import cleaning_data, read_dataset, save_data_locally, save_data_wandb
from utils.logging import get_logger

__dir__ = os.path.dirname(__file__)
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))


logger = get_logger()


def make_dataset(input_path, output_path):
    run = wandb.init(project="mlops for bank churn", job_type="cleaning")
    data = read_dataset(input_path, logger)

    logger.info("Start cleaning data...")
    cleaned_data = cleaning_data(data)
    logger.info("Cleaning done!!!")

    save_data_locally(cleaned_data, output_path)

    logger.info("Start saving data to wandb ...")
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

    make_dataset(args.input_path, args.output_path)
