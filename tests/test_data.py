from churn_prediction.data.dataset import read_dataset
import pytest
import logging

logger = logging.getLogger()

def test_read_dataset():
    with pytest.raises(AssertionError, match="We only support csv files for now"):
        read_dataset("sample.xlsx", logger)

    with pytest.raises(FileNotFoundError):
        read_dataset("sample.csv", logger)
