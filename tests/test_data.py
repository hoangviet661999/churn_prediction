from churn_prediction.data.dataset import read_dataset
import pytest
import logging
import pandas as pd

logger = logging.getLogger()

@pytest.fixture
def sample_data():
    sample = pd.DataFrame({
        "CreditScore": [],
        "Geography": ['France', 'Spain', 'Germany'],
        "Gender": [],
        "Age": [],
        "Tenure": [],
        "Balance": [],
        "NumOfProducts": [],
        "HasCrCard": [],
        "IsActiveMember": [],
        "EstimatedSalary": [],
        "Exited": []
    })

    return sample

def test_read_dataset():
    with pytest.raises(AssertionError, match="We only support csv files for now"):
        read_dataset("sample.xlsx", logger)

    with pytest.raises(FileNotFoundError):
        read_dataset("sample.csv", logger)
