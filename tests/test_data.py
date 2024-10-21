from churn_prediction.data.dataset import read_dataset, cleaning_data
import pytest
import logging
import pandas as pd

logger = logging.getLogger()

@pytest.fixture
def sample_data():
    sample = pd.DataFrame({
        "Customer ID": ['0001', '0002', '0010', '0100'],
        "CreditScore": [500, 600, 800, 700],
        "Geography": ['France', 'Spain', 'Germany', 'Spain'],
        "Gender": ['Male', 'Male', 'Female', 'Male'],
        "Age": [20, 30, 40, 35],
        "Tenure": [2, 4, 10, 1],
        "Balance": [0, 50000, 25000, 100000],
        "NumOfProducts": [2, 1, 3, 4],
        "HasCrCard": [0, 1, 1, 0],
        "IsActiveMember": [1, 0, 0, 1],
        "EstimatedSalary": [25000, 50000, 30000, 70000],
        "Exited": [1, 1, 1, 0]
    })

    return sample

def test_read_dataset():
    with pytest.raises(AssertionError, match="We only support csv files for now"):
        read_dataset("sample.xlsx", logger)

    with pytest.raises(FileNotFoundError):
        read_dataset("sample.csv", logger)

def test_cleaning_data(sample_data):
    cleaned_data = cleaning_data(sample_data)

    assert len(cleaned_data.columns) == 11
    assert set(cleaned_data.columns) == set([
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
            "Exited"
            ])
    
def test_process_data(sample_data):
    pass