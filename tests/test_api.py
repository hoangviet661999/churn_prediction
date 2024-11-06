from app import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_read_root():
    reponse = client.get("/")
    assert reponse.status_code == 200
    assert reponse.json() == "Welcome"


def test_make_prediction():
    form_data = {
        "CreditScore": 0,
        "Geography": "France",
        "Gender": "Male",
        "Age": 10,
        "Tenure": 2,
        "Balance": 0.0,
        "NumOfProducts": 2,
        "HasCrCard": 0,
        "IsActiveMember": 0,
        "EstimatedSalary": 5.0,
    }
    reponse = client.post("/refer", data=form_data)
    assert reponse.status_code == 200
    assert reponse.json() == "Non-exited" or reponse.json() == "Exited"


def test_validation_error():
    form_data = {
        "CreditScore": 0,
        "Geography": "France",
        "Gender": "Male",
        "Age": 10,
        "Tenure": 2,
        "Balance": 0.0,
        "NumOfProducts": 2,
        "HasCrCard": 0,
        "IsActiveMember": 2,
        "EstimatedSalary": 5.0,
    }
    reponse = client.post("/refer", data=form_data)
    assert reponse.status_code == 422
