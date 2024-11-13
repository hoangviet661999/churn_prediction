from typing import Annotated, Literal

import joblib
import pandas as pd
import uvicorn
from churn_prediction.data.dataset import process_data
from churn_prediction.model import inference, load_model
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)


class Feature_Needed(BaseModel):
    CreditScore: int
    Geography: Literal["France", "Spain", "Germany"]
    Gender: Literal["Male", "Female"]
    Age: int
    Tenure: int = Field(ge=0, le=10)
    Balance: float
    NumOfProducts: int = Field(ge=1, le=4)
    HasCrCard: int = Field(ge=0, le=1)
    IsActiveMember: int = Field(ge=0, le=1)
    EstimatedSalary: float


@app.get("/")
def read_root():
    """Root endpoint."""
    return "Welcome"


@app.post("/refer")
async def make_prediction(item: Annotated[Feature_Needed, Form()]):
    data = pd.DataFrame(
        {
            "CreditScore": [item.CreditScore],
            "Geography": [item.Geography],
            "Gender": [item.Gender],
            "Age": [item.Age],
            "Tenure": [item.Tenure],
            "Balance": [item.Balance],
            "NumOfProducts": [item.NumOfProducts],
            "HasCrCard": [item.HasCrCard],
            "IsActiveMember": [item.IsActiveMember],
            "EstimatedSalary": [item.EstimatedSalary],
        }
    )
    data_pipeline = joblib.load("outputs/2024-11-05/15-17-16/data_pipeline.pkl")
    model = load_model("outputs/2024-11-05/15-17-16/rf.pth")

    data, _ = process_data(data, training=False, data_pipeline=data_pipeline)
    y = inference(model, data)

    if y == 0:
        return "Non-exited"
    return "Exited"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
