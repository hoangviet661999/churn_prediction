from typing import Annotated, Literal

import uvicorn
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Feature_Needed(BaseModel):
    CreditScore: int
    Geography: Literal["France", "Spain", "Germany"]
    Gender: Literal["Male", "Female"]
    Age: int
    Tenure: int = Field(gt=0, le=10)
    Balance: float
    NumOfProducts: int = Field(gt=1, le=4)
    HasCrCard: int = Field(gt=0, le=1)
    IsActiveMember: int = Field(gt=0, le=1)
    EstimatedSalary: float


@app.post("/refer")
async def inference(item: Annotated[Feature_Needed, Form()]):
    return "OK"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
