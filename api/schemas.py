"""Pydantic v2 request/response models."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
AddOn = Literal["Yes", "No", "No internet service"]

EXAMPLE = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 5,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 89.5,
    "TotalCharges": 450.2,
}


class Customer(BaseModel):
    """One customer (same columns as the training data)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE})

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: AddOn
    OnlineBackup: AddOn
    DeviceProtection: AddOn
    TechSupport: AddOn
    StreamingTV: AddOn
    StreamingMovies: AddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0, le=500)
    TotalCharges: float | None = Field(default=None, ge=0, le=50000)


class Prediction(BaseModel):
    """Prediction for one customer."""

    churn_probability: float
    churn: bool
    label: Literal["Yes", "No"]


class SinglePrediction(Prediction):
    """Single-predict response (adds model version)."""

    version: str


class BatchRequest(BaseModel):
    """Batch of customers (1 to 1000)."""

    customers: list[Customer] = Field(min_length=1, max_length=1000)


class BatchPrediction(BaseModel):
    """Batch response, same order as the request."""

    version: str
    count: int
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    """Loaded model metadata."""

    version: str
    trained_at: str
    algorithm: str
    threshold: float
    metrics: dict[str, float]
    features: list[str]


class Health(BaseModel):
    """Liveness info."""

    status: str
    model_loaded: bool
