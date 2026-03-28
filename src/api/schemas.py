"""
Schémas Pydantic pour les requêtes et réponses de l'API.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    optimal_threshold: float
    origin: str
    deployment_status: str


class MetricsResponse(BaseModel):
    auroc: float
    f1: float
    accuracy: float
    optimal_threshold: float


class PredictResponse(BaseModel):
    label: str  # "genuine" ou "fraud"
    fraud_probability: float
    confidence: float
    threshold_used: float
    model_version: str
