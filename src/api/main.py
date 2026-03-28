"""
API FastAPI — IDNet Fraud Detector
Endpoints : /health, /predict, /model-info, /metrics
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.api.predictor import predictor
from src.api.schemas import (
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictResponse,
)

_file_dependency = File(...)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


# Lifespan : chargement du modèle au démarrage


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Démarrage de l'API — chargement du modèle @champion...")
    try:
        predictor.load()
        logger.info("Modèle chargé avec succès.")
    except Exception as e:
        logger.error(f"Échec du chargement du modèle : {e}")
        # On démarre quand même — /health signalera model_loaded=False
    yield
    logger.info("Arrêt de l'API.")


#        Application

app = FastAPI(
    title="ID Fraud Sentinel",
    description="API de détection de fraude documentaire — EfficientNet-B0",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Statut de l'API et du modèle chargé."""
    info = predictor.get_info() if predictor.is_loaded else {}
    return HealthResponse(
        status="ok" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        model_name=info.get("model_name", "unknown"),
        model_version=info.get("model_version", "unknown"),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Monitoring"])
def model_info():
    """Métadonnées du modèle champion (version, seuil, tags MLflow)."""
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return ModelInfoResponse(**predictor.get_info())


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
def metrics():
    """Métriques de performance du modèle champion (depuis MLflow)."""
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    try:
        return MetricsResponse(**predictor.get_metrics())
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur récupération métriques : {e}"
        ) from e  # noqa: B904


@app.post("/predict", response_model=PredictResponse, tags=["Inférence"])
async def predict(file: UploadFile = _file_dependency):
    """
    Prédit si un document est genuine (0) ou fraud (1).

    - **file** : image JPG ou PNG du document
    - **label** : `genuine` ou `fraud`
    - **fraud_probability** : probabilité de fraude (0–1)
    - **confidence** : confiance sur le label retourné
    - **threshold_used** : seuil optimal du modèle champion
    """
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Format non supporté : {file.content_type}. Acceptés : JPG, PNG",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=422, detail="Fichier vide")

    try:
        result = predictor.predict(image_bytes)
        return PredictResponse(**result)
    except Exception as e:
        logger.error(f"Erreur inférence : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur inférence : {e}") from e
