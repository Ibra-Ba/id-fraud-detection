"""
Chargement du modèle @champion depuis MLflow Registry et inférence.
"""

import io
import logging
import os

import mlflow.pytorch
import numpy as np
import torch
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from PIL import Image

from src.data.dataset import VAL_TF
from src.models.config import DEVICE

load_dotenv()
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "IDNet-Fraud-Detector")
ALIAS = "champion"


class FraudPredictor:
    """Encapsule le modèle champion et sa configuration de seuil."""

    def __init__(self):
        self.model = None
        self.version: str = "unknown"
        self.threshold: float = 0.5
        self.tags: dict = {}
        self._loaded = False

    def load(self):
        """Charge le modèle @champion depuis MLflow Registry au démarrage."""
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise RuntimeError("MLFLOW_TRACKING_URI non défini")

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        # Récupère la version associée à l'alias @champion
        mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        self.version = mv.version
        self.tags = mv.tags or {}
        self.threshold = float(self.tags.get("optimal_threshold", 0.5))

        model_uri = f"models:/{MODEL_NAME}@{ALIAS}"
        logger.info(f"Chargement du modèle {model_uri} (v{self.version})...")
        self.model = mlflow.pytorch.load_model(model_uri, map_location=DEVICE)
        self.model.eval()
        self._loaded = True
        logger.info(f"Modèle v{self.version} chargé (threshold={self.threshold:.4f})")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, image_bytes: bytes) -> dict:
        """
        Prédit genuine/fraud à partir des bytes d'une image.
        Retourne un dict compatible avec PredictResponse.
        """
        if not self._loaded:
            raise RuntimeError("Modèle non chargé")

        # Prétraitement
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(image)
        result = VAL_TF(image=arr)
        tensor = result["image"] if isinstance(result, dict) else result
        tensor = tensor.unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)

        # Inférence
        with torch.no_grad():
            logits = self.model(tensor)  # type: ignore
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        fraud_prob = float(probs[1])
        label = "fraud" if fraud_prob >= self.threshold else "genuine"
        confidence = fraud_prob if label == "fraud" else float(probs[0])

        return {
            "label": label,
            "fraud_probability": round(fraud_prob, 4),
            "confidence": round(confidence, 4),
            "threshold_used": self.threshold,
            "model_version": self.version,
        }

    def get_info(self) -> dict:
        return {
            "model_name": MODEL_NAME,
            "model_version": self.version,
            "optimal_threshold": self.threshold,
            "origin": self.tags.get("origin", "unknown"),
            "deployment_status": self.tags.get("deployment_status", "unknown"),
        }

    def get_metrics(self) -> dict:
        """Retourne les métriques loggées dans MLflow pour cette version."""
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(tracking_uri)  # type: ignore
        client = MlflowClient()

        mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        run = client.get_run(mv.run_id)  # type: ignore
        m = run.data.metrics

        return {
            "auroc": round(m.get("test_auroc", m.get("vl_auroc", 0.0)), 4),
            "f1": round(m.get("test_f1", m.get("vl_f1", 0.0)), 4),
            "accuracy": round(m.get("test_accuracy", m.get("vl_accuracy", 0.0)), 4),
            "optimal_threshold": self.threshold,
        }


# Instance singleton — partagée par toute l'application
predictor = FraudPredictor()
