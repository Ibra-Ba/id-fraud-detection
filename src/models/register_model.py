import os

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()


def register_and_promote(run_id, model_name="IDNet-Fraud-Detector", threshold=None):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)  # type: ignore
    client = MlflowClient()

    # Récupération automatique du threshold si non fourni ---

    if threshold is None:
        print(f"[INFO] Recherche du threshold dans le run {run_id}...")
        run = client.get_run(run_id)

        # 1)  metrics
        threshold_val = run.data.metrics.get("optimal_threshold")

        # 2) fallback → params
        if threshold_val is None:
            print("[WARN] 'optimal_threshold' non trouvé dans metrics → fallback params")
            threshold_str = run.data.params.get("optimal_threshold")
            if threshold_str is not None:
                try:
                    threshold_val = float(threshold_str)
                except ValueError:
                    threshold_val = None

        # 3) fallback → tags
        if threshold_val is None:
            print("[WARN] 'optimal_threshold' non trouvé dans params → fallback tags")
            threshold_str = run.data.tags.get("optimal_threshold")
            if threshold_str is not None:
                try:
                    threshold_val = float(threshold_str)
                except ValueError:
                    threshold_val = None

        # 4) fallback final
        if threshold_val is None:
            print("[WARN] Aucun threshold trouvé → utilisation défaut = 0.5")
            threshold = 0.5
        else:
            threshold = float(threshold_val)

        print(f"[INFO] Threshold utilisé : {threshold}")
    # ---------------------------------------------------------------------

    print(f"--- Début de la publication du modèle : {model_name} ---")

    # 1. Enregistrement initial (Création de la version)
    model_uri = f"runs:/{run_id}/model"
    print(f"[1/3] Enregistrement du run {run_id}...")
    mv = mlflow.register_model(model_uri, model_name)
    version = mv.version

    # 2. Ajout des Tags
    print(f"[2/3] Ajout des tags à la V{version}...")
    client.set_model_version_tag(model_name, version, "optimal_threshold", str(threshold))
    client.set_model_version_tag(model_name, version, "deployment_status", "validated")
    client.set_model_version_tag(model_name, version, "origin", "wsl2_training")

    # 3. Promotion via Alias (Champion)
    print(f"[3/3] Assignation de l'alias '@champion' à la V{version}...")
    try:
        client.set_registered_model_alias(model_name, "champion", version)
        print(f"✅ Succès ! La V{version} est désormais le modèle 'champion'.")
    except Exception as e:
        print(f"⚠️ Note: Promotion partielle effectuée. Détail: {e}")

    print(f"--- Publication terminée (Version {version}) ---")
    return version


if __name__ == "__main__":
    # Remplacer par le vrai RUN_ID après un entraînement
    # Si threshold=None, threshod dans MLflow sera utilisé.
    RUN_ID = "71a95d1c305d455dad1fdbbcb1168db5"
    register_and_promote(RUN_ID, threshold=None)
