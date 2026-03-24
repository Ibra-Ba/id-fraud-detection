import os

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()


def register_and_promote(run_id, model_name="IDNet-Fraud-Detector", threshold=0.9262):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)  # pyright: ignore[reportArgumentType]
    client = MlflowClient()

    print("--- Début de la publication du modèle ---")

    # 1. Enregistrement initial (Création de la version)
    model_uri = f"runs:/{run_id}/model"
    print(f"[1/3] Enregistrement du run {run_id}...")
    mv = mlflow.register_model(model_uri, model_name)
    version = mv.version

    # 2. Ajout des Tags
    print(f"[2/3] Ajout des tags métriques à la V{version}...")
    client.set_model_version_tag(model_name, version, "optimal_threshold", str(threshold))
    client.set_model_version_tag(model_name, version, "deployment_status", "validated")
    # taguer l'auteur ou l'environnement
    client.set_model_version_tag(model_name, version, "origin", "wsl2_training")

    # 3. Promotion via Alias (La norme "New UI")
    # On utilise l'alias 'champion' pour désigner le meilleur modèle actuel
    print(f"[3/3] Assignation de l'alias '@champion' à la V{version}...")
    try:
        client.set_registered_model_alias(model_name, "champion", version)

        print("Succès ! Le modèle est désormais 'champion' et en 'Production'.")
    except Exception as e:
        print(f"Note: Promotion partielle effectuée. Détail: {e}")

    print(f"--- Publication terminée (Version {version}) ---")
    return version


if __name__ == "__main__":

    RUN_ID = "adec664f8bd4409db957b3d6eea78aee"
    # calculé précédemment
    register_and_promote(RUN_ID, threshold=0.9262)
