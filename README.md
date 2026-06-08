# Id-Fraud-Sentinel — MLOps Pipeline (Bac+5 Bloc 4 Lead Data Science)

Pipeline MLOps complet sur la détection automatique de fraude documentaire par deep learning.
Le modèle EfficientNet-B0 est entraîné et enregistré dans MLflow. Ce projet y ajoute la couche MLOps : CI/CD, monitoring de drift, réentraînement conditionnel et rollback automatique.

---

## Application d'inférence

**[VoxUp/Idnet-API](https://huggingface.co/spaces/VoxUp/Idnet-API)** — interface Streamlit déployée sur Hugging Face Spaces.

Upload une image de carte d'identité (PNG/JPG) → le modèle retourne :
- Verdict : **genuine** ou **fraud**
- Score de confiance (probabilité fraude)
- Visualisation Grad-CAM — zones de l'image ayant activé la détection

Le modèle est chargé au démarrage via l'alias `@champion` MLflow. CPU uniquement, budget zéro.

---

## Architecture générale

```
Modèle source (MLflow Registry)
  └─ IDNet-Fraud-Detector @champion  ←  EfficientNet-B0, AUROC 0.9587
       │
       ├─ cd.yml        charge par alias → déploie HF Space → refresh référence monitoring
       ├─ monitoring.yml  cron 06h UTC → drift report Evidently → alerte si drift
       └─ ct.yml         workflow_dispatch → vérifie santé du champion
```

---

## Dataset & Preprocessing

**Dataset** : IDNet ESP (Zenodo) — cartes d'identité espagnoles, 6 types de fraude (falsification textuelle, remplacement photo, face morphing, retouche numérique, altération fonds sécurisés, manipulation éléments de sécurité).

**Déséquilibre** : distribution originale 85% fraude / 15% genuine. Rééchantillonné à **30% fraude / 70% genuine** pour équilibrer l'apprentissage sans perdre la représentation des fraudes rares.

**Transforms** :
- Train : flip horizontal, rotation ±15°, color jitter, RandomCrop, normalisation ImageNet
- Val/Test : resize 224×224, normalisation ImageNet (déterministe)

**Loss** : CrossEntropyLoss pondérée — poids inversement proportionnels à la racine carrée des fréquences de classe, pour compenser le déséquilibre résiduel.

**Validation** : Great Expectations sur les splits train/val/test — zéro data leakage vérifié.

---

## Modèle

EfficientNet-B0 en transfer learning, entraîné en **2 phases** :

| Phase | Paramètres entraînés | Learning rate |
|---|---|---|
| 1 — Freeze backbone | Head classifier uniquement | 1e-3 |
| 2 — Fine-tuning partiel | Blocs 4→7 + classifier | 1e-4 |

Early stopping patience=3 sur la val AUROC.

**Résultats run E3** :

| Métrique | Valeur |
|---|---|
| AUROC | 0.9587 |
| F1 | 0.8476 |
| Recall fraude | 95.05% |
| Seuil optimal | 0.1014 |
| Précision @ seuil | 57.6% |

Seuil calibré via `precision_recall_curve` sous contrainte recall ≥ 95% (exigence KYC). Tous les types de fraude détectés, y compris face morphing.

---

## Stack technique

| Composant | Technologie |
|---|---|
| Modèle | EfficientNet-B0 (PyTorch) |
| Registry | MLflow 2.9.2 — NeonDB + S3 |
| Artifacts modèle | `s3://mlflow-remote-storage/idnet-fraud-bloc6/` |
| Données / prédictions | `s3://cni-fraud-detection/` |
| Monitoring | Evidently 0.6.5 |
| App d'inférence | Streamlit — HF Space `VoxUp/Idnet-API` |
| CI/CD | GitHub Actions |
| Expérience monitoring | MLflow `idnet-monitoring` |
| Environnement | Python 3.10.12, CPU uniquement, WSL2 |

---

## Workflows GitHub Actions

### CI (`ci.yml`) — push develop / PR main

```
lint (Black + Ruff)
  └─ test-unit (99 tests)
       └─ test-integration (5 tests)  ← push develop uniquement
```

### CD (`cd.yml`) — push main (si champion changé)

```
snapshot-champion          ← capture version actuelle pour rollback
  └─ register-model        ← charge @champion par alias
       └─ deploy-hf-space
            ├─ tag-deployed
            ├─ refresh-reference   ← régénère référence monitoring (S3 + MLflow)
            ├─ rollback            ← si failure → réalias + re-push HF Space + email
            └─ notify              ← email toujours envoyé
```

### Monitoring (`monitoring.yml`) — cron 06h UTC + dispatch

```
sync S3
  └─ generate_predictions (train_ref + test_run)
       └─ build_reference_csv
            └─ simulate_threshold (recall ≥ 95%)
                 └─ Evidently drift report → MLflow idnet-monitoring
                      └─ si drift → déclenche ct.yml
```

### CT (`ct.yml`) — workflow_dispatch

```
retrain_pipeline (stub)
  ├─ vérifie AUROC champion ≥ MIN_AUROC
  ├─ logue dans idnet-monitoring
  └─ exit(2) si dégradé → réentraînement manuel requis
```

---

## Référence monitoring

La référence Evidently est découplée du run champion :

- **MLflow** : expérience `idnet-monitoring`, run le plus récent tagué `reference_type=train_with_preds`
- **Fallback S3** : `s3://cni-fraud-detection/data/processed/train_with_preds.csv`
- **Refresh auto** : à chaque nouveau champion déployé (job `refresh-reference` dans `cd.yml`)

Bootstrap initial (one-shot) :
```bash
python -m src.monitoring.bootstrap_reference
```

---

## Installation

```bash
# 1. Environnement
conda create -n idnet-bac5 python=3.10.12 -y
conda activate idnet-bac5

# 2. PyTorch CPU en premier
pip install torch==2.2.2 torchvision==0.17.2 numpy==1.26.4 \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Dépendances
pip install -r requirements-cpu.txt
pip install -r requirements.txt

# 4. Package éditable
pip install -e .
```

### Variables d'environnement (`.env`)

```dotenv
MLFLOW_TRACKING_URI=https://VoxUp-mlflow-remote-server.hf.space
MLFLOW_MODEL_NAME=IDNet-Fraud-Detector
MLFLOW_ARTIFACT_ROOT=s3://mlflow-remote-storage/idnet-fraud-bloc6/
S3_BUCKET=cni-fraud-detection
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=eu-west-3
DATA_PROCESSED_DIR=data/processed
```

### GitHub Secrets requis

```
MLFLOW_TRACKING_URI · AWS_ACCESS_KEY_ID · AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION · S3_BUCKET · HF_TOKEN · HF_SPACE_REPO
GMAIL_USERNAME · GMAIL_APP_PASSWORD · GITHUB_ACCESS_TOKEN
```

---

## Tests

```bash
pytest tests/unit/ -v        # 99 tests unitaires
pytest tests/integration/ -v # 5 tests d'intégration
pytest -v                    # tous (104 tests)
```

---

## Lancement manuel

```bash
# Pipeline monitoring complet
python -m src.monitoring.run_monitoring_pipeline --target-recall 0.95

# Sans régénération des prédictions (données S3 existantes)
python -m src.monitoring.run_monitoring_pipeline --skip-generate

# Vérification du champion
python -m src.training.retrain_pipeline
```

---

## Structure du projet

```
src/
├── api/                          # FastAPI — endpoints inférence
├── data/                         # dataset, pipeline, validation GX
├── models/
│   ├── config.py                 # constantes (DEVICE, IMAGE_SIZE…)
│   ├── efficientnet.py           # FraudClassifier EfficientNet-B0
│   ├── evaluate.py               # évaluation + métriques
│   ├── register_model.py         # enregistrement MLflow Registry
│   ├── train.py                  # entraînement 2 phases
│   └── simulate_threshold.py     # recalcul seuil optimal ← Bac+5
├── monitoring/
│   ├── monitor_pro.py            # rapport Evidently + métriques MLflow
│   ├── run_monitoring_pipeline.py
│   ├── generate_predictions.py   # inférence → JSONs S3
│   ├── build_reference_csv.py    # JSONs S3 → CSV référence
│   └── bootstrap_reference.py   # init unique référence
└── training/
    └── retrain_pipeline.py       # CT stub — vérifie le champion

tests/
├── unit/          # 99 tests unitaires
└── integration/   # 5 tests d'intégration

.github/workflows/
├── ci.yml · cd.yml · ct.yml · monitoring.yml · changelog.yml
```

---

## Liens

- App d'inférence : https://huggingface.co/spaces/VoxUp/Idnet-API
- MLflow Server : https://VoxUp-mlflow-remote-server.hf.space
