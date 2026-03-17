# Guide de contribution

## Setup initial

```bash
# 1. Cloner le repo
git clone https://github.com/<username>/fraud-detection.git
cd fraud-detection

# 2. Créer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate       # WSL2 / Linux / Mac

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Installer pre-commit
pip install pre-commit
pre-commit install               # hooks sur git commit
pre-commit install --hook-type commit-msg  # hook sur le message
```

## Stratégie de branches

```
main        → production stable (protégée — PR obligatoire)
develop     → intégration continue
feature/*   → nouvelles fonctionnalités  ex: feature/add-monitoring
hotfix/*    → corrections urgentes       ex: hotfix/fix-api-crash
```

## Workflow quotidien

```bash
# Toujours partir de develop
git checkout develop
git pull origin develop

# Créer une branche feature
git checkout -b feature/ma-feature

# ... coder ...

# Commiter (pre-commit s'exécute automatiquement)
git add .
git commit -m "feat: add data drift detection"

# Pousser et ouvrir une PR vers develop
git push origin feature/ma-feature
```

## Format des commits (conventionnels)

```
<type>: <description courte en minuscules>

Types autorisés :
  feat      → nouvelle fonctionnalité
  fix       → correction de bug
  model     → modification du modèle ML
  data      → modification des données
  ci        → modification CI/CD
  test      → ajout ou modification de tests
  docs      → documentation
  refactor  → refactoring
  chore     → maintenance (deps, config...)

Exemples :
  feat: add efficientnet-b0 classifier
  fix: correct auroc threshold in quality gate
  model: unfreeze backbone after 5 epochs
  data: add european subset download script
  ci: add continuous training workflow
```

## Versioning (Semantic Versioning)

```
v1.0.0  → release majeure
v1.1.0  → nouvelle feature
v1.1.1  → bug fix
```

Pour créer une release :
```bash
git tag v1.0.0
git push origin v1.0.0
# → déclenche automatiquement cd.yml + changelog.yml
```
