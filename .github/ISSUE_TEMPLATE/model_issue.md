---
name: Model issue
about: Problème de performance ou de dérive du modèle
title: 'model: '
labels: model-quality
---

## Problème détecté
- [ ] Dérive des données (data drift)
- [ ] Dérive du modèle (model drift)
- [ ] AUROC sous le seuil minimum
- [ ] Autre

## Métriques observées
| Métrique | Valeur actuelle | Seuil |
|---|---|---|
| AUROC | | 0.90 |
| F1 | | 0.85 |

## Run MLflow concerné
`run_id`:

## Action suggérée
- [ ] Re-training
- [ ] Ajustement du seuil
- [ ] Vérification des données
