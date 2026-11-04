# Projet 6 — Pipeline de Machine Learning de bout en bout

**Difficulté** : ★★★★ — **Modules** : 3, 5, 10 — **Durée indicative** : 8 à 12 heures

## Énoncé du problème

Un établissement veut identifier **en début de trimestre** les étudiants susceptibles d'échouer, pour leur proposer un accompagnement. Les seules informations disponibles à ce moment sont les habitudes de travail, l'assiduité, la note du trimestre précédent, le groupe et les activités — pas les notes du trimestre en cours (qui définissent la réussite et ne sont connues qu'à la fin).

Il faut livrer un **pipeline reproductible** : configuration, chargement des données brutes, nettoyage, contrôles de qualité, sélection de modèle, évaluation honnête (incertitude, sous-groupes), rapport, fiche modèle, et un outil de prédiction — le tout testé.

## Objectifs d'apprentissage

- Structurer un projet en modules courts et testables (configuration, données, modélisation, évaluation, rapport).
- Séparer nettoyage des données (déterministe) et prétraitement appris (dans le pipeline, sans fuite).
- Empêcher par construction l'utilisation de variables qui fuient la cible.
- Sélectionner un modèle par validation croisée en privilégiant la simplicité en cas de quasi-égalité.
- Produire automatiquement un rapport d'évaluation et une fiche modèle (leçons 10.1 et 10.2).

## Données

| Fichier | Rôle |
|---|---|
| `datasets/raw/students_raw.csv` | 308 lignes brutes (doublons, formats incohérents, valeurs manquantes ou impossibles) |
| `datasets/generated/student_activities.csv` | activités et heures de bénévolat |

Données **synthétiques** générées par `scripts/generate_datasets.py`. La cible `passed` (note finale ≥ 10) est calculée à partir des notes de maths, physique et français, qui sont **exclues des variables** (la configuration refuse de les utiliser). Les étudiants dont une note manque (absents en maths, physique manquante) ne peuvent pas être étiquetés et sont exclus : inventer leur note reviendrait à inventer la réponse.

## Architecture

```text
06_end_to_end_ml_project/
├── config.json                  paramètres (graine, variables, seuil, nombre de plis...)
├── run.py                       train | predict
├── e2e_pipeline/
│   ├── config.py                configuration typée et validée (refus des variables qui fuient)
│   ├── data.py                  chargement, nettoyage, cible, fusion, contrôles de qualité
│   ├── modeling.py              ColumnTransformer + candidats + sélection par validation croisée
│   ├── evaluation.py            métriques de test, intervalles bootstrap, sous-groupes
│   ├── reporting.py             report.md et model_card.md
│   └── pipeline.py              orchestration train / predict
├── tests/test_project.py
└── outputs/                     model.joblib, model_info.json, metrics.json, report.md, model_card.md
```

## Étapes

1. **Configuration** : lisez `config.json` dans une dataclass ; validez les valeurs et interdisez les colonnes qui fuient.
2. **Nettoyage** déterministe : doublons, groupes, assiduité, valeurs impossibles → manquantes. **Pas d'imputation ici** : elle sera apprise dans le pipeline, sur l'entraînement seulement.
3. **Cible** : note finale pondérée, `passed`, exclusion des lignes non étiquetables.
4. **Fusion** avec les activités ; **contrôles de qualité** qui arrêtent le pipeline en cas de problème.
5. **Découpage** stratifié 80/20.
6. **Prétraitement** : imputation médiane avec indicateurs de valeurs manquantes + standardisation ; imputation du mode + one-hot (catégories inconnues ignorées).
7. **Sélection** : référence (classe majoritaire), régression logistique, forêt aléatoire, gradient boosting ; validation croisée stratifiée (ROC-AUC) ; on retient le modèle le plus simple à moins d'une erreur standard du meilleur.
8. **Évaluation finale** : ROC-AUC et accuracy avec intervalles bootstrap, précision, rappel, F1, métriques par groupe.
9. **Livrables** : modèle, informations (variables, seuil, versions), métriques, rapport, fiche modèle ; commande `predict`.
10. **Tests** de chaque étape.

## Lancer la solution de référence

```bash
python projects/06_end_to_end_ml_project/run.py train
python projects/06_end_to_end_ml_project/run.py predict nouveaux_etudiants.csv predictions.csv
python -m pytest projects/06_end_to_end_ml_project -q
```

Le fichier d'entrée de `predict` doit contenir les colonnes : `study_hours_week, attendance_rate, sleep_hours, previous_score, volunteer_hours_year, group, club` (les valeurs manquantes sont acceptées).

## Indicateurs et résultats attendus

| Étape | Résultat |
|---|---|
| Données | 308 lignes brutes → 300 après doublons → **285 étiquetées** (15 sans note) ; taux de réussite ≈ 44,6 % |
| Validation croisée (ROC-AUC) | logistique ≈ 0,845 ± 0,039 ; gradient boosting ≈ 0,843 ; forêt ≈ 0,830 ; référence 0,5 |
| Modèle retenu | régression logistique (la plus simple, à égalité avec le gradient boosting) |
| Test (n = 57) | ROC-AUC ≈ 0,83 (IC 95 % ≈ [0,71 ; 0,92]) ; accuracy ≈ 0,74 |

L'intervalle de confiance est large : 57 étudiants de test ne permettent pas une estimation précise. Les métriques par groupe (12 à 26 étudiants chacun) sont encore plus incertaines.

## Défis pour aller plus loin

- Choisissez le seuil de décision selon l'usage (repérer au moins 80 % des étudiants en difficulté) sur des prédictions de validation croisée, et documentez le compromis.
- Remplacez le découpage unique par une validation croisée imbriquée pour estimer la performance de la procédure complète.
- Ajoutez un contrôle de dérive (PSI, leçon 10.2) entre les données d'entraînement et un nouveau fichier avant la prédiction.
- Ajoutez une commande `evaluate` qui recalcule les métriques d'un modèle sauvegardé sur un nouveau fichier étiqueté.

## Solution de référence

Le paquet `e2e_pipeline/` (voir l'architecture) et `run.py`.

## Pistes d'amélioration

- Journal d'exécution (module `logging`) et identifiant de version pour chaque entraînement.
- Configuration des candidats et de leurs grilles d'hyperparamètres dans `config.json`.
- Calibration des probabilités et analyse de l'équité selon plusieurs critères (leçon 10.2).
- Ne jamais utiliser une telle prédiction pour sanctionner : elle doit ouvrir un accompagnement, décidé par des personnes.
