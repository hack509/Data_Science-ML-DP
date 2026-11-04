# Module 3 — Analyse de données

Un modèle de Machine Learning ne vaut que ce que valent ses données. Ce module enseigne le travail qui occupe la majeure partie du temps d'un data scientist : inspecter, nettoyer, transformer, explorer et contrôler des données, à partir de fichiers locaux volontairement imparfaits.

## Connaissances requises

- Module 1, leçon 1.5 (pandas).
- Module 2, leçon 2.3 (statistiques descriptives).

## Durée estimée

12 à 15 heures (4 leçons de 2 à 3 h + exercices).

## Objectifs du module

- Charger des données locales et en dresser un premier diagnostic.
- Traiter valeurs manquantes, doublons, types incorrects et valeurs aberrantes en justifiant chaque choix.
- Créer de nouvelles variables utiles (*feature engineering*).
- Mener une analyse exploratoire (EDA) structurée avec agrégations.
- Fusionner des tables et automatiser des contrôles de qualité.

## Leçons

1. [Inspection et nettoyage des données](01_inspection_nettoyage.md)
2. [Valeurs aberrantes, transformations et création de variables](02_outliers_transformation_features.md)
3. [Analyse exploratoire, agrégations et statistiques descriptives](03_eda_groupby.md)
4. [Fusion de tables et contrôles de qualité](04_fusion_qualite.md)

Corrigés : [corrections.md](corrections.md).

## Exercices requis

| Identifiant | Sujet | Niveau |
|---|---|---|
| B03 | Nettoyage d'un fichier CSV (rappel du module 1) | débutant |
| I02 | Valeurs manquantes et valeurs aberrantes | intermédiaire |
| I03 | Fusion de tables et agrégations | intermédiaire |

## Jalon pratique

Nettoyer intégralement `datasets/raw/students_raw.csv` (exercice I02), le fusionner avec `student_activities.csv` (exercice I03) et produire un tableau de synthèse par groupe et par activité, accompagné d'un rapport de qualité.

## Critères de validation

- Exercices I02 et I03 réussis.
- Score au quiz du module ≥ 70 %.
- Vous savez justifier, pour une colonne donnée, le choix entre supprimer, imputer par la médiane ou signaler une valeur manquante.
