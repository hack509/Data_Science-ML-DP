# Module 1 — Python pour la Data Science

Ce module consolide les bases de Python sous l'angle de l'analyse de données : on y manipule des collections, on écrit des fonctions réutilisables, on lit et écrit des fichiers, puis on passe aux deux bibliothèques incontournables, **NumPy** et **pandas**.

## Connaissances requises

- Savoir lancer un script Python et utiliser un terminal.
- Avoir déjà écrit une variable, un `if` et une boucle `for` (même maladroitement).
- Environnement installé (voir `docs/installation_guide.md`).

## Durée estimée

12 à 16 heures (5 leçons de 2 à 3 h + exercices).

## Objectifs du module

- Choisir la bonne structure de données (liste, tuple, ensemble, dictionnaire).
- Écrire des fonctions claires, documentées et testables.
- Lire et écrire des fichiers texte, CSV et JSON en gérant les erreurs.
- Comprendre les bases de la programmation orientée objet et des environnements virtuels.
- Calculer efficacement avec des tableaux NumPy (vectorisation).
- Charger, filtrer, transformer et sauvegarder des données avec pandas.

## Leçons

1. [Bases de Python : types, collections, conditions et boucles](01_bases_python.md)
2. [Fonctions, modules, fichiers et exceptions](02_fonctions_modules_fichiers.md)
3. [Programmation orientée objet et environnements virtuels](03_poo_et_environnements.md)
4. [NumPy : calcul numérique vectorisé](04_numpy.md)
5. [pandas : DataFrames, CSV et JSON](05_pandas_csv_json.md)

Corrigés des quiz et des exercices guidés : [corrections.md](corrections.md).

## Exercices requis

| Identifiant | Sujet | Niveau |
|---|---|---|
| B01 | Moyennes d'étudiants | débutant |
| B02 | Analyse de ventes | débutant |
| B03 | Nettoyage d'un fichier CSV | débutant |
| B04 | Import et transformation de JSON | débutant |
| I01 | Script de traitement de données réutilisable | intermédiaire |

## Jalon pratique

Écrire un script `clean_sales.py` qui lit `datasets/raw/sales_raw.csv`, le nettoie, calcule le chiffre d'affaires par région et enregistre le résultat en CSV et en JSON (c'est l'exercice I01).

## Critères de validation

- Les tests des exercices B01 à B04 et I01 passent (`pytest exercises/beginner exercises/intermediate/I01_data_processing_script`).
- Score au quiz du module ≥ 70 %.
- Vous savez expliquer la différence entre une liste Python et un tableau NumPy, et entre une `Series` et un `DataFrame`.
