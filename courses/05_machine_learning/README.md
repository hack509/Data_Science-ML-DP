# Module 5 — Machine Learning

Ce module présente l'apprentissage automatique avec **scikit-learn** : les grandes familles de problèmes, la méthodologie d'évaluation (la partie la plus importante !) et neuf algorithmes classiques. Chaque algorithme est présenté selon la même fiche : intuition, idée mathématique, implémentation, données, entraînement, évaluation, interprétation, erreurs fréquentes, exercices et corrigé.

Tous les jeux de données sont **synthétiques** et générés localement : les scores obtenus illustrent les méthodes et ne disent rien des performances sur des données réelles.

## Connaissances requises

- Modules 1 à 4 ; en particulier vecteurs et matrices (2.2), statistiques (2.3), descente de gradient (2.5), nettoyage (3.1, 3.2).

## Durée estimée

25 à 30 heures (6 leçons de 3 à 4 h + exercices).

## Objectifs du module

- Distinguer IA, Machine Learning et Deep Learning ; apprentissage supervisé et non supervisé.
- Évaluer un modèle sans se tromper : jeux d'entraînement / validation / test, validation croisée, fuite de données.
- Diagnostiquer sous-apprentissage et surapprentissage (compromis biais-variance).
- Entraîner, évaluer et interpréter : régression linéaire, régression logistique, arbres de décision, forêts aléatoires, k plus proches voisins, SVM, Naive Bayes, k-means, ACP.
- Construire des pipelines de prétraitement, régler les hyperparamètres et sauvegarder un modèle.

## Leçons

1. [Introduction au Machine Learning et méthodologie d'évaluation](01_introduction_ml.md)
2. [Régression : régression linéaire et métriques](02_regression.md)
3. [Classification : logistique, k-NN, Naive Bayes, SVM et métriques](03_classification.md)
4. [Arbres de décision et forêts aléatoires](04_arbres_forets.md)
5. [Apprentissage non supervisé : k-means et ACP](05_non_supervise.md)
6. [Pipelines, validation croisée, hyperparamètres et persistance](06_pipelines_validation.md)

Corrigés : [corrections.md](corrections.md).

## Exercices requis

| Identifiant | Sujet | Niveau |
|---|---|---|
| I07 | Pipeline de régression sur les prix immobiliers | intermédiaire |
| I08 | Métriques de classification codées à la main | intermédiaire |
| I09 | k-means implémenté de zéro | intermédiaire |
| A05 | Validation croisée de zéro et fuite de données | avancé |

## Jalon pratique

Projet 2 (prix immobiliers) : pipeline complet avec prétraitement, comparaison de modèles par validation croisée, réglage d'hyperparamètres, évaluation finale sur un jeu de test intouché et sauvegarde du modèle.

## Critères de validation

- Exercices I07, I08, I09 et A05 réussis.
- Score au quiz du module ≥ 70 %.
- Vous savez expliquer pourquoi on ne touche au jeu de test qu'une seule fois, et repérer une fuite de données dans un script.
