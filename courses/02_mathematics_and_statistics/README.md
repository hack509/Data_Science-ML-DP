# Module 2 — Mathématiques et statistiques pour l'IA

Le Machine Learning repose sur quatre piliers mathématiques : l'**algèbre** (manipuler des expressions et des fonctions), l'**algèbre linéaire** (vecteurs et matrices), les **probabilités et statistiques** (décrire et raisonner sous incertitude) et le **calcul différentiel** (optimiser). Ce module les introduit pas à pas, toujours avec Python pour calculer, visualiser et vérifier.

## Connaissances requises

- Module 1 (en particulier NumPy, leçon 1.4).
- Arithmétique de base : fractions, puissances, priorités opératoires. Les rappels nécessaires sont faits dans la leçon 2.1.

## Durée estimée

15 à 20 heures (5 leçons de 2 à 4 h + exercices).

## Objectifs du module

- Lire et manipuler une formule, tracer une fonction et interpréter sa courbe.
- Calculer avec des vecteurs et des matrices, et relier le produit matriciel à la prédiction d'un modèle linéaire.
- Décrire une distribution (centre, dispersion, forme) et la relation entre deux variables.
- Raisonner avec des probabilités, des lois usuelles, des intervalles de confiance et des tests d'hypothèse.
- Comprendre dérivée, gradient et descente de gradient, le moteur de l'apprentissage des modèles.

## Leçons

1. [Algèbre, fonctions et graphiques](01_algebre_fonctions.md)
2. [Vecteurs, matrices et algèbre linéaire](02_vecteurs_matrices.md)
3. [Statistiques descriptives, corrélation et covariance](03_statistiques_descriptives.md)
4. [Probabilités, lois, échantillonnage et inférence](04_probabilites_inference.md)
5. [Dérivées, gradients et descente de gradient](05_derivees_gradient.md)

Corrigés : [corrections.md](corrections.md).

## Exercices requis

| Identifiant | Sujet | Niveau |
|---|---|---|
| B05 | Statistiques descriptives sans bibliothèque | débutant |
| B06 | Opérations matricielles | débutant |
| I04 | Descente de gradient pour la régression linéaire | intermédiaire |
| I05 | Intervalle de confiance et test d'hypothèse | intermédiaire |

## Jalon pratique

Implémenter une régression linéaire **de zéro** par descente de gradient (exercice I04), puis vérifier que les coefficients obtenus sont identiques à ceux de la solution exacte calculée avec l'algèbre linéaire.

## Critères de validation

- Exercices B05, B06, I04 et I05 réussis.
- Score au quiz du module ≥ 70 %.
- Vous savez expliquer avec vos mots : pourquoi la médiane résiste aux valeurs aberrantes, ce qu'est une p-value (et ce qu'elle n'est pas), et pourquoi on suit l'opposé du gradient.
