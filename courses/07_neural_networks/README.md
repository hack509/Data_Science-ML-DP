# Module 7 — Réseaux de neurones : architectures, entraînement et expérimentation

Le module 6 a présenté les outils (tenseurs, autograd, boucles d'entraînement). Ce module approfondit les **réseaux de neurones eux-mêmes** : d'où vient le perceptron, ce que change chaque fonction d'activation, pourquoi l'initialisation des poids compte, comment lire une courbe d'apprentissage, comment déboguer un réseau qui n'apprend pas, et comment mener des expériences rigoureuses pour comparer des architectures.

## Connaissances requises

- Module 6 complet (en particulier les leçons 6.2 et 6.3).

## Durée estimée

10 à 12 heures (3 leçons + défis).

## Objectifs du module

- Expliquer le perceptron, le perceptron multicouche et le rôle des activations ReLU, sigmoïde et softmax.
- Relier entropie croisée et maximum de vraisemblance.
- Choisir une initialisation adaptée et diagnostiquer gradients qui s'évanouissent ou explosent.
- Lire des courbes d'apprentissage et appliquer une méthode de débogage systématique.
- Concevoir une expérience comparative (plusieurs graines, même budget) et en tirer des conclusions prudentes.

## Leçons

1. [Perceptron, perceptron multicouche et fonctions d'activation](01_perceptron_mlp.md)
2. [Initialisation, courbes d'apprentissage et débogage](02_initialisation_debogage.md)
3. [Expérimenter : comparer des architectures et des hyperparamètres](03_experimentation_architectures.md)

Corrigés : [corrections.md](corrections.md).

## Exercices requis

| Identifiant | Sujet | Niveau |
|---|---|---|
| A01 | Réseau NumPy multicouche (rappel du module 6) | avancé |
| A02 | Boucle d'entraînement réutilisable (rappel du module 6) | avancé |

Les défis autonomes des leçons 7.2 et 7.3 constituent la pratique principale de ce module.

## Jalon pratique

Mener une étude comparative d'au moins 4 architectures sur un même problème, avec 3 graines aléatoires chacune, et rédiger un paragraphe de conclusion qui distingue les écarts significatifs du bruit.

## Critères de validation

- Score au quiz du module ≥ 70 %.
- Vous savez appliquer le test « surapprendre un seul lot » pour déboguer un réseau.
- Vous savez expliquer pourquoi un réseau profond à activations sigmoïdes apprend mal.
