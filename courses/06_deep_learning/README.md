# Module 6 — Deep Learning avec PyTorch

Ce module construit pas à pas la compréhension des réseaux de neurones : d'un neurone mathématique écrit à la main, à un réseau en NumPy avec rétropropagation codée soi-même, puis aux mêmes réseaux avec **PyTorch** (tenseurs, différentiation automatique, modules, optimiseurs, boucles d'entraînement). Tous les exemples tournent **sur CPU** en quelques secondes ; aucun GPU n'est nécessaire.

## Connaissances requises

- Module 2 (vecteurs, matrices, dérivées, descente de gradient) et module 5 (méthodologie d'évaluation).
- PyTorch installé (`python -m pip install -r requirements-torch.txt`) ; vérification : `python setup/check_environment.py`.

## Durée estimée

16 à 20 heures (4 leçons + exercices A01 et A02).

## Objectifs du module

- Manipuler des tenseurs et comprendre la différentiation automatique (*autograd*) et le graphe de calcul.
- Expliquer neurone, couche, fonction d'activation, fonction de perte, propagation avant et rétropropagation.
- Écrire une boucle d'entraînement et de validation complète en PyTorch.
- Résoudre des problèmes de classification binaire et multiclasse.
- Utiliser régularisation, dropout, normalisation par lots et arrêt précoce.
- Sauvegarder et recharger des poids ; estimer mémoire et temps de calcul.

## Leçons

1. [Tenseurs, opérations et différentiation automatique](01_tenseurs_autograd.md)
2. [Du neurone au réseau : un réseau de neurones en NumPy](02_neurone_reseau_numpy.md)
3. [Entraîner un réseau avec PyTorch : classification binaire](03_pytorch_entrainement.md)
4. [Classification multiclasse et régularisation](04_multiclasse_regularisation.md)

Corrigés : [corrections.md](corrections.md).

## Exercices requis

| Identifiant | Sujet | Niveau |
|---|---|---|
| A01 | Réseau de neurones en NumPy avec rétropropagation | avancé |
| A02 | Boucle d'entraînement PyTorch réutilisable | avancé |

## Jalon pratique

Entraîner sur CPU un réseau PyTorch de classification multiclasse, avec jeu de validation, arrêt précoce et sauvegarde des meilleurs poids, puis recharger le modèle et vérifier qu'il prédit à l'identique.

## Critères de validation

- Exercices A01 et A02 réussis.
- Score au quiz du module ≥ 70 %.
- Vous savez expliquer avec vos mots ce que font `loss.backward()`, `optimizer.step()` et `optimizer.zero_grad()`.

## GPU (optionnel)

Tout le module fonctionne sur CPU. Si vous disposez d'une carte NVIDIA, voir `docs/installation_guide.md` (section « GPU (optionnel) ») ; le code des leçons utilise `device = "cuda" if torch.cuda.is_available() else "cpu"` quand c'est pertinent.
