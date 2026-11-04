# Module 9 — Vision par ordinateur

Ce module montre comment un ordinateur « voit » une image (un tableau de nombres), comment la préparer, puis comment les réseaux de neurones **convolutifs** (CNN) apprennent à reconnaître des formes. Les données sont 300 petites images en niveaux de gris (cercles, carrés, triangles) **générées localement** par `scripts/generate_datasets.py` avec Pillow. Tout s'entraîne sur CPU en quelques secondes ; aucune bibliothèque supplémentaire (comme torchvision) ni aucun modèle pré-entraîné n'est nécessaire.

## Connaissances requises

- Module 6 (PyTorch) et module 5 (évaluation, matrice de confusion).

## Durée estimée

12 à 15 heures (3 leçons + exercice A04 + projet 5).

## Objectifs du module

- Représenter une image comme un tableau de pixels et de canaux ; charger, redimensionner et normaliser des images.
- Comprendre la convolution et le pooling, les calculer à la main et avec PyTorch.
- Entraîner et évaluer un CNN de classification ; lire une matrice de confusion.
- Augmenter les données pour améliorer la robustesse ; réaliser l'inférence sur une nouvelle image.

## Leçons

1. [Images, pixels, canaux et prétraitement](01_images_pixels.md)
2. [Convolution, pooling et réseau convolutif](02_convolution_cnn.md)
3. [Augmentation de données et inférence](03_augmentation_inference.md)

Corrigés : [corrections.md](corrections.md).

## Exercices requis

| Identifiant | Sujet | Niveau |
|---|---|---|
| A04 | Convolution 2D et max-pooling en NumPy | avancé |

## Jalon pratique

Projet 5 (`projects/05_image_classification`) : un classifieur d'images complet, avec augmentation, courbes d'apprentissage, matrice de confusion, sauvegarde et script d'inférence.

## Critères de validation

- Exercice A04 réussi.
- Score au quiz du module ≥ 70 %.
- Vous savez calculer la taille de sortie d'une convolution et expliquer pourquoi un CNN a beaucoup moins de paramètres qu'un MLP sur des images.
