# Projet 5 — Classification d'images

**Difficulté** : ★★★☆ — **Modules** : 6, 7, 9 — **Durée indicative** : 6 à 10 heures — **PyTorch requis**

## Énoncé du problème

Une chaîne de tri doit reconnaître automatiquement la forme de pièces photographiées (cercle, carré, triangle) à partir de petites images en niveaux de gris. Les images varient en taille, position, intensité, orientation et contiennent du bruit. Il faut un classifieur **entraîné sur CPU en moins d'une minute**, évalué honnêtement, et un script d'inférence qui traite de nouveaux fichiers PNG.

## Objectifs d'apprentissage

- Charger un jeu d'images à partir de fichiers et d'un index CSV (leçon 9.1).
- Concevoir un CNN (convolutions, normalisation par lots, pooling, dropout) et compter ses paramètres (leçon 9.2).
- Entraîner avec augmentation de données, validation et arrêt précoce (leçons 6.4, 9.3).
- Évaluer : précision, rappel par classe, matrice de confusion, courbes d'apprentissage.
- Sauvegarder poids et métadonnées ; prédire sur de nouveaux fichiers (leçon 9.3).

## Données

300 images PNG 28 × 28 en niveaux de gris, **générées localement** par `scripts/generate_datasets.py` avec Pillow : 100 cercles, 100 carrés, 100 triangles (`datasets/generated/shapes/<classe>/`). L'index `datasets/generated/shapes/index.csv` donne le chemin et l'étiquette de chaque fichier. Aucune image n'est téléchargée.

## Architecture

```text
05_image_classification/
├── run.py                         entraîner (défaut) ou predict image.png...
├── shape_classifier/training.py   chargement, modèle, augmentation, entraînement, évaluation, prédiction
├── tests/test_project.py
└── outputs/                       weights.pt, metadata.json, metrics.json, learning_curves.png
```

Le modèle : Conv(16) → BatchNorm → ReLU → MaxPool → Conv(32) → BatchNorm → ReLU → MaxPool → Dropout(0,3) → Dense(64) → Dense(3), soit 105 507 paramètres.

## Étapes

1. **Charger** les PNG listés dans l'index, convertir en tableaux, encoder les étiquettes.
2. **Découper** 64 / 16 / 20 % stratifié (entraînement / validation / test).
3. **Normaliser** avec la moyenne et l'écart-type des pixels d'entraînement.
4. **Augmenter** à la volée les seules images d'entraînement : retournements, rotations de 90°, translations, contraste, bruit.
5. **Entraîner** avec Adam, lots de 32, arrêt précoce sur la perte de validation (patience 12) et restauration des meilleurs poids.
6. **Évaluer une fois** sur le test : précision, rappel par classe, matrice de confusion ; tracer les courbes d'apprentissage.
7. **Sauvegarder** poids + métadonnées (classes, taille, normalisation) ; **prédire** sur des fichiers PNG.

## Lancer la solution de référence

```bash
python projects/05_image_classification/run.py            # about 15-30 s on CPU
python projects/05_image_classification/run.py --quick    # 8 epochs, for tests
python projects/05_image_classification/run.py predict datasets/generated/shapes/triangle/triangle_007.png
python -m pytest projects/05_image_classification -q
```

## Indicateurs et résultats attendus

Sur la machine de test (CPU, 2 cœurs) : 60 époques en ≈ 13 s, meilleure époque ≈ 53, **précision de test 1,0** (60 images, 20 par classe, aucune confusion). Ce score parfait vient d'un problème synthétique simple et d'un jeu de test petit : avec 60 images, il est compatible avec une précision réelle de l'ordre de 95 % (intervalle de confiance large). Comparez avec la leçon 9.3, où le même type de réseau **sans** augmentation plafonnait vers 86 %.

## Défis pour aller plus loin

- Générez un jeu de test « difficile » (formes plus petites ou plus grandes, contraste faible, forme sombre sur fond clair) et mesurez la chute de précision ; ajoutez des augmentations qui la corrigent.
- Ajoutez une quatrième classe (losange, étoile) dans `scripts/generate_datasets.py` et réentraînez.
- Comparez 3 architectures sur 3 graines chacune (leçon 7.3) : nombre de filtres, avec ou sans BatchNorm, avec ou sans dropout.
- Visualisez les cartes de caractéristiques de la première couche pour chaque classe.

## Solution de référence

`shape_classifier/training.py` : `load_images`, `make_model`, `augment`, `train`, `plot_history`, `run`, `predict_files`.

## Pistes d'amélioration

- Validation croisée stratifiée pour une estimation plus fiable sur un si petit jeu.
- Seuil de confiance avec réponse « incertain » et journal des cas douteux.
- Rotation d'angles arbitraires et changements d'échelle (avec `torch.nn.functional.affine_grid`).
