# Leçon 9.1 — Images, pixels, canaux et prétraitement

## Prérequis

- Leçon 1.4 (tableaux NumPy, formes, axes) ; leçon 6.1 (tenseurs).

## Objectifs d'apprentissage

- Expliquer comment une image est représentée en mémoire (pixels, canaux, type `uint8`).
- Charger des images avec Pillow et les convertir en tableaux NumPy et en tenseurs PyTorch.
- Redimensionner, recadrer et convertir en niveaux de gris.
- Normaliser des images (division par 255, standardisation par canal) et passer du format HWC au format CHW.
- Constituer un jeu de données d'images avec un index et vérifier sa cohérence.

## Concepts clés

- **Pixel** : plus petit élément d'une image ; une intensité (niveaux de gris) ou plusieurs (couleur).
- **Canal** (*channel*) : une composante de couleur (rouge, vert, bleu) ; une image RVB a 3 canaux.
- **Résolution** : nombre de pixels en hauteur × largeur.
- **`uint8`** : entier non signé sur 8 bits (0 à 255), le type habituel des images stockées.
- **HWC / CHW** : ordre des axes (hauteur, largeur, canaux) en NumPy et Pillow ; (canaux, hauteur, largeur) en PyTorch.
- **Normalisation** : mise à l'échelle des intensités (0-1 ou moyenne 0 / écart-type 1).
- **Interpolation** : calcul des nouveaux pixels lors d'un redimensionnement.

## Explication détaillée

### Une image est un tableau

Une image en niveaux de gris de 28 × 28 pixels est une matrice de 28 lignes et 28 colonnes ; chaque case contient une intensité de 0 (noir) à 255 (blanc). Une image couleur ajoute un troisième axe : 3 valeurs (rouge, vert, bleu) par pixel, soit un tableau de forme (hauteur, largeur, 3). Une photo de 4000 × 3000 pixels en couleur contient 36 millions de nombres : c'est pourquoi on travaille souvent sur des images réduites.

Tout ce que vous savez de NumPy s'applique : `image[10:20, 5:15]` recadre, `image[:, ::-1]` retourne horizontalement, `image.mean()` donne la luminosité moyenne, `255 - image` inverse les couleurs.

### Charger et convertir

Pillow (`PIL.Image`) lit et écrit les formats courants (PNG, JPEG...). `Image.open(chemin)` puis `np.asarray(image)` donne le tableau. `image.convert("L")` convertit en niveaux de gris, `image.resize((largeur, hauteur))` redimensionne (attention : Pillow donne la taille sous la forme **largeur, hauteur**, NumPy la forme **hauteur, largeur**). Le PNG est sans perte ; le JPEG compresse avec perte (petits artefacts autour des contours).

### Normaliser

Les réseaux apprennent mieux avec des entrées petites et centrées :

1. Conversion en réels et division par 255 : intensités entre 0 et 1.
2. Éventuellement, standardisation par canal avec la moyenne et l'écart-type calculés **sur le jeu d'entraînement** : (x − μ) / σ.

PyTorch attend des tenseurs de forme (N, C, H, W) : nombre d'images, canaux, hauteur, largeur. Pour une image NumPy HWC, on transpose avec `np.transpose(img, (2, 0, 1))` ; pour des images en niveaux de gris de forme (N, H, W), on ajoute l'axe des canaux avec `x[:, None, :, :]`.

### Le jeu de données du module

`scripts/generate_datasets.py` dessine 100 cercles, 100 carrés et 100 triangles de taille, position, intensité et orientation (pour les triangles) aléatoires, avec du bruit. Les fichiers sont dans `datasets/generated/shapes/<classe>/` et un index CSV (`shapes/index.csv`) associe chaque chemin à son étiquette. Le fichier `shapes.npz` contient les mêmes images sous forme de tableaux NumPy, plus rapide à charger.

## Intuition mathématique

```text
image niveaux de gris : I ∈ {0, ..., 255}^(H×W)      image couleur : I ∈ {0, ..., 255}^(H×W×3)
normalisation : x = I / 255 ∈ [0, 1]       standardisation : x' = (x − μ_c) / σ_c  (par canal c)
niveaux de gris depuis RVB (formule ITU-R 601, utilisée par Pillow) : L = 0,299 R + 0,587 G + 0,114 B
```

## Applications concrètes

- Contrôle qualité industriel (pièces défectueuses), lecture de documents, imagerie médicale, tri d'images.
- Toutes ces applications commencent par un prétraitement identique à l'entraînement et en production.

## Exemples de code Python

### Exemple 1 — charger une image et l'inspecter

```python
import numpy as np
import pandas as pd
from PIL import Image

index = pd.read_csv("datasets/generated/shapes/index.csv")
print(index.head(3).to_string(index=False))
print(index["label"].value_counts().to_dict())

image = Image.open("datasets/" + index.loc[0, "path"])
print("Pillow:", image.size, image.mode)
array = np.asarray(image)
print("NumPy:", array.shape, array.dtype, "min", array.min(), "max", array.max())
center = array[10:18, 10:18]
print("8x8 crop around the centre:\n", center)
```

### Explication du code

- L'index relie chaque fichier PNG à son étiquette : 100 images par classe.
- Pillow rapporte la taille (largeur, hauteur) et le mode `L` (niveaux de gris sur 8 bits). Converti en NumPy, c'est un tableau (28, 28) de `uint8`.
- Le recadrage central montre les valeurs brutes : intensités élevées (forme claire) sur fond sombre bruité.

### Sortie attendue

```text output
                                  path  label  label_id
generated/shapes/circle/circle_000.png circle         0
generated/shapes/circle/circle_001.png circle         0
generated/shapes/circle/circle_002.png circle         0
{'circle': 100, 'square': 100, 'triangle': 100}
Pillow: (28, 28) L
NumPy: (28, 28) uint8 min 0 max 234
8x8 crop around the centre:
 [[194 200 192 202 205 194 202 202]
 [201 212 209 209 232 190 207 196]
 [200 196 198 205 187 200 218 234]
 [190 205 209 202 198 221 219 191]
 [193 217 226 212 205 211 197   0]
 [193 220 190 211 217 202 211   0]
 [189 212 212 202 207 197   5   0]
 [181 197 223 198 205   5   0   0]]
```

### Exemple 2 — couleurs, canaux et opérations de base

```python
from pathlib import Path

rgb = np.zeros((40, 60, 3), dtype=np.uint8)
rgb[:, :20, 0] = 255           # red band
rgb[:, 20:40, 1] = 255         # green band
rgb[:, 40:, 2] = 255           # blue band
rgb[10:30, 25:35] = [255, 255, 0]  # yellow square
color_image = Image.fromarray(rgb)
gray = np.asarray(color_image.convert("L"))
print("RGB shape:", rgb.shape, "-> gray shape:", gray.shape)
print("gray value of pure red, green, blue, yellow:", gray[0, 0], gray[0, 30], gray[0, 50], gray[20, 30])

shape = np.asarray(Image.open("datasets/" + index.loc[0, "path"]))
flipped = shape[:, ::-1]
inverted = 255 - shape
resized = np.asarray(Image.fromarray(shape).resize((14, 14), Image.Resampling.BILINEAR))
print("flipped:", flipped.shape, "| inverted mean:", round(inverted.mean(), 1), "| resized:", resized.shape)

output_dir = Path("outputs/module09")
output_dir.mkdir(parents=True, exist_ok=True)
color_image.save(output_dir / "rgb_bands.png")
Image.fromarray(np.hstack([shape, flipped, inverted])).resize((168, 56), Image.Resampling.NEAREST).save(output_dir / "basic_operations.png")
```

### Explication du code

- On fabrique une image couleur à la main : chaque bande allume un seul canal ; le jaune combine rouge et vert.
- La conversion en niveaux de gris applique la pondération 0,299 R + 0,587 G + 0,114 B : le vert pur paraît plus clair (150) que le rouge (76) et surtout que le bleu (29), comme pour l'œil humain.
- Retourner, inverser et redimensionner sont des opérations sur des tableaux. `NEAREST` agrandit sans lisser (pratique pour regarder de petites images, dans `outputs/module09/basic_operations.png`).

### Sortie attendue

```text output
RGB shape: (40, 60, 3) -> gray shape: (40, 60)
gray value of pure red, green, blue, yellow: 76 150 29 226
flipped: (28, 28) | inverted mean: 205.6 | resized: (14, 14)
```

### Exemple 3 — préparer tout le jeu de données pour PyTorch

```python
import torch
from sklearn.model_selection import train_test_split

data = np.load("datasets/generated/shapes.npz")
images, labels, class_names = data["images"], data["labels"], [str(c) for c in data["class_names"]]
print("images:", images.shape, images.dtype, "| classes:", class_names)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, stratify=labels, random_state=0)
x_train = X_train.astype(np.float32) / 255.0
mean, std = x_train.mean(), x_train.std()  # statistics from the training images only
print(f"training pixels: mean={mean:.3f} std={std:.3f}")


def to_tensor(batch: np.ndarray) -> torch.Tensor:
    x = (batch.astype(np.float32) / 255.0 - mean) / std
    return torch.from_numpy(x[:, None, :, :])  # (N, H, W) -> (N, C=1, H, W)


train_tensor, test_tensor = to_tensor(X_train), to_tensor(X_test)
print("train tensor:", tuple(train_tensor.shape), train_tensor.dtype)
print(f"standardized train mean={train_tensor.mean():.3f} std={train_tensor.std():.3f} | test mean={test_tensor.mean():.3f}")
for name in class_names:
    k = class_names.index(name)
    print(f"{name:<9} mean intensity={images[labels == k].mean():.1f}")
```

### Explication du code

- Le fichier `.npz` charge les 300 images d'un coup.
- La moyenne et l'écart-type de standardisation sont calculés sur les images d'**entraînement** uniquement, puis appliqués aux deux jeux (comme `StandardScaler` dans un pipeline).
- `x[:, None, :, :]` ajoute l'axe des canaux : PyTorch attend (N, C, H, W).
- L'intensité moyenne diffère selon la classe (un carré de même « rayon » couvre plus de pixels qu'un cercle, lui-même plus qu'un triangle) : un indice simple qu'un modèle pourrait exploiter, mais qui ne suffit pas à classer de façon fiable.

### Sortie attendue

```text output
images: (300, 28, 28) uint8 | classes: ['circle', 'square', 'triangle']
training pixels: mean=0.219 std=0.347
train tensor: (225, 1, 28, 28) torch.float32
standardized train mean=-0.000 std=1.000 | test mean=0.038
circle    mean intensity=63.2
square    mean intensity=77.8
triangle  mean intensity=29.0
```

## Erreurs fréquentes

- **Confondre (largeur, hauteur) de Pillow et (hauteur, largeur) de NumPy**.
- **Calculs en `uint8`** : `200 + 100` donne 44 (dépassement). Convertissez en `float32` avant de calculer.
- **Oublier l'axe des canaux** ou l'ordre CHW : erreur de forme dans `nn.Conv2d`.
- **Normaliser différemment à l'entraînement et en production**.
- **Calculer les statistiques de normalisation sur toutes les images**, test compris.
- **Redimensionner en déformant** (changer le rapport largeur/hauteur) sans le vouloir.

## Exercices guidés

1. Affichez la valeur et la position du pixel le plus lumineux de la première image.
2. Calculez pour chaque classe la proportion moyenne de pixels dont l'intensité dépasse 100.
3. Créez une image 28 × 28 noire avec un carré blanc de 10 pixels au centre, enregistrez-la en PNG puis rechargez-la et vérifiez qu'elle est identique.
4. Vérifiez qu'il y a exactement 300 fichiers PNG dans `datasets/generated/shapes/` et qu'ils correspondent tous à une ligne de l'index.

Corrigés : [corrections.md](corrections.md), section « Leçon 9.1 ».

## Défis autonomes

- Écrivez une fonction `montage(images, n_columns)` qui assemble une grille d'images en une seule image PNG.
- Entraînez une régression logistique sur les pixels aplatis (784 variables). Quelle précision obtenez-vous ? (On comparera au CNN dans la leçon 9.2.)

## Quiz

1. Quelle est la forme d'un tableau NumPy représentant une image couleur de 640 × 480 pixels ?
2. Pourquoi convertir en `float32` avant de normaliser ?
3. Que signifie CHW ? Pourquoi PyTorch l'utilise-t-il ?
4. Pourquoi le vert paraît-il plus clair que le bleu en niveaux de gris ?
5. Sur quelles images calculer la moyenne de normalisation ?
6. Quelle différence entre PNG et JPEG ?

Réponses : [corrections.md](corrections.md), section « Leçon 9.1 ».

## Résumé

- Une image est un tableau (H, W) ou (H, W, C) d'entiers 0-255.
- Pillow lit, convertit, redimensionne ; NumPy recadre, retourne, calcule.
- Normaliser (÷ 255, standardiser avec les statistiques d'entraînement) et passer au format (N, C, H, W) pour PyTorch.

## Pour aller plus loin

- Leçon suivante : [Convolution, pooling et réseau convolutif](02_convolution_cnn.md).
