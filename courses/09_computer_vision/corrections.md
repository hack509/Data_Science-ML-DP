# Corrigés — Module 9 : Vision par ordinateur

## Leçon 9.1

### Quiz

1. (480, 640, 3) : hauteur, largeur, canaux.
2. Pour éviter les dépassements et les arrondis des entiers 8 bits (200 + 100 = 44 en `uint8`) et obtenir des valeurs fractionnaires.
3. Canaux, hauteur, largeur. Les convolutions de PyTorch sont organisées pour ce format (un filtre par canal d'entrée).
4. Parce que la conversion pondère le vert (0,587) bien plus que le bleu (0,114), à l'image de la sensibilité de l'œil humain.
5. Sur les images d'entraînement seulement.
6. PNG est sans perte (idéal pour des données et des images synthétiques) ; JPEG compresse avec perte (photos, fichiers plus légers mais artefacts).

### Exercices guidés

```python
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

data = np.load("datasets/generated/shapes.npz")
images, labels, class_names = data["images"], data["labels"], [str(c) for c in data["class_names"]]
first = images[0]
row, col = np.unravel_index(first.argmax(), first.shape)
print("brightest pixel:", int(first.max()), "at", (int(row), int(col)))

for k, name in enumerate(class_names):
    print(f"{name:<9} share of pixels > 100: {(images[labels == k] > 100).mean():.3f}")

square = np.zeros((28, 28), dtype=np.uint8)
square[9:19, 9:19] = 255
path = Path("outputs/module09/my_square.png")
path.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(square).save(path)
print("identical after reload:", np.array_equal(np.asarray(Image.open(path)), square))

index = pd.read_csv("datasets/generated/shapes/index.csv")
files = {p.relative_to("datasets").as_posix() for p in Path("datasets/generated/shapes").rglob("*.png")}
print("png files:", len(files), "| all indexed:", files == set(index["path"]))
```

```text output
brightest pixel: 234 at (12, 17)
circle    share of pixels > 100: 0.287
square    share of pixels > 100: 0.376
triangle  share of pixels > 100: 0.125
identical after reload: True
png files: 300 | all indexed: True
```

## Leçon 9.2

### Quiz

1. Grâce aux connexions locales (chaque neurone ne voit qu'une petite fenêtre) et au partage des poids (le même filtre est appliqué partout).
2. 26 × 26.
3. Il garde le maximum de chaque bloc 2 × 2 et divise la hauteur et la largeur par 2. Il n'a aucun paramètre.
4. Des bords et des contrastes orientés, des textures simples.
5. L'utilisation des mêmes poids (le même filtre) à toutes les positions de l'image.
6. 8 × (1 × 3 × 3 + 1) = 80.

### Exercices guidés

```python
import numpy as np
from torch import nn


def output_size(n, k, p, s):
    return (n + 2 * p - k) // s + 1


print("n=32, k=5, p=0, s=1 ->", output_size(32, 5, 0, 1), "| n=32, k=3, p=1, s=2 ->", output_size(32, 3, 1, 2))
for layer in (nn.Conv2d(16, 32, 3), nn.Conv2d(3, 64, 7)):
    print(layer, "->", sum(p.numel() for p in layer.parameters()), "parameters")

data = np.load("datasets/generated/shapes.npz")
triangle = data["images"][data["labels"] == 2][0].astype(np.float32) / 255.0
blur = np.ones((3, 3)) / 9
blurred = np.zeros((26, 26))
for i in range(26):
    for j in range(26):
        blurred[i, j] = (triangle[i : i + 3, j : j + 3] * blur).sum()
background_before = triangle[:5, :5].std()
background_after = blurred[:4, :4].std()
print(f"background noise std: before={background_before:.3f} after blur={background_after:.3f}")
```

```text output
n=32, k=5, p=0, s=1 -> 28 | n=32, k=3, p=1, s=2 -> 16
Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1)) -> 4640 parameters
Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1)) -> 9472 parameters
background noise std: before=0.033 after blur=0.013
```

Exercice 4 : avec 4 filtres, le modèle a moins de paramètres et une précision en général un peu plus faible ; avec 32 filtres, beaucoup plus de paramètres pour un gain faible ou nul sur un jeu aussi petit. Mesurez sur plusieurs graines avant de conclure (leçon 7.3).

## Leçon 9.3

### Quiz

1. Le modèle voit à chaque époque des variantes différentes des images : il ne peut pas mémoriser les exemples et doit apprendre les caractéristiques invariantes de chaque classe.
2. Valable : petites translations, légères rotations (± 10°), bruit. Dangereuse : retournement horizontal ou rotation de 180° (un 6 devient un 9).
3. Le test doit représenter les données réelles sur lesquelles le modèle sera utilisé ; augmenter le test mesurerait autre chose.
4. L'architecture (le code qui la recrée), la liste des classes dans le bon ordre, la taille attendue et les paramètres de normalisation.
5. Une différence entre les données d'entraînement et celles rencontrées en utilisation (autre appareil photo, autre éclairage, autres objets).
6. Non : un modèle peut être très confiant et faux, en particulier sur des images différentes de celles de l'entraînement.

### Exercices guidés

```python
import torch


def contrast(batch: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    factor = 0.7 + 0.6 * torch.rand(batch.shape[0], 1, 1, 1, generator=generator)
    mean = batch.mean(dim=(1, 2, 3), keepdim=True)
    return mean + factor * (batch - mean)


generator = torch.Generator().manual_seed(0)
batch = torch.rand(4, 1, 28, 28, generator=generator)
augmented = contrast(batch, generator)
print("means preserved:", torch.allclose(batch.mean(dim=(1, 2, 3)), augmented.mean(dim=(1, 2, 3)), atol=1e-6))
print("std ratios:", [round(v, 2) for v in (augmented.std(dim=(1, 2, 3)) / batch.std(dim=(1, 2, 3))).tolist()])


def predict_with_tta(model, x: torch.Tensor) -> torch.Tensor:
    """Average probabilities over the image and its three 90-degree rotations."""
    with torch.no_grad():
        probabilities = [torch.softmax(model(torch.rot90(x, k, dims=(-2, -1))), dim=1) for k in range(4)]
    return torch.stack(probabilities).mean(dim=0)


def label_with_threshold(probabilities: torch.Tensor, class_names, threshold: float = 0.7) -> str:
    confidence, index = probabilities.max(dim=0)
    return class_names[int(index)] if confidence >= threshold else "incertain"


print(label_with_threshold(torch.tensor([0.5, 0.3, 0.2]), ["circle", "square", "triangle"]))
print(label_with_threshold(torch.tensor([0.9, 0.05, 0.05]), ["circle", "square", "triangle"]))
```

```text output
means preserved: True
std ratios: [1.09, 1.0, 0.94, 1.08]
incertain
circle
```

Exercices 2 et 3 : reprenez la fonction `train` de la leçon (avec `augment` sans la boucle de rotations), puis comparez `model(Xte).argmax(1)` et `predict_with_tta(model, Xte).argmax(1)` sur le jeu de test pour les 3 graines. Sans rotations, le réseau doit apprendre l'invariance à l'orientation à partir des seules données et la précision baisse en général ; la TTA apporte au mieux un petit gain quand le modèle est déjà invariant aux rotations.
