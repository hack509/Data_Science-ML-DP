# Leçon 9.2 — Convolution, pooling et réseau convolutif

## Prérequis

- Leçon 9.1 ; leçons 6.3 et 6.4 (boucles d'entraînement, multiclasse) ; leçon 5.3 (matrice de confusion).

## Objectifs d'apprentissage

- Calculer une convolution 2D à la main et comprendre ce qu'un filtre détecte.
- Calculer la taille de sortie d'une convolution (noyau, pas, remplissage) et d'un pooling.
- Comprendre pourquoi un CNN a peu de paramètres et respecte la structure spatiale.
- Entraîner un petit CNN avec PyTorch sur CPU et le comparer à un MLP.
- Évaluer avec une matrice de confusion et visualiser les filtres et cartes de caractéristiques.

## Concepts clés

- **Convolution** : un petit filtre (*kernel*, ex. 3 × 3) glisse sur l'image ; à chaque position, on calcule la somme des produits filtre × pixels.
- **Carte de caractéristiques** (*feature map*) : image produite par un filtre.
- **Pas** (*stride*) : décalage du filtre entre deux positions. **Remplissage** (*padding*) : bordure ajoutée autour de l'image.
- **Pooling** (*max pooling*) : réduction de taille en gardant le maximum de chaque bloc (ex. 2 × 2).
- **Partage des poids** (*weight sharing*) : le même filtre est appliqué partout ; il détecte un motif où qu'il soit.
- **Champ réceptif** (*receptive field*) : zone de l'image d'entrée qui influence un neurone ; il grandit avec la profondeur.
- **Couche convolutive** (`nn.Conv2d(in_channels, out_channels, kernel_size)`).

## Explication détaillée

### Pourquoi pas un MLP ?

Un MLP qui reçoit une image 28 × 28 l'aplatit en 784 nombres : il perd la notion de voisinage et doit apprendre séparément à reconnaître un coin en haut à gauche et un coin en bas à droite. Une première couche dense de 128 neurones a déjà 784 × 128 ≈ 100 000 poids. Pour une photo de 224 × 224 en couleur, ce serait 19 millions de poids pour une seule couche !

### La convolution

Un filtre 3 × 3 contient 9 poids. On le place sur chaque fenêtre 3 × 3 de l'image, on multiplie terme à terme et on additionne : on obtient un nombre par position, donc une nouvelle image (carte de caractéristiques). Un filtre comme

```text
[-1  0  1]
[-1  0  1]
[-1  0  1]
```

répond fortement là où l'intensité augmente de gauche à droite : c'est un **détecteur de bords verticaux**. Dans un CNN, les valeurs des filtres ne sont pas choisies à la main : elles sont **apprises** par rétropropagation. Les premières couches apprennent des bords et des textures, les suivantes des combinaisons (coins, courbes), les dernières des parties d'objets.

Deux propriétés expliquent l'efficacité des CNN : **connexions locales** (chaque sortie ne regarde qu'une petite fenêtre) et **partage des poids** (le même filtre partout), d'où très peu de paramètres et une **équivariance par translation** (un motif décalé produit une réponse décalée).

Remarque : en Deep Learning, l'opération appelée « convolution » est en réalité une **corrélation croisée** (le filtre n'est pas retourné) ; comme les filtres sont appris, cela ne change rien.

### Taille des sorties

Pour une entrée de taille n, un filtre k, un remplissage p et un pas s :

```text
taille de sortie = ⌊(n + 2p − k) / s⌋ + 1
```

Exemples : 28 avec k = 3, p = 1, s = 1 → 28 (le remplissage « same » conserve la taille) ; 28 avec k = 3, p = 0 → 26 ; max pooling 2 × 2 de pas 2 : 28 → 14.

### Pooling

Le max pooling garde la plus forte réponse de chaque bloc 2 × 2 : il divise la taille par 2, réduit le calcul des couches suivantes et rend le réseau un peu tolérant aux petits décalages. Il n'a aucun paramètre.

### Architecture typique

[Conv → ReLU → Pool] × quelques blocs, avec un nombre de filtres croissant (8, 16, 32...), puis aplatissement et une ou deux couches denses vers les classes. Le nombre de paramètres d'une `Conv2d(C_in, C_out, k)` est C_out × (C_in × k × k + 1).

## Intuition mathématique

```text
sortie(i, j) = Σ_c Σ_u Σ_v  K[c, u, v] · X[c, i·s + u, j·s + v] + b
paramètres Conv2d : C_out × (C_in × k² + 1)        Linear(784, 128) : 784 × 128 + 128 = 100 480
max pooling 2×2 : sortie(i, j) = max(X[2i:2i+2, 2j:2j+2])
```

## Applications concrètes

- Classification d'images (contrôle qualité, tri de déchets, diagnostic sur radiographies).
- Détection et segmentation d'objets (conduite autonome, imagerie médicale), qui réutilisent les mêmes blocs convolutifs.
- Analyse de signaux 1D (audio, capteurs) avec `Conv1d`.

## Exemples de code Python

### Exemple 1 — une convolution à la main : détecter les bords

```python
from pathlib import Path

import numpy as np
from PIL import Image


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Valid cross-correlation (no padding, stride 1), written with explicit loops for clarity."""
    kh, kw = kernel.shape
    out = np.zeros((image.shape[0] - kh + 1, image.shape[1] - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = (image[i : i + kh, j : j + kw] * kernel).sum()
    return out


data = np.load("datasets/generated/shapes.npz")
images, labels = data["images"], data["labels"]
square = images[labels == 1][0].astype(np.float32) / 255.0
vertical_edges = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
horizontal_edges = vertical_edges.T
response_v = convolve2d(square, vertical_edges)
response_h = convolve2d(square, horizontal_edges)
print("input:", square.shape, "-> output:", response_v.shape)
print("strongest vertical-edge response at column:", np.unravel_index(np.abs(response_v).argmax(), response_v.shape)[1])
print("max |vertical| =", round(np.abs(response_v).max(), 2), "| max |horizontal| =", round(np.abs(response_h).max(), 2))


def to_png(array):
    scaled = (array - array.min()) / (array.max() - array.min() + 1e-8) * 255
    return Image.fromarray(scaled.astype(np.uint8)).resize((112, 112), Image.Resampling.NEAREST)


Path("outputs/module09").mkdir(parents=True, exist_ok=True)
for name, array in [("square", square), ("vertical_edges", response_v), ("horizontal_edges", response_h)]:
    to_png(array).save(f"outputs/module09/conv_{name}.png")
```

### Explication du code

- La fonction parcourt chaque fenêtre 3 × 3 et calcule la somme des produits : sans remplissage, une image 28 × 28 donne une sortie 26 × 26.
- Le filtre vertical répond fortement sur les bords gauche et droit du carré (valeurs positives d'un côté, négatives de l'autre), le filtre horizontal sur les bords haut et bas : voir `outputs/module09/conv_vertical_edges.png`.
- Le bruit de l'image produit de petites réponses partout : un réseau apprendra des filtres plus robustes.

### Sortie attendue

```text output
input: (28, 28) -> output: (26, 26)
strongest vertical-edge response at column: 25
max |vertical| = 1.85 | max |horizontal| = 1.92
```

### Exemple 2 — les couches PyTorch : formes et nombre de paramètres

```python
import torch
from torch import nn

torch.manual_seed(0)
x = torch.randn(8, 1, 28, 28)  # batch of 8 grayscale images
conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
pool = nn.MaxPool2d(2)
h = pool(torch.relu(conv(x)))
print("after conv (padding=1):", tuple(conv(x).shape), "| after pool:", tuple(h.shape))
print("conv parameters:", sum(p.numel() for p in conv.parameters()), "= 8 x (1 x 3 x 3 + 1)")
print("dense layer 784 -> 128 parameters:", sum(p.numel() for p in nn.Linear(784, 128).parameters()))

manual = nn.Conv2d(1, 1, kernel_size=3, bias=False)
with torch.no_grad():
    manual.weight[:] = torch.tensor(vertical_edges).view(1, 1, 3, 3)
    torch_response = manual(torch.tensor(square).view(1, 1, 28, 28))[0, 0].numpy()
print("PyTorch convolution equals our loop version:", np.allclose(torch_response, response_v, atol=1e-5))
```

### Explication du code

- Avec `padding=1`, la convolution 3 × 3 conserve la taille 28 × 28 ; le pooling 2 × 2 la ramène à 14 × 14. Les 8 filtres produisent 8 canaux.
- La couche convolutive n'a que 80 paramètres, contre plus de 100 000 pour une couche dense de 128 neurones sur l'image aplatie.
- En imposant notre filtre de bords à une `nn.Conv2d`, on retrouve exactement le résultat de la boucle : PyTorch calcule bien une corrélation croisée.

### Sortie attendue

```text output
after conv (padding=1): (8, 8, 28, 28) | after pool: (8, 8, 14, 14)
conv parameters: 80 = 8 x (1 x 3 x 3 + 1)
dense layer 784 -> 128 parameters: 100480
PyTorch convolution equals our loop version: True
```

### Exemple 3 — entraîner un CNN et le comparer à un MLP

```python
import time

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from lab.plotting import apply_style

class_names = [str(c) for c in data["class_names"]]
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, stratify=labels, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=0)
mean, std = (X_train / 255.0).mean(), (X_train / 255.0).std()
prep = lambda a: torch.tensor(((a / 255.0 - mean) / std)[:, None], dtype=torch.float32)
Xtr, Xva, Xte = prep(X_train), prep(X_val), prep(X_test)
ytr, yva, yte = (torch.tensor(a, dtype=torch.int64) for a in (y_train, y_val, y_test))


class SmallCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 7 * 7, 32), nn.ReLU(), nn.Linear(32, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def train(model, epochs=30, lr=3e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(0)
    best = (0.0, None)
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(Xtr), generator=generator).split(32):
            optimizer.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_acc = (model(Xva).argmax(1) == yva).float().mean().item()
        if val_acc >= best[0]:
            best = (val_acc, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    return model


results = {}
for name, make in [("MLP", lambda: nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 3))), ("CNN", SmallCNN)]:
    torch.manual_seed(0)
    start = time.perf_counter()
    model = train(make())
    with torch.no_grad():
        test_pred = model(Xte).argmax(1)
    results[name] = (model, test_pred)
    params = sum(p.numel() for p in model.parameters())
    print(f"{name}: parameters={params:>6}  test accuracy={(test_pred == yte).float().mean().item():.3f}")

cnn, cnn_pred = results["CNN"]
cm = confusion_matrix(yte.numpy(), cnn_pred.numpy())
print("CNN confusion matrix (rows = true", class_names, "):\n", cm)

apply_style()
filters = cnn.features[0].weight.detach()[:, 0]
fig, axes = plt.subplots(1, 8, figsize=(10, 1.6))
for ax, kernel in zip(axes, filters):
    ax.imshow(kernel, cmap="gray")
    ax.axis("off")
fig.suptitle("Filtres appris par la première couche")
fig.savefig("outputs/module09/learned_filters.png", bbox_inches="tight")
plt.close(fig)
```

### Explication du code

- Découpage entraînement / validation / test ; standardisation avec les statistiques d'entraînement.
- Le CNN suit l'architecture typique : deux blocs convolution-ReLU-pooling (28 → 14 → 7), puis une petite partie dense.
- La fonction `train` garde les poids de la meilleure époque de validation (arrêt précoce simplifié).
- Comparaison : sur ces petites images, le CNN atteint une meilleure précision que le MLP **avec moins de paramètres**, grâce au partage des poids et à la structure spatiale. La matrice de confusion montre les confusions restantes.
- Les triangles sont souvent pris pour des cercles (10 sur 25 dans la matrice) : avec seulement 180 images d'entraînement, le réseau n'a pas vu assez de variations de taille, de position et d'orientation. C'est précisément ce que la leçon 9.3 améliore avec l'augmentation de données. Sur 75 images de test, l'écart MLP/CNN (3 images) est d'ailleurs à confirmer avec plusieurs graines.
- Les filtres appris (`outputs/module09/learned_filters.png`) ressemblent souvent à des détecteurs de bords orientés, appris sans qu'on les ait programmés.

### Sortie attendue

```text output
MLP: parameters= 50435  test accuracy=0.800
CNN: parameters= 26467  test accuracy=0.840
CNN confusion matrix (rows = true ['circle', 'square', 'triangle'] ):
 [[24  1  0]
 [ 1 24  0]
 [10  0 15]]
```

## Erreurs fréquentes

- **Mauvaise taille d'entrée de la couche dense** après les convolutions : calculez C × H × W (ici 16 × 7 × 7) ou utilisez `nn.LazyLinear`.
- **Oublier l'axe des canaux** (N, 1, H, W) pour les images en niveaux de gris.
- **Trop de pooling** sur de petites images : 28 → 14 → 7 → 3 → 1, il ne reste plus d'information spatiale.
- **Évaluer le CNN en mode entraînement** (oubli de `model.eval()`).
- **Conclure qu'un CNN est meilleur** à partir d'une seule graine sur 75 images de test (leçon 7.3).

## Exercices guidés

1. Calculez la taille de sortie pour n = 32, k = 5, p = 0, s = 1, puis pour n = 32, k = 3, p = 1, s = 2.
2. Calculez le nombre de paramètres de `nn.Conv2d(16, 32, 3)` et de `nn.Conv2d(3, 64, 7)`.
3. Appliquez à la main un filtre de flou (moyenne 3 × 3) à un triangle et comparez le bruit avant et après.
4. Entraînez le CNN avec 4 puis 32 filtres dans la première couche (et 2× plus dans la deuxième). Comparez précision et paramètres.

Corrigés : [corrections.md](corrections.md), section « Leçon 9.2 ».

## Défis autonomes

- Exercice **A04** : convolution et max-pooling vectorisés en NumPy.
- Visualisez les 8 cartes de caractéristiques de la première couche pour un cercle, un carré et un triangle.
- Ajoutez une couche `nn.BatchNorm2d` après chaque convolution et mesurez l'effet sur 3 graines.

## Quiz

1. Pourquoi un CNN a-t-il beaucoup moins de paramètres qu'un MLP sur des images ?
2. Quelle est la taille de sortie d'une convolution 3 × 3 sans remplissage sur une image 28 × 28 ?
3. Que fait un max pooling 2 × 2 ? A-t-il des paramètres ?
4. Que détectent typiquement les filtres des premières couches ?
5. Qu'est-ce que le partage des poids ?
6. Combien de paramètres a `nn.Conv2d(1, 8, 3)` ?

Réponses : [corrections.md](corrections.md), section « Leçon 9.2 ».

## Résumé

- Convolution : un filtre appris glisse sur l'image ; connexions locales et partage des poids.
- Taille de sortie : ⌊(n + 2p − k)/s⌋ + 1 ; pooling 2 × 2 : taille divisée par 2.
- Un petit CNN entraîné sur CPU dépasse un MLP avec moins de paramètres sur des images.

## Pour aller plus loin

- Leçon suivante : [Augmentation de données et inférence](03_augmentation_inference.md).
