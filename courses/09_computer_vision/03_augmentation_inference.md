# Leçon 9.3 — Augmentation de données et inférence

## Prérequis

- Leçon 9.2 (CNN) ; leçon 7.3 (comparaisons avec plusieurs graines).

## Objectifs d'apprentissage

- Comprendre l'augmentation de données et choisir des transformations qui préservent l'étiquette.
- Implémenter des augmentations avec PyTorch et NumPy (retournements, rotations, translations, bruit, luminosité), sans bibliothèque supplémentaire.
- Mesurer l'effet de l'augmentation de façon équitable (plusieurs graines).
- Sauvegarder un modèle de vision avec son prétraitement et prédire sur une nouvelle image (inférence).
- Tester la robustesse du modèle à des images différentes de celles de l'entraînement.

## Concepts clés

- **Augmentation de données** (*data augmentation*) : créer, pendant l'entraînement, des variantes aléatoires des images (tournées, décalées, bruitées) qui gardent la même étiquette.
- **Invariance** : le modèle doit donner la même réponse malgré une transformation (un cercle décalé reste un cercle).
- **Inférence** (*inference*) : utilisation d'un modèle entraîné pour prédire sur de nouvelles données.
- **Décalage de distribution** (*distribution shift*) : différence entre les données d'entraînement et celles rencontrées en production.
- **Augmentation au test** (*test-time augmentation*, TTA) : moyenne des prédictions sur plusieurs versions transformées d'une même image.

## Explication détaillée

### Pourquoi augmenter ?

Avec 180 images d'entraînement, le CNN de la leçon 9.2 n'a vu qu'un petit échantillon des positions, tailles et orientations possibles. Plutôt que de collecter de nouvelles images (coûteux), on **fabrique** des variantes plausibles à chaque époque : le réseau ne voit jamais deux fois exactement la même image et doit apprendre ce qui définit vraiment la classe. C'est un régularisateur très efficace en vision.

### Choisir les transformations

Une augmentation n'est valable que si **l'étiquette reste vraie** :

- retournements horizontaux et verticaux, rotations de 90° : sans risque pour des formes géométriques ; mais un « 6 » retourné devient un « 9 », et une radiographie retournée place le cœur à droite ;
- petites translations : presque toujours valables ;
- bruit, variations de luminosité et de contraste : simulent des conditions de prise de vue différentes ;
- recadrages et changements d'échelle : valables tant que l'objet reste visible.

Les augmentations s'appliquent **uniquement à l'entraînement**. Le jeu de validation et le jeu de test restent intacts : ils doivent représenter les vraies données.

Les bibliothèques comme `torchvision.transforms` fournissent ces opérations toutes faites ; elles ne sont pas nécessaires ici, et écrire les transformations soi-même montre qu'il ne s'agit que d'opérations sur des tenseurs.

### Inférence

Utiliser un modèle sur une nouvelle image exige de reproduire **exactement** le prétraitement de l'entraînement : même taille, mêmes niveaux de gris, même normalisation (moyenne et écart-type d'entraînement), même format (N, C, H, W). On sauvegarde donc avec les poids un fichier de métadonnées (classes, taille, statistiques). La fonction d'inférence renvoie la classe **et** les probabilités : une probabilité maximale faible signale une image inhabituelle, à traiter avec prudence.

### Décalage de distribution

Un modèle n'est fiable que sur des images semblables à celles de l'entraînement. Une forme plus grande, plus petite, inversée (forme sombre sur fond clair) ou une photo réelle d'un objet peut le tromper avec une grande confiance. Tester explicitement ces cas fait partie de l'évaluation.

## Intuition mathématique

```text
sans augmentation : min_θ  (1/n) Σᵢ L(f_θ(xᵢ), yᵢ)
avec augmentation : min_θ  (1/n) Σᵢ E_T[ L(f_θ(T(xᵢ)), yᵢ) ]     T tirée au hasard parmi les transformations
TTA : p(y | x) ≈ (1/m) Σⱼ p_θ(y | Tⱼ(x))
```

On minimise l'erreur moyenne sur toutes les transformations plausibles : le modèle est encouragé à être invariant à ces transformations.

## Applications concrètes

- Toute application de vision avec peu d'images annotées (industrie, médecine, agriculture).
- Déploiement d'un modèle : un script d'inférence qui charge les poids et le prétraitement, contrôle la confiance et journalise les cas douteux.

## Exemples de code Python

### Exemple 1 — des augmentations écrites avec PyTorch

```python
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn


def augment(batch: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Random label-preserving transformations for a batch of shape (N, 1, H, W)."""
    out = batch.clone()
    n = out.shape[0]
    flip_h = torch.rand(n, generator=generator) < 0.5
    out[flip_h] = out[flip_h].flip(-1)
    flip_v = torch.rand(n, generator=generator) < 0.5
    out[flip_v] = out[flip_v].flip(-2)
    rotations = torch.randint(0, 4, (n,), generator=generator)
    for k in range(1, 4):
        mask = rotations == k
        out[mask] = torch.rot90(out[mask], k, dims=(-2, -1))
    shifts = torch.randint(-3, 4, (n, 2), generator=generator)
    for i in range(n):
        out[i] = torch.roll(out[i], shifts=(int(shifts[i, 0]), int(shifts[i, 1])), dims=(-2, -1))
    brightness = 1 + 0.2 * (torch.rand(n, 1, 1, 1, generator=generator) - 0.5)
    noise = 0.1 * torch.randn(out.shape, generator=generator)
    return out * brightness + noise


data = np.load("datasets/generated/shapes.npz")
images, labels = data["images"], data["labels"]
class_names = [str(c) for c in data["class_names"]]
triangle = torch.tensor(images[labels == 2][0] / 255.0, dtype=torch.float32).view(1, 1, 28, 28)
generator = torch.Generator().manual_seed(0)
variants = augment(triangle.repeat(6, 1, 1, 1), generator)
print("variants:", tuple(variants.shape), "| all different:", len({tuple(v.flatten()[:50].tolist()) for v in variants}) == 6)

strip = np.hstack([np.clip(v[0].numpy(), 0, 1) for v in variants])
Path("outputs/module09").mkdir(parents=True, exist_ok=True)
Image.fromarray((strip * 255).astype(np.uint8)).resize((6 * 84, 84), Image.Resampling.NEAREST).save("outputs/module09/augmentations.png")
print("saved: outputs/module09/augmentations.png")
```

### Explication du code

- Chaque image du lot subit, au hasard, un retournement horizontal et/ou vertical, une rotation de 0, 90, 180 ou 270°, une translation de −3 à +3 pixels (`torch.roll`, qui fait réapparaître de l'autre côté ce qui sort : acceptable ici car le fond est noir), une variation de luminosité de ±10 % et du bruit.
- Toutes ces transformations gardent l'étiquette : un triangle tourné reste un triangle.
- Les 6 variantes d'un même triangle sont visibles dans `outputs/module09/augmentations.png`.

### Sortie attendue

```text output
variants: (6, 1, 28, 28) | all different: True
saved: outputs/module09/augmentations.png
```

### Exemple 2 — mesurer l'effet de l'augmentation sur 3 graines

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, stratify=labels, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=0)
mean, std = float((X_train / 255.0).mean()), float((X_train / 255.0).std())


def prep(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(((array / 255.0 - mean) / std)[:, None], dtype=torch.float32)


Xtr, Xva, Xte = prep(X_train), prep(X_val), prep(X_test)
ytr, yva, yte = (torch.tensor(a, dtype=torch.int64) for a in (y_train, y_val, y_test))


def make_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16 * 7 * 7, 32), nn.ReLU(), nn.Linear(32, 3),
    )


def train(use_augmentation: bool, seed: int, epochs: int = 40) -> nn.Module:
    torch.manual_seed(seed)
    model = make_cnn()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(seed)
    best = (-1.0, None)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(Xtr), generator=generator).split(32):
            batch = Xtr[idx]
            if use_augmentation:
                # augment in pixel space (0-1), then standardize again
                batch = (augment(batch * std + mean, generator) - mean) / std
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(batch), ytr[idx]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_acc = (model(Xva).argmax(1) == yva).float().mean().item()
        if val_acc >= best[0]:
            best = (val_acc, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    return model


scores = {False: [], True: []}
models = {}
for use_augmentation in (False, True):
    for seed in (0, 1, 2):
        model = train(use_augmentation, seed)
        with torch.no_grad():
            scores[use_augmentation].append((model(Xte).argmax(1) == yte).float().mean().item())
        models[(use_augmentation, seed)] = model
    values = np.array(scores[use_augmentation])
    print(f"augmentation={use_augmentation!s:<5}: test accuracy per seed={values.round(3).tolist()}  mean={values.mean():.3f}")
```

### Explication du code

- Les augmentations sont appliquées **dans l'espace des pixels** (0 à 1), puis les images sont standardisées à nouveau ; le jeu de validation et le jeu de test ne sont jamais augmentés.
- Chaque configuration est entraînée avec 3 graines et le même budget (40 époques), en gardant la meilleure époque de validation.
- Sur ces données, l'augmentation améliore nettement la précision de test, et de façon cohérente sur les 3 graines : le réseau a appris que l'orientation et la position ne comptent pas, ce qui corrige en particulier les confusions entre triangles et cercles.

### Sortie attendue

```text output
augmentation=False: test accuracy per seed=[0.84, 0.853, 0.88]  mean=0.858
augmentation=True : test accuracy per seed=[0.973, 0.973, 0.987]  mean=0.978
```

### Exemple 3 — sauvegarder, recharger et prédire sur de nouvelles images

```python
import json

from PIL import ImageDraw

model_dir = Path("outputs/module09/shape_classifier")
model_dir.mkdir(parents=True, exist_ok=True)
best_model = models[(True, int(np.argmax(scores[True])))]
torch.save(best_model.state_dict(), model_dir / "weights.pt")
(model_dir / "metadata.json").write_text(json.dumps({"classes": class_names, "size": 28, "mean": mean, "std": std}, indent=2), encoding="utf-8")


def load_classifier(folder: Path):
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    model = make_cnn()
    model.load_state_dict(torch.load(folder / "weights.pt", weights_only=True))
    return model.eval(), metadata


def predict_image(path: Path, model, metadata) -> dict:
    image = Image.open(path).convert("L").resize((metadata["size"], metadata["size"]))
    x = (np.asarray(image, dtype=np.float32) / 255.0 - metadata["mean"]) / metadata["std"]
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.tensor(x).view(1, 1, metadata["size"], metadata["size"])), dim=1)[0]
    best = int(probabilities.argmax())
    return {"class": metadata["classes"][best], "confidence": round(float(probabilities[best]), 3)}


def draw(shape: str, size: int, center=(32, 32), canvas: int = 64, invert: bool = False) -> Path:
    image = Image.new("L", (canvas, canvas), 255 if invert else 0)
    pen = ImageDraw.Draw(image)
    color = 0 if invert else 220
    x, y, r = center[0], center[1], size
    if shape == "circle":
        pen.ellipse([x - r, y - r, x + r, y + r], fill=color)
    elif shape == "square":
        pen.rectangle([x - r, y - r, x + r, y + r], fill=color)
    else:
        pen.polygon([(x, y - r), (x + r, y + r), (x - r, y + r)], fill=color)
    path = model_dir / f"new_{shape}_{size}_{'inv' if invert else 'std'}.png"
    image.save(path)
    return path


classifier, metadata = load_classifier(model_dir)
for shape in ["circle", "square", "triangle"]:
    print(f"new {shape:<8} (64x64 image, resized):", predict_image(draw(shape, 16), classifier, metadata))
print("inverted triangle (dark on white):", predict_image(draw("triangle", 16, invert=True), classifier, metadata))
print("tiny circle:", predict_image(draw("circle", 4), classifier, metadata))
```

### Explication du code

- On sauvegarde les poids **et** les métadonnées (classes, taille, moyenne, écart-type). La fonction `load_classifier` recrée l'architecture et recharge les poids avec `weights_only=True`.
- `predict_image` applique le même prétraitement qu'à l'entraînement : niveaux de gris, redimensionnement à 28 × 28, normalisation, format (1, 1, 28, 28). Elle renvoie la classe et la confiance.
- Les nouveaux cercle et carré, dessinés en 64 × 64 puis redimensionnés, sont reconnus avec une confiance élevée. Le nouveau triangle est bien classé, mais avec une confiance faible (≈ 0,48) : il est isocèle (base horizontale), alors que les triangles d'entraînement sont équilatéraux et tournés au hasard. Une confiance basse est ici un signal utile : l'image ressemble moins aux exemples connus.
- Les deux derniers cas testent un **décalage de distribution** : une forme sombre sur fond clair et un cercle minuscule. Le modèle se trompe dans les deux cas **avec une confiance très élevée** (carré à 1,0 ; triangle à 0,96). C'est la leçon la plus importante de l'inférence : hors de la distribution d'entraînement, une confiance élevée ne garantit rien. Il faut tester ces situations, élargir les données ou les augmentations (inversion des couleurs, changements d'échelle), et prévoir un contrôle humain pour les cas inhabituels.

### Sortie attendue

```text output
new circle   (64x64 image, resized): {'class': 'circle', 'confidence': 0.995}
new square   (64x64 image, resized): {'class': 'square', 'confidence': 0.997}
new triangle (64x64 image, resized): {'class': 'triangle', 'confidence': 0.476}
inverted triangle (dark on white): {'class': 'square', 'confidence': 1.0}
tiny circle: {'class': 'triangle', 'confidence': 0.955}
```

## Erreurs fréquentes

- **Augmenter le jeu de validation ou de test**.
- **Augmentations qui changent l'étiquette** (retourner un texte, un chiffre, un panneau « tourner à gauche »).
- **Prétraitement d'inférence différent** de celui de l'entraînement (oublier la normalisation, inverser largeur et hauteur).
- **Ne sauvegarder que les poids** sans les classes ni les statistiques de normalisation.
- **Faire confiance à la probabilité maximale** sur des images très différentes des données d'entraînement.
- **Conclure sur l'effet de l'augmentation avec une seule graine**.

## Exercices guidés

1. Écrivez une augmentation « contraste » qui multiplie l'écart à la moyenne de chaque image par un facteur tiré entre 0,7 et 1,3.
2. Retirez les rotations de `augment` et mesurez l'effet sur la précision (3 graines).
3. Implémentez l'augmentation au test (TTA) : moyenne des probabilités sur l'image et ses 3 rotations de 90°. Comparez la précision de test avec et sans TTA.
4. Modifiez `predict_image` pour renvoyer « incertain » quand la confiance est inférieure à 0,7.

Corrigés : [corrections.md](corrections.md), section « Leçon 9.3 ».

## Défis autonomes

- Générez 100 nouvelles images de test plus difficiles (formes plus petites ou plus grandes, contraste plus faible) et mesurez la chute de précision. Proposez une augmentation qui la réduit.
- Projet 5 (`projects/05_image_classification`) : un pipeline de classification d'images complet.

## Quiz

1. Pourquoi l'augmentation de données réduit-elle le surapprentissage ?
2. Donnez une augmentation valable et une augmentation dangereuse pour la reconnaissance de chiffres manuscrits.
3. Pourquoi ne pas augmenter le jeu de test ?
4. Que faut-il sauvegarder en plus des poids pour faire de l'inférence ?
5. Qu'est-ce qu'un décalage de distribution ?
6. Une confiance de 0,99 garantit-elle une bonne prédiction ?

Réponses : [corrections.md](corrections.md), section « Leçon 9.3 ».

## Résumé

- L'augmentation crée des variantes plausibles qui gardent l'étiquette ; elle s'applique seulement à l'entraînement.
- Son effet se mesure sur plusieurs graines, à budget égal.
- L'inférence reproduit exactement le prétraitement ; on sauvegarde poids + métadonnées.
- Hors de la distribution d'entraînement, prédictions et confiance ne sont plus fiables.

## Pour aller plus loin

- Projet 5 (`projects/05_image_classification`).
- Module 10 : projets complets, déploiement et IA responsable.
