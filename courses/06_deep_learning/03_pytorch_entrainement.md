# Leçon 6.3 — Entraîner un réseau avec PyTorch : classification binaire

## Prérequis

- Leçons 6.1 et 6.2 ; leçon 5.3 (métriques de classification) ; leçon 5.6 (standardisation dans un pipeline).

## Objectifs d'apprentissage

- Définir un réseau avec `nn.Module` et `nn.Sequential`.
- Utiliser une fonction de perte (`nn.BCEWithLogitsLoss`) et un optimiseur (`SGD`, `Adam`).
- Charger les données par mini-lots avec `TensorDataset` et `DataLoader`.
- Écrire des boucles d'entraînement et de validation correctes (`model.train()`, `model.eval()`, `torch.no_grad()`).
- Comprendre taux d'apprentissage, taille de lot et nombre d'époques.
- Sauvegarder et recharger des poids (`state_dict`) ; estimer mémoire et temps de calcul.

## Concepts clés

- **`nn.Module`** : classe de base de tous les modèles PyTorch ; contient des paramètres et une méthode `forward`.
- **`nn.Linear(n_in, n_out)`** : couche dense (poids + biais).
- **Logit** : score avant la sigmoïde. `BCEWithLogitsLoss` combine sigmoïde et entropie croisée de façon numériquement stable.
- **Optimiseur** (*optimizer*) : algorithme de mise à jour des poids ; **SGD** (descente de gradient stochastique) ou **Adam** (taux adaptatif par paramètre).
- **Mini-lot** (*batch*) : sous-ensemble d'exemples utilisé pour une mise à jour ; **taille de lot** (*batch size*).
- **Époque** (*epoch*) : un passage sur tout le jeu d'entraînement.
- **`state_dict`** : dictionnaire des paramètres d'un modèle, à sauvegarder avec `torch.save`.

## Explication détaillée

### Le modèle

Un réseau PyTorch est une classe qui hérite de `nn.Module` : on déclare les couches dans `__init__` et on décrit le calcul dans `forward`. Pour un empilement simple, `nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 1))` suffit. La dernière couche renvoie un **logit** par exemple ; on n'applique pas la sigmoïde dans le modèle, car `BCEWithLogitsLoss` le fait elle-même de façon stable. Pour obtenir une probabilité à la prédiction : `torch.sigmoid(logits)`.

### Les quatre lignes de l'entraînement

Pour chaque mini-lot :

```python no-run
optimizer.zero_grad()          # 1. reset the gradients
loss = loss_fn(model(xb), yb)  # 2. forward pass and loss
loss.backward()                # 3. backpropagation (autograd)
optimizer.step()               # 4. update the parameters
```

C'est exactement ce que nous avons écrit à la main en NumPy (leçon 6.2) : PyTorch fournit la rétropropagation (`backward`) et la mise à jour (`step`).

### Mini-lots, époques, taux d'apprentissage

- La **descente de gradient stochastique** calcule le gradient sur un mini-lot (32 à 256 exemples) plutôt que sur tout le jeu : beaucoup plus de mises à jour par seconde, et un bruit qui aide à sortir des mauvais minima. `DataLoader(dataset, batch_size=32, shuffle=True)` découpe et mélange les données à chaque époque.
- **Taille de lot** : petite = mises à jour bruitées mais nombreuses ; grande = gradients précis mais plus de mémoire et souvent une moins bonne généralisation. 32 à 128 sont de bons points de départ.
- **Taux d'apprentissage** : le réglage le plus important. Adam avec `lr=1e-3` est un défaut raisonnable ; SGD demande souvent un taux plus élevé (0,01 à 0,1) et un moment (*momentum*).
- **Nombre d'époques** : jusqu'à ce que la perte de **validation** cesse de baisser (arrêt précoce, leçon 6.4).

### Entraînement et évaluation

`model.train()` et `model.eval()` changent le comportement de certaines couches (dropout, normalisation par lots, leçon 6.4). L'évaluation se fait dans `torch.no_grad()` : pas de graphe de calcul, moins de mémoire. On suit à chaque époque la perte d'entraînement et la perte de validation : c'est la **courbe d'apprentissage**.

### Préparer les données

Les réseaux exigent des entrées **standardisées** (moyennes et écarts-types calculés sur l'entraînement seulement), au format `float32`. Les étiquettes binaires sont des `float32` de forme (n, 1) pour `BCEWithLogitsLoss`.

### Mémoire et temps de calcul

Le nombre de paramètres d'une couche `Linear(n_in, n_out)` est n_in × n_out + n_out. En `float32`, chaque paramètre occupe 4 octets ; l'entraînement avec Adam stocke en plus le gradient et deux moyennes par paramètre (≈ 4 fois la taille du modèle), plus les activations intermédiaires de chaque lot (proportionnelles à la taille de lot). Un réseau de 1 million de paramètres pèse donc environ 4 Mo, et ≈ 16 Mo pendant l'entraînement : rien pour un ordinateur portable. Le temps d'une époque est proportionnel au nombre d'exemples × au nombre de paramètres ; mesurez-le avec `time.perf_counter()` sur quelques époques avant de lancer un long entraînement. `torch.get_num_threads()` indique le nombre de cœurs CPU utilisés.

### Sauvegarder et recharger

On sauvegarde le **`state_dict`** (les poids), pas l'objet Python entier : `torch.save(model.state_dict(), "model.pt")`. Pour recharger, on recrée la même architecture puis `model.load_state_dict(torch.load("model.pt", weights_only=True))`. L'option `weights_only=True` refuse de charger autre chose que des tenseurs (sécurité, comme pour joblib en leçon 5.6). Sauvegardez aussi les paramètres de standardisation, sans lesquels les prédictions seraient fausses.

## Intuition mathématique

```text
logit z = f_θ(x)            p = σ(z)
BCEWithLogits(z, y) = −[y log σ(z) + (1 − y) log(1 − σ(z))]   (calculée sans former σ(z) explicitement)
SGD :  θ ← θ − η ∇L_lot(θ)
Adam : θ ← θ − η · m̂ / (√v̂ + ε)    m, v = moyennes mobiles du gradient et de son carré
```

## Applications concrètes

- Toute classification binaire sur données tabulaires, textuelles ou images (spam, fraude, défaut, diagnostic).
- La même boucle d'entraînement sert, avec d'autres modèles et pertes, à tous les projets de Deep Learning.

## Exemples de code Python

### Exemple 1 — données, modèle et nombre de paramètres

```python
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
df = pd.read_csv("datasets/generated/binary_classification.csv")
X = df.drop(columns="target").to_numpy(dtype=np.float32)
y = df["target"].to_numpy(dtype=np.float32).reshape(-1, 1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

mean, std = X_train.mean(axis=0), X_train.std(axis=0)  # statistics from the training set only
X_train, X_val = (X_train - mean) / std, (X_val - mean) / std
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=32, shuffle=True)
X_val_t, y_val_t = torch.from_numpy(X_val), torch.from_numpy(y_val)


def make_model():
    return nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))


model = make_model()
print(model)
n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params} -> {n_params * 4 / 1024:.1f} KiB in float32")
print("batches per epoch:", len(train_loader))
```

### Explication du code

- Les données sont converties en `float32` et standardisées avec les statistiques de l'entraînement.
- `TensorDataset` associe entrées et étiquettes ; `DataLoader` produit des mini-lots de 32 exemples mélangés à chaque époque.
- Le réseau 6 → 32 → 16 → 1 a (6×32 + 32) + (32×16 + 16) + (16×1 + 1) = 769 paramètres : environ 3 Kio. Rien à voir avec les milliards de paramètres des grands modèles de langage, mais le principe est identique.

### Sortie attendue

```text output
Sequential(
  (0): Linear(in_features=6, out_features=32, bias=True)
  (1): ReLU()
  (2): Linear(in_features=32, out_features=16, bias=True)
  (3): ReLU()
  (4): Linear(in_features=16, out_features=1, bias=True)
)
parameters: 769 -> 3.0 KiB in float32
batches per epoch: 15
```

### Exemple 2 — boucles d'entraînement et de validation

```python
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, X, y, loss_fn):
    model.eval()
    logits = model(X)
    accuracy = ((logits > 0).float() == y).float().mean().item()
    return loss_fn(logits, y).item(), accuracy


loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
start = time.perf_counter()
for epoch in range(1, 61):
    train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
    val_loss, val_acc = evaluate(model, X_val_t, y_val_t, loss_fn)
    if epoch in (1, 10, 30, 60):
        print(f"epoch {epoch:>2}: train loss={train_loss:.3f}  val loss={val_loss:.3f}  val accuracy={val_acc:.3f}")
elapsed = time.perf_counter() - start
print("training took less than 30 s on CPU:", elapsed < 30)
```

### Explication du code

- `train_one_epoch` contient les quatre lignes clés ; la perte moyenne est pondérée par la taille de chaque lot.
- `evaluate` est décorée par `@torch.no_grad()` et passe le modèle en mode évaluation. Un logit positif correspond à une probabilité supérieure à 0,5.
- La perte de validation baisse puis se stabilise ; la précision de validation rejoint celle du SVM de la leçon 5.3 (≈ 85 %). Sur des données tabulaires de cette taille, un réseau n'est pas magique : il fait jeu égal avec les bons modèles classiques.
- La durée exacte dépend de la machine ; on vérifie seulement qu'elle reste courte.

### Sortie attendue

```text output
epoch  1: train loss=0.660  val loss=0.654  val accuracy=0.600
epoch 10: train loss=0.521  val loss=0.557  val accuracy=0.793
epoch 30: train loss=0.345  val loss=0.405  val accuracy=0.860
epoch 60: train loss=0.263  val loss=0.351  val accuracy=0.847
training took less than 30 s on CPU: True
```

### Exemple 3 — comparer des optimiseurs, sauvegarder et recharger

```python
from pathlib import Path

for name, make_optimizer in [
    ("SGD lr=0.01", lambda p: torch.optim.SGD(p, lr=0.01)),
    ("SGD lr=0.1 + momentum", lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)),
    ("Adam lr=1e-3", lambda p: torch.optim.Adam(p, lr=1e-3)),
]:
    torch.manual_seed(0)
    candidate = make_model()
    opt = make_optimizer(candidate.parameters())
    for _ in range(20):
        train_one_epoch(candidate, train_loader, loss_fn, opt)
    print(f"{name:<22} after 20 epochs: val loss={evaluate(candidate, X_val_t, y_val_t, loss_fn)[0]:.3f}")

model_dir = Path("outputs/module06")
model_dir.mkdir(parents=True, exist_ok=True)
torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std}, model_dir / "binary_mlp.pt")

checkpoint = torch.load(model_dir / "binary_mlp.pt", weights_only=False)
restored = make_model()
restored.load_state_dict(checkpoint["state_dict"])
restored.eval()
with torch.no_grad():
    same = torch.allclose(restored(X_val_t), model(X_val_t))
    new_point = torch.from_numpy(((X[:1] - checkpoint["mean"]) / checkpoint["std"]).astype(np.float32))
    print("identical predictions after reload:", same, "| probability for first row:", round(torch.sigmoid(restored(new_point)).item(), 3))
```

### Explication du code

- Avec la même initialisation (`manual_seed(0)`), SGD à petit taux progresse très lentement. Avec un moment et un taux plus élevé, SGD fait même mieux qu'Adam après 20 époques sur ce problème. Adam reste un excellent point de départ car il fonctionne correctement sans réglage ; mais aucun optimiseur ne gagne partout, et le taux d'apprentissage compte souvent plus que le choix de l'optimiseur.
- On sauvegarde un dictionnaire contenant les poids **et** les statistiques de standardisation. Comme il contient des tableaux NumPy, on le recharge avec `weights_only=False` : à réserver à vos propres fichiers (pour un fichier d'origine inconnue, ne sauvegardez/chargez que des tenseurs avec `weights_only=True`).
- Le modèle rechargé donne exactement les mêmes sorties.

### Sortie attendue

```text output
SGD lr=0.01            after 20 epochs: val loss=0.638
SGD lr=0.1 + momentum  after 20 epochs: val loss=0.393
Adam lr=1e-3           after 20 epochs: val loss=0.461
identical predictions after reload: True | probability for first row: 0.086
```

## Erreurs fréquentes

- **Oublier `optimizer.zero_grad()`** : accumulation des gradients.
- **Appliquer une sigmoïde avant `BCEWithLogitsLoss`** : double sigmoïde, apprentissage dégradé.
- **Étiquettes de mauvaise forme ou de mauvais type** : `(n,)` au lieu de `(n, 1)`, `int64` au lieu de `float32`.
- **Oublier `model.eval()` et `torch.no_grad()`** à l'évaluation.
- **Données non standardisées** : convergence lente ou instable.
- **Sauvegarder le modèle sans le prétraitement** : prédictions fausses après rechargement.
- **Évaluer sur l'entraînement** pour décider quand arrêter.

## Exercices guidés

1. Calculez à la main le nombre de paramètres d'un réseau 10 → 64 → 64 → 3.
2. Entraînez le modèle avec des tailles de lot de 8, 32 et 256 pendant 30 époques. Comparez la perte de validation et le temps.
3. Remplacez Adam par SGD sans moment avec `lr=1.0`. Que se passe-t-il ?
4. Calculez précision, rappel et F1 du modèle final sur le jeu de validation (réutilisez `sklearn.metrics`).

Corrigés : [corrections.md](corrections.md), section « Leçon 6.3 ».

## Défis autonomes

- Réécrivez le modèle sous forme de classe `class BinaryMLP(nn.Module)` avec une méthode `forward`.
- Tracez les courbes d'apprentissage (perte d'entraînement et de validation par époque) pour 200 époques. Où commence le surapprentissage ?
- Exercice **A02** : une fonction `fit` réutilisable avec historique et arrêt précoce.

## Quiz

1. Pourquoi la dernière couche renvoie-t-elle un logit et non une probabilité ?
2. Quelles sont les quatre étapes d'une itération d'entraînement ?
3. Qu'est-ce qu'une époque ? Un mini-lot ?
4. Que font `model.train()` et `model.eval()` ?
5. Combien de paramètres a `nn.Linear(100, 10)` ?
6. Que faut-il sauvegarder pour réutiliser un modèle ?

Réponses : [corrections.md](corrections.md), section « Leçon 6.3 ».

## Résumé

- `nn.Module` décrit le réseau ; la perte et l'optimiseur pilotent l'apprentissage.
- `zero_grad` → `forward` + perte → `backward` → `step`, pour chaque mini-lot.
- Validation à chaque époque, en mode évaluation et sans gradient.
- Sauvegarder `state_dict` et prétraitement ; recharger dans la même architecture.

## Pour aller plus loin

- Exercice **A02** (`exercises/advanced/A02_pytorch_training_loop`).
- Leçon suivante : [Classification multiclasse et régularisation](04_multiclasse_regularisation.md).
