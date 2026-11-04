# Corrigés — Module 6 : Deep Learning avec PyTorch

## Leçon 6.1

### Quiz

1. Un tenseur peut être placé sur GPU et enregistrer les opérations qui le produisent pour calculer automatiquement des gradients.
2. `float32` : deux fois moins de mémoire et des calculs plus rapides que `float64`, avec une précision suffisante pour l'apprentissage.
3. Il parcourt le graphe de calcul de la perte vers les paramètres et stocke dans `.grad` la dérivée de la perte par rapport à chaque tenseur feuille qui exige un gradient.
4. Parce que PyTorch additionne les nouveaux gradients aux anciens.
5. À exécuter des calculs sans construire de graphe : mises à jour des paramètres, évaluation, prédiction (moins de mémoire, plus rapide).
6. Si le tenseur exige un gradient (il faut `.detach()`) ou s'il est sur GPU (il faut `.cpu()`).

### Exercices guidés

```python
import torch

m = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
print(m.T, m.sum(dim=1), m @ m, sep="\n")

x = torch.tensor(2.0, requires_grad=True)
(x**3 - 2 * x).backward()
print("autograd:", x.grad.item(), "| formula 3x^2 - 2:", 3 * 2**2 - 2)

w = torch.tensor([1.0, 2.0], requires_grad=True)
g = w[0] ** 2 + 3 * w[1] ** 2 + w[0] * w[1]
g.backward()
print("gradient at (1, 2):", w.grad.tolist())

import pandas as pd

houses = pd.read_csv("datasets/generated/house_prices.csv")
X = torch.tensor(houses[["surface_m2", "distance_center_km"]].to_numpy(), dtype=torch.float32)
y = torch.tensor(houses["price_keur"].to_numpy(), dtype=torch.float32)
X = (X - X.mean(dim=0)) / X.std(dim=0)
weights = torch.zeros(2, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)
for _ in range(200):
    loss = ((X @ weights + bias - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        weights -= 0.1 * weights.grad
        bias -= 0.1 * bias.grad
    weights.grad, bias.grad = None, None
print("standardized weights:", [round(v, 2) for v in weights.detach().tolist()], "| RMSE:", round(loss.sqrt().item(), 1))
```

```text output
tensor([[1., 4., 7.],
        [2., 5., 8.],
        [3., 6., 9.]])
tensor([ 6., 15., 24.])
tensor([[ 30.,  36.,  42.],
        [ 66.,  81.,  96.],
        [102., 126., 150.]])
autograd: 10.0 | formula 3x^2 - 2: 10
gradient at (1, 2): [4.0, 13.0]
standardized weights: [111.5, -64.67] | RMSE: 66.1
```

Exercice 3 : on retrouve [4, 13], comme avec SymPy dans la leçon 2.5. Exercice 4 : la distance au centre a un poids négatif ; ajouter cette variable réduit l'erreur par rapport à la surface seule.

## Leçon 6.2

### Quiz

1. a = f(w·x + b) : l'activation appliquée à la somme pondérée des entrées plus un biais.
2. Sa frontière de décision est une droite, et aucune droite ne sépare deux croissants imbriqués.
3. Le réseau devient une seule fonction linéaire, quelle que soit sa profondeur.
4. Tous les neurones d'une couche resteraient identiques (même calcul, même gradient) : symétrie jamais brisée.
5. 1 si z > 0, 0 sinon (non définie en 0 ; on prend 0 en pratique).
6. En comparant les gradients calculés aux différences finies (L(θ + h) − L(θ − h)) / 2h sur quelques paramètres.

### Exercices guidés

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

z = 0.5 * 2 + (-1) * 1 + 0.2
print("neuron output:", round(1 / (1 + np.exp(-z)), 4))

X, y = make_moons(n_samples=600, noise=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
y_col = y_train.reshape(-1, 1).astype(float)
sigmoid = lambda t: 1 / (1 + np.exp(-t))


def train(n_hidden, activation="relu", epochs=3000, lr=0.5, zero_init=False, seed=1):
    rng = np.random.default_rng(seed)
    W1 = np.zeros((2, n_hidden)) if zero_init else rng.normal(0, np.sqrt(2 / 2), size=(2, n_hidden))
    W2 = np.zeros((n_hidden, 1)) if zero_init else rng.normal(0, np.sqrt(2 / n_hidden), size=(n_hidden, 1))
    b1, b2 = np.zeros(n_hidden), np.zeros(1)
    for _ in range(epochs):
        Z1 = X_train @ W1 + b1
        H = np.maximum(0, Z1) if activation == "relu" else np.tanh(Z1)
        P = sigmoid(H @ W2 + b2)
        dZ2 = (P - y_col) / len(X_train)
        dH = dZ2 @ W2.T
        dZ1 = dH * ((Z1 > 0) if activation == "relu" else (1 - H**2))
        W2 -= lr * H.T @ dZ2
        b2 -= lr * dZ2.sum(axis=0)
        W1 -= lr * X_train.T @ dZ1
        b1 -= lr * dZ1.sum(axis=0)
    H = np.maximum(0, X_test @ W1 + b1) if activation == "relu" else np.tanh(X_test @ W1 + b1)
    return ((sigmoid(H @ W2 + b2) >= 0.5).ravel() == y_test).mean(), W1


print("tanh, 16 hidden:", round(train(16, "tanh")[0], 3))
for n_hidden in [2, 4, 16, 64]:
    print(f"relu, {n_hidden:>2} hidden: test accuracy = {train(n_hidden)[0]:.3f}")
accuracy, W1 = train(16, zero_init=True)
print("zero init accuracy:", round(accuracy, 3), "| identical columns:", bool(np.allclose(W1, W1[:, :1])))
```

```text output
neuron output: 0.5498
tanh, 16 hidden: 0.953
relu,  2 hidden: test accuracy = 0.867
relu,  4 hidden: test accuracy = 0.947
relu, 16 hidden: test accuracy = 0.947
relu, 64 hidden: test accuracy = 0.947
zero init accuracy: 0.493 | identical columns: True
```

Exercice 4 : avec une initialisation nulle, toutes les colonnes de W1 restent identiques (ici elles restent même à 0, car le gradient de ReLU en 0 est nul) : le réseau ne fait pas mieux qu'une prédiction constante.

## Leçon 6.3

### Quiz

1. Parce que `BCEWithLogitsLoss` applique elle-même la sigmoïde de façon numériquement stable ; on applique `torch.sigmoid` seulement pour obtenir des probabilités à la prédiction.
2. `optimizer.zero_grad()`, calcul de la sortie et de la perte, `loss.backward()`, `optimizer.step()`.
3. Une époque est un passage complet sur le jeu d'entraînement ; un mini-lot est le sous-ensemble d'exemples utilisé pour une mise à jour.
4. Ils changent le comportement des couches comme le dropout et la normalisation par lots (actives en entraînement, figées en évaluation).
5. 100 × 10 + 10 = 1010.
6. Les poids (`state_dict`), l'architecture (le code qui la recrée) et tout le prétraitement (moyennes, écarts-types, vocabulaire...).

### Exercices guidés

```python
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

print("parameters of 10 -> 64 -> 64 -> 3:", (10 * 64 + 64) + (64 * 64 + 64) + (64 * 3 + 3))

df = pd.read_csv("datasets/generated/binary_classification.csv")
X = df.drop(columns="target").to_numpy(dtype=np.float32)
y = df["target"].to_numpy(dtype=np.float32).reshape(-1, 1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train_t, X_val_t = torch.from_numpy((X_train - mean) / std), torch.from_numpy((X_val - mean) / std)
y_train_t, y_val_t = torch.from_numpy(y_train), torch.from_numpy(y_val)
loss_fn = nn.BCEWithLogitsLoss()


def run(batch_size, make_optimizer, epochs=30):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
    optimizer = make_optimizer(model.parameters())
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return model, loss_fn(model(X_val_t), y_val_t).item()


for batch_size in [8, 32, 256]:
    start = time.perf_counter()
    _, val_loss = run(batch_size, lambda p: torch.optim.Adam(p, lr=1e-3))
    seconds = time.perf_counter() - start  # varies between computers, so it is not printed here
    print(f"batch {batch_size:>3}: val loss={val_loss:.3f}")

_, val_loss = run(32, lambda p: torch.optim.SGD(p, lr=1.0))
print("SGD lr=1.0: val loss =", round(val_loss, 3))

model, _ = run(32, lambda p: torch.optim.Adam(p, lr=1e-3), epochs=60)
with torch.no_grad():
    pred = (model(X_val_t) > 0).int().numpy().ravel()
print("precision", round(precision_score(y_val.ravel(), pred), 3), "| recall", round(recall_score(y_val.ravel(), pred), 3), "| F1", round(f1_score(y_val.ravel(), pred), 3))
```

```text output
parameters of 10 -> 64 -> 64 -> 3: 5059
batch   8: val loss=0.334
batch  32: val loss=0.405
batch 256: val loss=0.585
SGD lr=1.0: val loss = 0.255
precision 0.836 | recall 0.767 | F1 0.8
```

Exercice 2 : avec des petits lots, il y a beaucoup plus de mises à jour par époque (convergence plus rapide en nombre d'époques, mais chaque époque est plus lente) ; avec 256, seulement 2 mises à jour par époque et 30 époques ne suffisent pas. Exercice 3 : contrairement à ce que l'on pourrait craindre, SGD avec `lr=1.0` converge ici et obtient même la meilleure perte de validation : le réseau est petit, les entrées sont standardisées et la perte varie doucement. Sur d'autres problèmes (réseaux profonds, entrées non standardisées), un taux aussi élevé diverge. La leçon : on ne devine pas un bon taux d'apprentissage, on l'essaie et on mesure sur la validation.

## Leçon 6.4

### Quiz

1. Des probabilités positives dont la somme vaut 1, et l'ordre des logits est conservé.
2. Des entiers `int64` (indices des classes) de forme (n,).
3. La perte d'entraînement continue de baisser alors que la perte de validation stagne puis remonte.
4. Pour que les prédictions soient déterministes et utilisent tout le réseau (avec la bonne mise à l'échelle).
5. Elle standardise les activations de chaque neurone sur le mini-lot puis applique une échelle et un décalage appris ; cela stabilise et accélère l'entraînement.
6. Parce que les poids de la dernière époque sont ceux d'un modèle qui a déjà commencé à surapprendre.

### Exercices guidés

```python
import numpy as np
import torch
from torch import nn

z = np.array([1.0, 2.0, 3.0])
print("by hand:", (np.exp(z) / np.exp(z).sum()).round(4), "| torch:", torch.softmax(torch.tensor(z), dim=0).numpy().round(4))

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Dropout(0.5), nn.Linear(16, 2))
x = torch.randn(3, 4)
model.train()
print("train mode, two calls equal:", torch.equal(model(x), model(x)))
model.eval()
print("eval mode, two calls equal:", torch.equal(model(x), model(x)))
```

```text output
by hand: [0.09   0.2447 0.6652] | torch: [0.09   0.2447 0.6652]
train mode, two calls equal: False
eval mode, two calls equal: True
```

Exercices 2 et 3 : avec davantage de données d'entraînement, l'écart entre perte d'entraînement et de validation se réduit nettement : la quantité de données est le meilleur régularisateur. Un dropout de 0,8 désactive tant de neurones que le réseau sous-apprend (pertes d'entraînement et de validation élevées). Le code est celui de l'exemple 2 de la leçon, avec `train_size` et `dropout` modifiés.
