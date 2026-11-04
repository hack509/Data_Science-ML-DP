# Corrigés — Module 7 : Réseaux de neurones

## Leçon 7.1

### Quiz

1. Ses classes ne sont pas linéairement séparables : aucune droite ne place (0,1) et (1,0) d'un côté et (0,0), (1,1) de l'autre.
2. Le perceptron utilise une fonction seuil et ne corrige que les erreurs ; la régression logistique utilise la sigmoïde, produit des probabilités et minimise l'entropie croisée par descente de gradient.
3. Softmax (multiclasse) ; identité, c'est-à-dire aucune activation (régression).
4. Sa dérivée vaut 1 pour z > 0 : les gradients ne s'atténuent pas en traversant les couches, contrairement à la sigmoïde (dérivée ≤ 0,25) qui sature.
5. L'entropie croisée est la log-vraisemblance négative moyenne : la minimiser revient à maximiser la probabilité des données observées.
6. Qu'un MLP à une couche cachée assez large **peut** approcher toute fonction continue sur un domaine borné ; il ne dit ni combien de neurones il faut, ni qu'on trouvera ces poids par entraînement, ni que le modèle généralisera.

### Exercices guidés

```python
import math

import torch
from torch import nn

# 1. Two perceptron updates on AND, starting from w=(0,0), b=0, lr=1
w, b = [0.0, 0.0], 0.0
for x, target in [((0, 0), 0), ((0, 1), 0), ((1, 0), 0), ((1, 1), 1)]:
    prediction = int(w[0] * x[0] + w[1] * x[1] + b > 0)
    if prediction != target:
        w = [w[0] + (target - prediction) * x[0], w[1] + (target - prediction) * x[1]]
        b += target - prediction
    print(f"x={x} target={target} prediction={prediction} -> w={w} b={b}")

# 2. More activations on XOR
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
for name, activation in [("LeakyReLU", nn.LeakyReLU(0.1)), ("GELU", nn.GELU())]:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 8), activation, nn.Linear(8, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(500):
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(X), y)
        loss.backward()
        optimizer.step()
    print(name, "loss", round(loss.item(), 4), "predictions", (model(X) > 0).int().ravel().tolist())

# 3. Maximum derivative of the sigmoid and the product over 10 layers
print("max sigmoid derivative:", 0.25, "| product over 10 layers:", 0.25**10)

# 4. Cross-entropy for a class-1 example
for p in (0.9, 0.6, 0.1):
    print(f"p={p}: loss={-math.log(p):.3f}")
```

```text output
x=(0, 0) target=0 prediction=0 -> w=[0.0, 0.0] b=0.0
x=(0, 1) target=0 prediction=0 -> w=[0.0, 0.0] b=0.0
x=(1, 0) target=0 prediction=0 -> w=[0.0, 0.0] b=0.0
x=(1, 1) target=1 prediction=0 -> w=[1.0, 1.0] b=1.0
LeakyReLU loss 0.0001 predictions [0, 1, 1, 0]
GELU loss 0.0001 predictions [0, 1, 1, 0]
max sigmoid derivative: 0.25 | product over 10 layers: 9.5367431640625e-07
p=0.9: loss=0.105
p=0.6: loss=0.511
p=0.1: loss=2.303
```

Exercice 1 : au premier passage, les trois premiers exemples sont bien classés (sortie 0) et seul (1, 1) déclenche une mise à jour. Exercice 3 : 0,25¹⁰ ≈ 10⁻⁶ ; le gradient qui atteint la première couche peut être un million de fois plus petit qu'en sortie, d'où un apprentissage quasi nul des premières couches.

## Leçon 7.2

### Quiz

1. Elles diminuent à chaque couche et deviennent quasi nulles ; les gradients aussi, et le réseau n'apprend pas.
2. He (Kaiming) pour ReLU ; Xavier (Glorot) pour tanh.
3. La dérivée de la sigmoïde est au plus 0,25 ; la rétropropagation multiplie ces dérivées couche après couche, et le gradient des premières couches s'évanouit.
4. log(4) ≈ 1,386.
5. Il vérifie en quelques secondes que modèle, perte, optimiseur et données sont correctement branchés : un réseau sain doit pouvoir mémoriser 16 exemples.
6. Si collecter davantage de données améliorerait la performance de validation, ou si le modèle a atteint un plateau.

### Exercices guidés

```python
import math

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn

print("He std for Linear(256, 256):", round(math.sqrt(2 / 256), 4), "| Xavier std for Linear(256, 128):", round(math.sqrt(2 / (256 + 128)), 4))


def gradient_ratio(activation, depth=10, width=64):
    torch.manual_seed(0)
    layers = [nn.Linear(10, width), activation()]
    for _ in range(depth):
        layers += [nn.Linear(width, width), activation()]
    model = nn.Sequential(*layers, nn.Linear(width, 1))
    nn.functional.mse_loss(model(torch.randn(256, 10)), torch.randn(256, 1)).backward()
    norms = [m.weight.grad.norm().item() for m in model if isinstance(m, nn.Linear)]
    return norms[0] / norms[-1]


print("first/last gradient ratio: tanh", f"{gradient_ratio(nn.Tanh):.1e}", "| sigmoid", f"{gradient_ratio(nn.Sigmoid):.1e}")

df = pd.read_csv("datasets/generated/multiclass_classification.csv")
X = torch.tensor(df.drop(columns="target").to_numpy(), dtype=torch.float32)
X = (X - X.mean(dim=0)) / X.std(dim=0)
y = torch.tensor(df["target"].to_numpy(), dtype=torch.int64)
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 3), nn.Softmax(dim=1))  # bug: softmax before CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
for _ in range(200):
    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(model(X[:16]), y[:16])
    loss.backward()
    optimizer.step()
print("one-batch test with the softmax bug: loss =", round(loss.item(), 3), "(should be close to 0, cannot go below about 0.55)")

X_np, y_np = X.numpy(), y.numpy()
X_pool, X_val, y_pool, y_val = train_test_split(X_np, y_np, test_size=0.3, stratify=y_np, random_state=0)
for n in [50, 100, 200, 400, 600]:
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 3))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    xs, ys = torch.from_numpy(X_pool[:n]), torch.from_numpy(y_pool[:n])
    for _ in range(200):
        opt.zero_grad()
        nn.functional.cross_entropy(net(xs), ys).backward()
        opt.step()
    with torch.no_grad():
        accuracy = (net(torch.from_numpy(X_val)).argmax(1).numpy() == y_val).mean()
    print(f"n_train={n:>3}: validation accuracy={accuracy:.3f}")
```

```text output
He std for Linear(256, 256): 0.0884 | Xavier std for Linear(256, 128): 0.0722
first/last gradient ratio: tanh 5.5e-03 | sigmoid 1.5e-10
one-batch test with the softmax bug: loss = 0.552 (should be close to 0, cannot go below about 0.55)
n_train= 50: validation accuracy=0.863
n_train=100: validation accuracy=0.893
n_train=200: validation accuracy=0.922
n_train=400: validation accuracy=0.922
n_train=600: validation accuracy=0.930
```

Exercice 2 : avec tanh, le rapport (≈ 5·10⁻³) est bien meilleur qu'avec la sigmoïde (dérivée maximale 1 au lieu de 0,25), et même meilleur que celui de ReLU dans l'exemple de la leçon (≈ 5·10⁻⁴) : avec l'initialisation par défaut de PyTorch, les activations restent petites, dans la zone presque linéaire de tanh où sa dérivée vaut presque 1. Le défaut de tanh apparaît quand les activations saturent (poids trop grands, exemple 1 de la leçon). Retenez qu'il faut mesurer : les règles générales (« ReLU est meilleure ») ont des exceptions. Exercice 3 : avec un softmax en trop, `CrossEntropyLoss` applique un second softmax à des valeurs comprises entre 0 et 1 ; la perte ne peut plus descendre près de 0 et le test du lot unique révèle le bug. Exercice 4 : la précision de validation progresse avec la taille du jeu d'entraînement puis plafonne : au-delà de quelques centaines d'exemples, plus de données apporterait peu sur ce problème.

## Leçon 7.3

### Quiz

1. Pour mesurer la variabilité due à l'initialisation et à l'ordre des lots, et ne pas prendre ce bruit pour un effet de l'architecture.
2. Que chaque variante dispose des mêmes ressources d'entraînement (nombre d'époques ou de mises à jour, ou temps) et de la même règle de sélection.
3. L'écart (0,006) est plus petit que les écarts-types : on ne peut pas conclure qu'une architecture est meilleure.
4. Parce que l'effet du taux d'apprentissage est multiplicatif : passer de 0,001 à 0,01 compte autant que de 0,01 à 0,1.
5. Une expérience où l'on retire un composant à la fois (dropout, une couche, une variable...) pour mesurer sa contribution.
6. Pour chaque essai : date, version du code, données et découpage, architecture, hyperparamètres, graine, budget, métriques (moyenne et dispersion), temps de calcul.

### Exercices guidés

```python
print("parameters of 2 x 64:", (5 * 64 + 64) + (64 * 64 + 64) + (64 * 3 + 3))
```

```text output
parameters of 2 x 64: 4739
```

Exercices 2 à 4 : le code est celui des exemples 1 et 2 de la leçon, en ajoutant `"1 x 16": [16]` au dictionnaire des architectures, `(0, 1, 2, 3, 4)` comme graines, et une boucle sur `batch_size` dans `run(..., batch_size=...)`. Sur ces données presque linéairement séparables, aucune de ces variations ne fait apparaître de différence supérieure à l'écart-type entre graines : la conclusion de la leçon ne change pas.
