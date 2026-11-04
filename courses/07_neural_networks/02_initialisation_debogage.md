# Leçon 7.2 — Initialisation, courbes d'apprentissage et débogage

## Prérequis

- Leçon 7.1 ; leçons 6.3 et 6.4.

## Objectifs d'apprentissage

- Comprendre pourquoi l'échelle des poids initiaux détermine si un réseau profond peut apprendre.
- Utiliser les initialisations de Xavier/Glorot et de He/Kaiming.
- Diagnostiquer gradients qui s'évanouissent ou explosent en mesurant les normes de gradient par couche.
- Lire les courbes d'apprentissage (en fonction des époques et de la quantité de données).
- Appliquer une méthode de débogage systématique, dont le test « surapprendre un seul lot ».

## Concepts clés

- **Initialisation** (*weight initialization*) : valeurs des poids avant l'entraînement.
- **Xavier / Glorot** : variance 2 / (n_entrées + n_sorties), adaptée à tanh et sigmoïde.
- **He / Kaiming** : variance 2 / n_entrées, adaptée à ReLU (défaut conseillé ; PyTorch utilise par défaut une initialisation uniforme de la famille Kaiming).
- **Gradient évanescent / explosif** (*vanishing / exploding gradient*).
- **Norme du gradient** (*gradient norm*) : taille du vecteur de gradient d'une couche.
- **Écrêtage du gradient** (*gradient clipping*) : limiter la norme du gradient.
- **Courbe d'apprentissage** (*learning curve*) : performance en fonction des époques ou de la taille du jeu d'entraînement.

## Explication détaillée

### Pourquoi l'initialisation compte

À chaque couche, l'activation est h = f(hW + b). Si les poids sont trop petits, la variance des activations est multipliée à chaque couche par un facteur inférieur à 1 : après 20 couches, les activations sont quasi nulles, et les gradients aussi. Trop grands, et tout explose (ou, avec tanh et sigmoïde, sature, ce qui annule aussi les gradients). L'idée des initialisations de Xavier et de He est de choisir la variance des poids pour que la variance des activations **reste constante** d'une couche à l'autre : pour ReLU, qui annule la moitié des valeurs, il faut une variance double (2 / n_entrées).

### Diagnostiquer les gradients

Après un `loss.backward()`, on peut afficher la norme du gradient de chaque couche : `layer.weight.grad.norm()`. Dans un réseau sain, les normes restent du même ordre de grandeur d'une couche à l'autre. Si elles décroissent de plusieurs ordres de grandeur en remontant vers l'entrée, les premières couches n'apprennent presque pas (gradient évanescent). Remèdes : ReLU, bonne initialisation, normalisation (BatchNorm, LayerNorm), connexions résiduelles (on ajoute l'entrée d'un bloc à sa sortie : y = x + f(x)). Si les normes explosent (`inf`, `nan`) : taux d'apprentissage plus faible, écrêtage du gradient (`torch.nn.utils.clip_grad_norm_`), normalisation.

### Lire les courbes d'apprentissage

| Ce que l'on voit | Diagnostic | Action |
|---|---|---|
| perte d'entraînement qui ne baisse pas | bug, taux trop faible/fort, données mal préparées | méthode de débogage ci-dessous |
| entraînement et validation élevés et proches | sous-apprentissage | modèle plus grand, plus d'époques, meilleures variables |
| entraînement bas, validation qui remonte | surapprentissage | régularisation, arrêt précoce, plus de données |
| courbes très bruitées | taux trop élevé, lots trop petits | baisser le taux, augmenter la taille de lot |
| `nan` soudain | explosion numérique | taux plus faible, écrêtage, vérifier log(0) et divisions |

La **courbe d'apprentissage en fonction de la taille des données** répond à une question coûteuse : « collecter plus de données aiderait-il ? » Si l'erreur de validation baisse encore nettement quand la taille augmente, oui ; si elle a atteint un plateau proche de l'erreur d'entraînement, non (il faut un meilleur modèle ou de meilleures variables).

### Méthode de débogage systématique

1. **Vérifier les données** : formes, types, étiquettes alignées avec les entrées, valeurs standardisées, pas de `nan`. Affichez quelques exemples.
2. **Vérifier la perte initiale** : pour K classes équilibrées et une initialisation correcte, la perte au départ doit être proche de log(K) (0,693 pour 2 classes, 1,099 pour 3).
3. **Surapprendre un seul petit lot** : un réseau correct doit atteindre une perte quasi nulle sur 16 exemples en quelques centaines d'itérations. S'il n'y arrive pas, il y a un bug (perte mal choisie, `zero_grad` oublié, activation manquante, étiquettes mélangées...).
4. **Balayer le taux d'apprentissage** sur quelques époques (10⁻⁴ à 1) et choisir la zone où la perte baisse le plus vite sans exploser.
5. **Seulement ensuite**, entraîner sur tout le jeu, surveiller la validation et régulariser.
6. **Changer une seule chose à la fois** et garder une trace de chaque essai (graine, hyperparamètres, résultats).

## Intuition mathématique

```text
Var(sortie d'une couche linéaire) ≈ n_entrées · Var(w) · Var(entrée)
Xavier : Var(w) = 2 / (n_in + n_out)      He : Var(w) = 2 / n_in   (ReLU garde la moitié de la variance)
gradient de la couche 1 ∝ Π_{l=2..L} (Wˡ · f'(zˡ))   → un produit de L facteurs : évanescent si < 1, explosif si > 1
perte initiale attendue (K classes équilibrées, sorties ≈ uniformes) = −log(1/K) = log K
```

## Applications concrètes

- Toute mise au point d'un réseau : la majorité des échecs viennent de bugs de données ou de réglages, pas de l'architecture.
- Décider d'investir dans la collecte de données grâce aux courbes d'apprentissage en fonction de la taille.

## Exemples de code Python

### Exemple 1 — l'effet de l'initialisation sur un réseau de 20 couches

```python
import torch
from torch import nn

torch.manual_seed(0)
x = torch.randn(512, 256)


def activation_stds(init_std, activation, depth=20, width=256):
    h, stds = x, []
    for _ in range(depth):
        layer = nn.Linear(width, width, bias=False)
        if init_std == "he":
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        elif init_std == "xavier":
            nn.init.xavier_normal_(layer.weight)
        else:
            nn.init.normal_(layer.weight, std=init_std)
        with torch.no_grad():
            h = activation(layer(h))
        stds.append(h.std().item())
    return stds


for label, init, act in [
    ("std=0.01 + ReLU", 0.01, torch.relu),
    ("std=1.0 + ReLU", 1.0, torch.relu),
    ("He + ReLU", "he", torch.relu),
    ("std=1.0 + tanh", 1.0, torch.tanh),
    ("Xavier + tanh", "xavier", torch.tanh),
]:
    stds = activation_stds(init, act)
    print(f"{label:<16} activation std at layers 1, 10, 20: {stds[0]:.3g}, {stds[9]:.3g}, {stds[19]:.3g}")
```

### Explication du code

On fait traverser à 512 vecteurs aléatoires 20 couches de largeur 256, sans entraînement, et on mesure l'écart-type des activations. Avec des poids trop petits (0,01), le signal disparaît en quelques couches ; trop grands (1,0), il explose avec ReLU (facteur ≈ √128 par couche). Avec tanh et des poids trop grands, l'écart-type semble stable (≈ 0,97) : c'est un piège, les activations sont **saturées** à ±1, là où la dérivée de tanh est presque nulle. L'initialisation de He garde un signal stable sur 20 couches ReLU. Xavier garde tanh dans sa zone utile, avec une lente décroissance (0,63 → 0,16), bien plus douce qu'avec des poids trop petits : c'est l'une des raisons pour lesquelles ReLU + He est le choix par défaut des réseaux profonds.

### Sortie attendue

```text output
std=0.01 + ReLU  activation std at layers 1, 10, 20: 0.0933, 2.71e-10, 1.03e-19
std=1.0 + ReLU   activation std at layers 1, 10, 20: 9.39, 3.05e+10, 1e+21
He + ReLU        activation std at layers 1, 10, 20: 0.824, 0.783, 0.92
std=1.0 + tanh   activation std at layers 1, 10, 20: 0.975, 0.974, 0.974
Xavier + tanh    activation std at layers 1, 10, 20: 0.626, 0.228, 0.162
```

### Exemple 2 — gradients par couche : sigmoïde contre ReLU

```python
def gradient_norms(activation, depth=10, width=64):
    torch.manual_seed(0)
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(width, width), activation()]
    model = nn.Sequential(nn.Linear(10, width), activation(), *layers, nn.Linear(width, 1))
    inputs, targets = torch.randn(256, 10), torch.randn(256, 1)
    nn.functional.mse_loss(model(inputs), targets).backward()
    linear_layers = [m for m in model if isinstance(m, nn.Linear)]
    return [layer.weight.grad.norm().item() for layer in linear_layers]


for name, activation in [("sigmoid", nn.Sigmoid), ("ReLU", nn.ReLU)]:
    norms = gradient_norms(activation)
    print(f"{name:<8} gradient norm: first layer={norms[0]:.2e}  middle={norms[len(norms) // 2]:.2e}  last={norms[-1]:.2e}  ratio first/last={norms[0] / norms[-1]:.1e}")
```

### Explication du code

Après une seule rétropropagation, on compare la norme du gradient de la première couche (près de l'entrée) à celle de la dernière. Avec 11 couches sigmoïdes, le gradient de la première couche est plusieurs ordres de grandeur plus petit que celui de la dernière : les premières couches n'apprennent quasiment pas. Avec ReLU (et l'initialisation par défaut de PyTorch, un peu plus petite que He), le rapport est meilleur d'environ six ordres de grandeur, même si une atténuation subsiste : c'est ce que corrigent la normalisation et les connexions résiduelles.

### Sortie attendue

```text output
sigmoid  gradient norm: first layer=1.36e-10  middle=3.16e-05  last=5.85e-01  ratio first/last=2.3e-10
ReLU     gradient norm: first layer=1.71e-05  middle=3.92e-04  last=3.16e-02  ratio first/last=5.4e-04
```

### Exemple 3 — débogage : perte initiale, surapprendre un lot, balayage du taux

```python
import math

import numpy as np
import pandas as pd

df = pd.read_csv("datasets/generated/multiclass_classification.csv")
X = torch.tensor(df.drop(columns="target").to_numpy(), dtype=torch.float32)
X = (X - X.mean(dim=0)) / X.std(dim=0)
y = torch.tensor(df["target"].to_numpy(), dtype=torch.int64)


def make_model():
    return nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))


torch.manual_seed(0)
model = make_model()
loss_fn = nn.CrossEntropyLoss()
with torch.no_grad():
    print(f"initial loss={loss_fn(model(X), y).item():.3f}  expected about log(3)={math.log(3):.3f}")

batch_x, batch_y = X[:16], y[:16]
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
for step in range(200):
    optimizer.zero_grad()
    loss = loss_fn(model(batch_x), batch_y)
    loss.backward()
    optimizer.step()
print(f"overfit one batch of 16: loss after 200 steps={loss.item():.4f}")

# The same test with a classic bug: labels shuffled, so they no longer match the inputs.
torch.manual_seed(0)
buggy = make_model()
optimizer = torch.optim.Adam(buggy.parameters(), lr=1e-2)
for step in range(200):
    optimizer.zero_grad()
    loss = loss_fn(buggy(X), y[torch.randperm(len(y))])  # bug: new random labels at every step
    loss.backward()
    optimizer.step()
print(f"with shuffled labels (bug): loss after 200 steps={loss.item():.3f}")

for lr in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
    torch.manual_seed(0)
    candidate = make_model()
    opt = torch.optim.Adam(candidate.parameters(), lr=lr)
    for _ in range(50):
        opt.zero_grad()
        loss = loss_fn(candidate(X), y)
        loss.backward()
        opt.step()
    print(f"lr={lr:<7} loss after 50 full-batch steps={loss.item():.3f}")
```

### Explication du code

- La perte initiale est proche de log(3) ≈ 1,099 : sorties presque uniformes, initialisation correcte.
- Sur un seul lot de 16 exemples, la perte tombe presque à zéro : le modèle, la perte et l'optimiseur fonctionnent.
- Avec des étiquettes mélangées à chaque pas (un bug réaliste : indices de mélange différents pour X et y), la perte reste bloquée vers log(3) : le réseau ne peut rien apprendre. Ce contraste est exactement ce que le test révèle.
- Le balayage du taux d'apprentissage montre une zone efficace (10⁻² à 10⁻¹ ici) ; à 10⁻⁴ l'apprentissage est trop lent, et pour des taux très élevés la perte devient instable ou remonte.

### Sortie attendue

```text output
initial loss=1.130  expected about log(3)=1.099
overfit one batch of 16: loss after 200 steps=0.0000
with shuffled labels (bug): loss after 200 steps=1.098
lr=0.0001  loss after 50 full-batch steps=1.056
lr=0.001   loss after 50 full-batch steps=0.392
lr=0.01    loss after 50 full-batch steps=0.142
lr=0.1     loss after 50 full-batch steps=0.106
lr=1.0     loss after 50 full-batch steps=0.830
lr=10.0    loss after 50 full-batch steps=14.045
```

## Erreurs fréquentes

- **Changer plusieurs réglages à la fois** : impossible de savoir ce qui a eu un effet.
- **Initialiser à la main avec une échelle arbitraire** (par exemple `randn` sans mise à l'échelle).
- **Ignorer une perte initiale anormale** (bien supérieure à log K) : souvent des sorties trop grandes ou des données non standardisées.
- **Entraîner longtemps avant d'avoir vérifié qu'on peut surapprendre un lot**.
- **Mélanger X et y avec des permutations différentes**.
- **Conclure d'une seule exécution** : la graine aléatoire change les résultats (leçon 7.3).

## Exercices guidés

1. Calculez l'écart-type de He pour une couche `Linear(256, 256)` suivie de ReLU, et celui de Xavier pour `Linear(256, 128)`.
2. Dans l'exemple 2, remplacez la sigmoïde par tanh. Le rapport des normes est-il meilleur ou pire ?
3. Introduisez volontairement un autre bug (appliquer `softmax` avant `CrossEntropyLoss`) et observez le test du lot unique.
4. Tracez la précision de validation d'un MLP en fonction de la taille du jeu d'entraînement (50, 100, 200, 400, 600 exemples) sur `multiclass_classification.csv`.

Corrigés : [corrections.md](corrections.md), section « Leçon 7.2 ».

## Défis autonomes

- Construisez un réseau de 30 couches ReLU avec et sans connexions résiduelles (`x = x + layer(x)`) et comparez les normes de gradient de la première couche.
- Ajoutez `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` à une boucle avec un taux d'apprentissage de 1,0 et observez la stabilité.

## Quiz

1. Que se passe-t-il pour les activations d'un réseau profond si les poids initiaux sont trop petits ?
2. Quelle initialisation pour ReLU ? Pour tanh ?
3. Pourquoi un réseau profond à activations sigmoïdes apprend-il mal ?
4. Quelle perte initiale attendre pour 4 classes équilibrées ?
5. Pourquoi le test « surapprendre un seul lot » est-il utile ?
6. Que montre une courbe d'apprentissage en fonction de la taille des données ?

Réponses : [corrections.md](corrections.md), section « Leçon 7.2 ».

## Résumé

- L'initialisation règle l'échelle du signal : He pour ReLU, Xavier pour tanh/sigmoïde.
- Gradients évanescents (sigmoïde, profondeur) et explosifs (taux, échelle) se diagnostiquent avec les normes par couche.
- Débogage : données → perte initiale → un lot → taux d'apprentissage → entraînement complet, une modification à la fois.

## Pour aller plus loin

- Leçon suivante : [Expérimenter : comparer des architectures et des hyperparamètres](03_experimentation_architectures.md).
