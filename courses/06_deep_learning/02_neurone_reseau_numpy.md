# Leçon 6.2 — Du neurone au réseau : un réseau de neurones en NumPy

## Prérequis

- Leçon 6.1 ; leçons 2.2 (produit matriciel) et 2.5 (règle de la chaîne) ; leçon 5.3 (régression logistique, entropie croisée).

## Objectifs d'apprentissage

- Décrire un neurone artificiel : somme pondérée, biais, fonction d'activation.
- Comprendre pourquoi empiler des couches avec des activations non linéaires permet d'apprendre des frontières complexes.
- Écrire la propagation avant (*forward propagation*) d'un réseau à une couche cachée.
- Dériver et coder la rétropropagation (*backpropagation*), puis la vérifier numériquement.
- Entraîner ce réseau par descente de gradient et comparer à un neurone seul.

## Concepts clés

- **Neurone** (*neuron*) : calcule a = f(w·x + b), où f est une **fonction d'activation**.
- **Couche dense** (*dense / fully connected layer*) : chaque neurone reçoit toutes les sorties de la couche précédente ; en matrice : A = f(XW + b).
- **Couche cachée** (*hidden layer*) : couche entre l'entrée et la sortie.
- **Activation** : sigmoïde, tanh, **ReLU** (max(0, z)) ; elles introduisent la non-linéarité.
- **Propagation avant** : calcul des sorties, couche après couche.
- **Rétropropagation** : calcul des gradients de la perte, de la sortie vers l'entrée, par la règle de la chaîne.
- **Époque** (*epoch*) : un passage complet sur les données d'entraînement.

## Explication détaillée

### Un neurone = une régression logistique

Le neurone le plus simple calcule un score z = w·x + b et le transforme par une sigmoïde : c'est exactement la régression logistique de la leçon 5.3. Sa frontière de décision (z = 0) est une **droite** (un hyperplan). Il ne peut donc pas séparer deux classes imbriquées comme deux croissants de lune.

### Empiler des couches

Une couche cachée de h neurones transforme l'entrée x (2 variables) en h nouvelles variables : H = ReLU(XW₁ + b₁). Un neurone de sortie combine ensuite ces variables : p = σ(HW₂ + b₂). Chaque neurone caché apprend une « demi-droite » utile ; leur combinaison dessine des frontières courbes. Plus il y a de neurones, plus les formes possibles sont riches.

Pourquoi l'activation non linéaire est-elle indispensable ? Sans elle, H = XW₁ + b₁ puis XW₁W₂ + ... reste une fonction **linéaire** de X : empiler dix couches linéaires équivaut à une seule. La non-linéarité est ce qui donne leur puissance aux réseaux.

### La propagation avant

Avec X de forme (n, 2), W₁ de forme (2, h), W₂ de forme (h, 1) :

```text
Z1 = X W1 + b1        (n, h)
H  = ReLU(Z1)         (n, h)
Z2 = H W2 + b2        (n, 1)
P  = σ(Z2)            (n, 1)    probabilité de la classe 1
L  = −mean[y log P + (1 − y) log(1 − P)]    entropie croisée binaire
```

### La rétropropagation

On remonte la chaîne. Une simplification remarquable : pour la sigmoïde suivie de l'entropie croisée, ∂L/∂Z2 = (P − y)/n. Puis :

```text
dZ2 = (P − y) / n                (n, 1)
dW2 = Hᵀ dZ2                     (h, 1)      db2 = somme de dZ2
dH  = dZ2 W2ᵀ                    (n, h)
dZ1 = dH ⊙ 1[Z1 > 0]             (n, h)      dérivée de ReLU : 1 si z > 0, sinon 0
dW1 = Xᵀ dZ1                     (2, h)      db1 = somme de dZ1 sur les lignes
```

Chaque gradient a **la même forme** que le paramètre correspondant : un excellent moyen de vérifier ses formules. On contrôle ensuite le résultat par différences finies (*gradient checking*), comme dans la leçon 2.5.

### L'initialisation

Si tous les poids valent 0, tous les neurones cachés calculent la même chose et reçoivent le même gradient : ils restent identiques pour toujours (problème de **symétrie**). On initialise donc les poids **aléatoirement**, avec une échelle adaptée au nombre d'entrées (initialisation de He pour ReLU : écart-type √(2/n_entrées)), et les biais à 0. La leçon 7.2 revient sur ce point.

## Intuition mathématique

La rétropropagation n'est que la règle de la chaîne organisée efficacement : on calcule une fois le gradient par rapport à chaque couche intermédiaire (dZ2, dH, dZ1) et on le réutilise pour tous les poids de la couche. Son coût est du même ordre que celui de la propagation avant, ce qui rend possible l'entraînement de réseaux de millions de paramètres.

## Applications concrètes

- Comprendre ce que font `nn.Linear`, `nn.ReLU` et `loss.backward()` dans PyTorch (leçon 6.3).
- Déboguer un réseau qui n'apprend pas : connaître les formes et les gradients attendus.
- Implémenter une couche personnalisée et vérifier ses gradients.

## Exemples de code Python

### Exemple 1 — un neurone seul échoue sur des données non linéaires

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=600, noise=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
y_train_col = y_train.reshape(-1, 1).astype(float)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)  # avoid log(0)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


rng = np.random.default_rng(0)
w, b = rng.normal(0, 0.1, size=(2, 1)), 0.0
for epoch in range(2000):
    p = sigmoid(X_train @ w + b)
    dz = (p - y_train_col) / len(X_train)
    w -= 1.0 * (X_train.T @ dz)
    b -= 1.0 * dz.sum()
neuron_accuracy = ((sigmoid(X_test @ w + b) >= 0.5).ravel() == y_test).mean()
print("single neuron: loss =", round(binary_cross_entropy(sigmoid(X_train @ w + b), y_train_col), 3), "| test accuracy =", round(neuron_accuracy, 3))
```

### Explication du code

`make_moons` (scikit-learn) génère localement deux croissants imbriqués avec un peu de bruit. Le neurone unique est entraîné par descente de gradient avec le gradient simplifié (p − y)/n. Sa frontière est une droite : il plafonne autour de 85 % de bonnes réponses, car une partie de chaque croissant se trouve du « mauvais côté » de toute droite.

### Sortie attendue

```text output
single neuron: loss = 0.255 | test accuracy = 0.86
```

### Exemple 2 — un réseau à une couche cachée, avec vérification du gradient

```python
def init_params(n_inputs, n_hidden, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, np.sqrt(2 / n_inputs), size=(n_inputs, n_hidden)),
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(0, np.sqrt(2 / n_hidden), size=(n_hidden, 1)),
        "b2": np.zeros(1),
    }


def forward(params, X):
    Z1 = X @ params["W1"] + params["b1"]
    H = np.maximum(0, Z1)
    P = sigmoid(H @ params["W2"] + params["b2"])
    return P, (Z1, H)


def backward(params, X, y, P, cache):
    Z1, H = cache
    dZ2 = (P - y) / len(X)
    dH = dZ2 @ params["W2"].T
    dZ1 = dH * (Z1 > 0)
    return {"W1": X.T @ dZ1, "b1": dZ1.sum(axis=0), "W2": H.T @ dZ2, "b2": dZ2.sum(axis=0)}


params = init_params(2, 16)
P, cache = forward(params, X_train)
grads = backward(params, X_train, y_train_col, P, cache)
print({name: g.shape for name, g in grads.items()})

# Gradient checking on a few weights.
h = 1e-6
for name, index in [("W1", (0, 3)), ("W2", (5, 0)), ("b1", (2,))]:
    original = params[name][index]
    params[name][index] = original + h
    loss_plus = binary_cross_entropy(forward(params, X_train)[0], y_train_col)
    params[name][index] = original - h
    loss_minus = binary_cross_entropy(forward(params, X_train)[0], y_train_col)
    params[name][index] = original
    numeric = (loss_plus - loss_minus) / (2 * h)
    print(f"{name}{index}: backprop={grads[name][index]: .6f}  numeric={numeric: .6f}")
```

### Explication du code

- `init_params` utilise l'initialisation de He (écart-type √(2/n_entrées)) adaptée à ReLU.
- `forward` renvoie aussi les valeurs intermédiaires (`cache`) dont la rétropropagation a besoin.
- `backward` applique les formules de la section « Explication détaillée » ; les formes des gradients sont identiques à celles des paramètres.
- La vérification par différences finies donne les mêmes valeurs que la rétropropagation, à 10⁻⁶ près : les formules sont correctes. (Le léger écart possible vient de l'écrêtage des probabilités et de l'arrondi.)

### Sortie attendue

```text output
{'W1': (2, 16), 'b1': (16,), 'W2': (16, 1), 'b2': (1,)}
W1(0, 3): backprop=-0.003104  numeric=-0.003104
W2(5, 0): backprop= 0.214699  numeric= 0.214699
b1(2,): backprop= 0.005599  numeric= 0.005599
```

### Exemple 3 — entraîner le réseau et visualiser la frontière

```python
from pathlib import Path

import matplotlib.pyplot as plt

from lab.plotting import CATEGORICAL, apply_style

params = init_params(2, 16, seed=1)
learning_rate = 0.5
history = []
for epoch in range(3000):
    P, cache = forward(params, X_train)
    history.append(binary_cross_entropy(P, y_train_col))
    grads = backward(params, X_train, y_train_col, P, cache)
    for name in params:
        params[name] -= learning_rate * grads[name]

test_accuracy = ((forward(params, X_test)[0] >= 0.5).ravel() == y_test).mean()
print(f"network: loss {history[0]:.3f} -> {history[-1]:.3f} | test accuracy = {test_accuracy:.3f}")

apply_style()
xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, (title, proba) in zip(axes, [("neurone seul", sigmoid(grid @ w + b)), ("réseau 2-16-1", forward(params, grid)[0])]):
    ax.contourf(xx, yy, proba.reshape(xx.shape), levels=[0, 0.5, 1], colors=["#f6d3c4", "#cde2fb"], alpha=0.8)
    for label, color in zip([0, 1], [CATEGORICAL[1], CATEGORICAL[0]]):
        ax.scatter(*X_test[y_test == label].T, s=10, color=color, label=f"classe {label}")
    ax.set_title(title)
    ax.grid(False)
axes[0].legend(loc="lower left")
Path("outputs/module06").mkdir(parents=True, exist_ok=True)
fig.savefig("outputs/module06/numpy_network_boundary.png", bbox_inches="tight")
plt.close(fig)
```

### Explication du code

- La boucle d'entraînement est la même qu'au module 2 : forward, perte, gradients, mise à jour de chaque paramètre.
- Le réseau atteint une précision nettement supérieure au neurone seul : sa frontière, visible dans `outputs/module06/numpy_network_boundary.png`, épouse la forme des croissants.
- La perte d'entraînement devient très faible (≈ 0,03) alors que les données contiennent du bruit : le réseau commence à apprendre ce bruit par cœur. Seule la précision de **test** mesure la généralisation ; la leçon 6.4 montre comment surveiller et limiter ce surapprentissage.
- `np.meshgrid` construit une grille de points sur laquelle on évalue le modèle pour colorier les zones de décision.

### Sortie attendue

```text output
network: loss 0.922 -> 0.029 | test accuracy = 0.947
```

## Erreurs fréquentes

- **Initialiser tous les poids à zéro** : symétrie, les neurones restent identiques.
- **Oublier l'activation non linéaire** : le réseau reste linéaire.
- **Erreurs de forme** (n,) contre (n, 1) : le broadcasting produit silencieusement une matrice (n, n). Gardez y en colonne.
- **`log(0)`** dans l'entropie croisée : écrêtez les probabilités.
- **Taux d'apprentissage trop grand** : la perte oscille ou devient `nan`.
- **Ne pas vérifier les gradients** quand on écrit une rétropropagation à la main.

## Exercices guidés

1. Calculez à la main la sortie d'un neurone avec w = (0,5 ; −1), b = 0,2, x = (2 ; 1) et une activation sigmoïde.
2. Remplacez ReLU par tanh dans le réseau (dérivée : 1 − tanh²) et comparez la précision.
3. Entraînez le réseau avec 2, 4, 16 et 64 neurones cachés. Comment évolue la précision de test ?
4. Que se passe-t-il si l'on initialise tous les poids à 0 ? Vérifiez que les colonnes de W1 restent identiques.

Corrigés : [corrections.md](corrections.md), section « Leçon 6.2 ».

## Défis autonomes

- Ajoutez une deuxième couche cachée et écrivez sa rétropropagation. Vérifiez vos gradients numériquement.
- Implémentez la descente de gradient par mini-lots (32 exemples) et comparez la vitesse de convergence.
- Exercice **A01** : réseau NumPy générique et testé.

## Quiz

1. Quelle est la sortie d'un neurone ?
2. Pourquoi un neurone seul ne peut-il pas séparer les deux croissants ?
3. Que se passe-t-il si l'on retire toutes les activations non linéaires d'un réseau ?
4. Pourquoi ne pas initialiser les poids à zéro ?
5. Quelle est la dérivée de ReLU ?
6. Comment vérifier qu'une rétropropagation écrite à la main est correcte ?

Réponses : [corrections.md](corrections.md), section « Leçon 6.2 ».

## Résumé

- Neurone = somme pondérée + biais + activation ; un neurone sigmoïde est une régression logistique.
- Couches cachées + activations non linéaires = frontières complexes.
- Forward : calculer les sorties ; backward : la règle de la chaîne, couche par couche ; les gradients ont la forme des paramètres.
- Initialisation aléatoire adaptée et vérification numérique des gradients.

## Pour aller plus loin

- Exercice **A01** (`exercises/advanced/A01_numpy_neural_network`).
- Leçon suivante : [Entraîner un réseau avec PyTorch](03_pytorch_entrainement.md).
