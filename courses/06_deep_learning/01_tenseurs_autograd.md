# Leçon 6.1 — Tenseurs, opérations et différentiation automatique

## Prérequis

- Leçon 1.4 (NumPy) et leçon 2.5 (dérivées, règle de la chaîne, descente de gradient).
- PyTorch installé.

## Objectifs d'apprentissage

- Créer des tenseurs PyTorch et passer de NumPy à PyTorch et inversement.
- Connaître les attributs `shape`, `dtype`, `device` et les opérations courantes.
- Comprendre le graphe de calcul et la différentiation automatique (`requires_grad`, `backward`, `grad`).
- Écrire une descente de gradient avec autograd.
- Savoir quand désactiver le calcul des gradients (`torch.no_grad`).

## Concepts clés

- **Tenseur** (*tensor*) : tableau multidimensionnel, équivalent PyTorch du `ndarray` NumPy, qui peut vivre sur CPU ou GPU et mémoriser comment il a été calculé.
- **Type** (*dtype*) : `torch.float32` (par défaut pour les réels en Deep Learning), `torch.int64` (étiquettes de classes)...
- **Périphérique** (*device*) : `cpu` ou `cuda` (GPU NVIDIA).
- **Graphe de calcul** (*computational graph*) : enregistrement des opérations qui ont produit un tenseur.
- **Différentiation automatique** (*automatic differentiation*, autograd) : calcul exact des gradients en parcourant ce graphe à l'envers (règle de la chaîne).
- **Feuille** (*leaf tensor*) : tenseur créé par l'utilisateur (un paramètre), dont on veut le gradient.

## Explication détaillée

### Des tableaux NumPy aux tenseurs

L'API des tenseurs ressemble beaucoup à NumPy : `torch.tensor([...])`, `torch.zeros`, `torch.randn`, indexation, broadcasting, `@` pour le produit matriciel, `sum(dim=0)` au lieu de `sum(axis=0)`. Trois différences importantes :

1. **Le type par défaut** des réels est `float32` (simple précision) : deux fois moins de mémoire que le `float64` de NumPy, précision suffisante pour l'apprentissage. Mélanger `float64` et `float32` provoque des erreurs ; convertissez avec `.float()`.
2. **Le périphérique** : un tenseur est sur le CPU par défaut. Sur une machine avec GPU, `tensor.to("cuda")` le déplace ; toutes les opérations doivent combiner des tenseurs du **même** périphérique. Ce cours fonctionne entièrement sur CPU.
3. **L'autograd** : un tenseur créé avec `requires_grad=True` enregistre toutes les opérations qui l'utilisent.

`torch.from_numpy(array)` partage la mémoire avec le tableau NumPy (pas de copie) ; `tensor.numpy()` fait l'inverse pour un tenseur CPU sans gradient (sinon `tensor.detach().numpy()`).

### Le graphe de calcul

Quand on écrit `y = (w * x + b) ** 2` avec `w` et `b` qui exigent un gradient, PyTorch construit un graphe : chaque nœud est une opération (multiplication, addition, puissance) et connaît sa dérivée locale. L'appel `y.backward()` parcourt le graphe de la sortie vers les entrées en multipliant les dérivées locales : c'est exactement la règle de la chaîne de la leçon 2.5, appliquée automatiquement. Les gradients ∂y/∂w et ∂y/∂b sont alors disponibles dans `w.grad` et `b.grad`.

Trois règles pratiques :

- `backward()` s'appelle sur un **scalaire** (la perte moyenne d'un lot, par exemple).
- Les gradients **s'accumulent** : un deuxième `backward()` ajoute au gradient existant. Il faut les remettre à zéro à chaque itération (`w.grad = None` ou `optimizer.zero_grad()`).
- La mise à jour des paramètres ne doit pas être enregistrée dans le graphe : on l'écrit dans un bloc `with torch.no_grad():`. Ce même bloc s'utilise pour la prédiction et l'évaluation, ce qui économise mémoire et temps.

### Pourquoi l'autograd change tout

Dans la leçon 2.5, nous avons dérivé à la main les formules du gradient de la MSE. Pour un réseau de millions de paramètres et de dizaines de couches, c'est impossible. L'autograd calcule ces gradients exactement, pour n'importe quelle combinaison d'opérations dérivables. Il suffit d'écrire la **propagation avant** (*forward*) ; la **rétropropagation** (*backward*) est automatique.

## Intuition mathématique

Pour y = (w·x + b)², posons u = w·x + b. Alors :

```text
∂y/∂u = 2u      ∂u/∂w = x      ∂u/∂b = 1
∂y/∂w = 2u · x  ∂y/∂b = 2u     (règle de la chaîne)
avec w = 2, x = 3, b = 1 : u = 7, ∂y/∂w = 42, ∂y/∂b = 14
```

`backward()` calcule précisément ces produits de dérivées locales, du nœud de sortie vers les feuilles (mode « inverse » de la différentiation automatique, efficace quand il y a beaucoup de paramètres et une seule sortie scalaire).

## Applications concrètes

- Entraîner n'importe quel réseau de neurones : perte → `backward()` → mise à jour.
- Optimiser des fonctions arbitraires (calibration de paramètres physiques, ajustement de courbes).
- Calculer la sensibilité d'une prédiction aux entrées (gradient par rapport à l'image : cartes de saillance).

## Exemples de code Python

### Exemple 1 — créer et manipuler des tenseurs

```python
import numpy as np
import torch

torch.manual_seed(0)
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.ones(2, 2)
print(a.shape, a.dtype, a.device)
print("a + b =\n", a + b)
print("a @ a =\n", a @ a)
print("column sums:", a.sum(dim=0), "| mean:", a.mean().item())

x_np = np.arange(6, dtype=np.float64).reshape(2, 3)
x = torch.from_numpy(x_np)
print("from NumPy:", x.dtype, "-> float32:", x.float().dtype)
x_np[0, 0] = 100.0
print("shared memory:", x[0, 0].item())

r = torch.randn(3, 4)
print("random normal shape:", tuple(r.shape), "| reshaped:", tuple(r.view(4, 3).shape), "| first row:", r[0].round(decimals=3))
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device used in this course:", device)
```

### Explication du code

- `a.sum(dim=0)` somme les colonnes (`dim` joue le rôle de `axis`) ; `.item()` extrait un nombre Python d'un tenseur à un seul élément.
- `torch.from_numpy` partage la mémoire : modifier le tableau NumPy modifie le tenseur. Un tableau NumPy de réels est en `float64` : on le convertit en `float32` avec `.float()` avant de l'utiliser dans un réseau.
- `view` change la forme sans copier (comme `reshape` de NumPy).
- La dernière ligne affiche `cpu` sur une machine sans GPU NVIDIA : c'est le cas prévu par ce cours.

### Sortie attendue

```text output
torch.Size([2, 2]) torch.float32 cpu
a + b =
 tensor([[2., 3.],
        [4., 5.]])
a @ a =
 tensor([[ 7., 10.],
        [15., 22.]])
column sums: tensor([4., 6.]) | mean: 2.5
from NumPy: torch.float64 -> float32: torch.float32
shared memory: 100.0
random normal shape: (3, 4) | reshaped: (4, 3) | first row: tensor([ 1.5410, -0.2930, -2.1790,  0.5680])
device used in this course: cpu
```

### Exemple 2 — l'autograd sur un calcul simple

```python
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(3.0)

y = (w * x + b) ** 2
print("y =", y.item(), "| grad_fn:", type(y.grad_fn).__name__)
y.backward()
print("dy/dw =", w.grad.item(), "| dy/db =", b.grad.item())

# Gradients accumulate if we call backward again without resetting them.
y = (w * x + b) ** 2
y.backward()
print("after a second backward without reset:", w.grad.item())
w.grad = None
b.grad = None

with torch.no_grad():
    z = w * x
print("requires_grad inside no_grad:", z.requires_grad)
```

### Explication du code

- `grad_fn` indique la dernière opération enregistrée (ici la puissance) : c'est le point d'entrée du graphe.
- Les gradients obtenus (42 et 14) sont ceux calculés à la main dans la section « Intuition mathématique ».
- Un second `backward()` sans remise à zéro **additionne** : 42 + 42 = 84. C'est voulu (utile pour accumuler sur plusieurs mini-lots), mais source de bugs fréquents.
- Dans `torch.no_grad()`, aucune opération n'est enregistrée.

### Sortie attendue

```text output
y = 49.0 | grad_fn: PowBackward0
dy/dw = 42.0 | dy/db = 14.0
after a second backward without reset: 84.0
requires_grad inside no_grad: False
```

### Exemple 3 — régression linéaire par descente de gradient avec autograd

```python
import pandas as pd

houses = pd.read_csv("datasets/generated/house_prices.csv")
surface = torch.tensor(houses["surface_m2"].to_numpy(), dtype=torch.float32)
price = torch.tensor(houses["price_keur"].to_numpy(), dtype=torch.float32)
x = (surface - surface.mean()) / surface.std()

w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
learning_rate = 0.1
for step in range(100):
    prediction = w * x + b
    loss = ((prediction - price) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    w.grad = None
    b.grad = None
    if step in (0, 10, 99):
        print(f"step {step:>2}: loss={loss.item():10.1f}")

slope = w.item() / surface.std().item()
print(f"price ~ {slope:.3f} * surface + {b.item() - slope * surface.mean().item():.2f}")
```

### Explication du code

C'est la régression de la leçon 2.5, mais sans aucune formule de gradient : `loss.backward()` calcule ∂loss/∂w et ∂loss/∂b. La mise à jour est faite dans `torch.no_grad()` (sinon PyTorch voudrait dériver la mise à jour elle-même), puis les gradients sont remis à zéro. On retrouve la même droite qu'avec la solution exacte (≈ 2,84 k€ par m²) : `torch.std` utilise par défaut la correction n − 1, ce qui ne change pas la pente une fois reconvertie.

### Sortie attendue

```text output
step  0: loss=  102330.7
step 10: loss=    9628.7
step 99: loss=    8546.5
price ~ 2.838 * surface + 22.69
```

## Erreurs fréquentes

- **Oublier de remettre les gradients à zéro** : ils s'accumulent et l'entraînement diverge ou se comporte bizarrement.
- **Modifier un paramètre hors de `torch.no_grad()`** : `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation`.
- **Mélanger `float64` et `float32`** : `RuntimeError: expected scalar type Float but found Double`. Convertissez avec `.float()` ou `dtype=torch.float32`.
- **Appeler `.numpy()` sur un tenseur qui exige un gradient** : utilisez `.detach().numpy()`.
- **Mélanger CPU et GPU** dans une même opération.
- **Appeler `backward()` sur un tenseur non scalaire** sans argument : réduisez d'abord (`.mean()`, `.sum()`).

## Exercices guidés

1. Créez un tenseur 3 × 3 de nombres de 1 à 9, calculez sa transposée, la somme de chaque ligne et le produit matriciel avec lui-même.
2. Avec l'autograd, calculez la dérivée de f(x) = x³ − 2x en x = 2 et vérifiez avec la formule 3x² − 2.
3. Calculez le gradient de g(w₁, w₂) = w₁² + 3w₂² + w₁w₂ en (1, 2) avec l'autograd, et comparez au résultat SymPy de la leçon 2.5.
4. Modifiez l'exemple 3 pour utiliser deux variables (surface et distance au centre, standardisées).

Corrigés : [corrections.md](corrections.md), section « Leçon 6.1 ».

## Défis autonomes

- Minimisez la fonction de Rosenbrock f(x, y) = (1 − x)² + 100(y − x²)² avec l'autograd en partant de (−1, 1). Testez plusieurs taux d'apprentissage. Le minimum est (1, 1).
- Mesurez avec `time.perf_counter` le temps d'un produit matriciel 1000 × 1000 en NumPy (`float64`) et en PyTorch (`float32`) sur votre CPU.

## Quiz

1. Quelle est la principale différence d'usage entre un tenseur et un tableau NumPy ?
2. Quel est le type réel par défaut dans PyTorch ? Pourquoi ?
3. Que fait `loss.backward()` ?
4. Pourquoi faut-il remettre les gradients à zéro ?
5. À quoi sert `torch.no_grad()` ?
6. Pourquoi `tensor.numpy()` échoue-t-il parfois ?

Réponses : [corrections.md](corrections.md), section « Leçon 6.1 ».

## Résumé

- Les tenseurs sont des tableaux NumPy capables de vivre sur GPU et de retenir leur historique de calcul.
- `requires_grad=True` + `backward()` = gradients exacts par la règle de la chaîne, sans formule écrite à la main.
- Les gradients s'accumulent : remise à zéro à chaque itération ; mises à jour et évaluation dans `torch.no_grad()`.

## Pour aller plus loin

- Leçon suivante : [Du neurone au réseau : un réseau de neurones en NumPy](02_neurone_reseau_numpy.md).
