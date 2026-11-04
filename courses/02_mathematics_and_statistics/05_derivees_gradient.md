# Leçon 2.5 — Dérivées, gradients et descente de gradient

## Prérequis

- Leçons 2.1 (fonctions) et 2.2 (vecteurs, moindres carrés).

## Objectifs d'apprentissage

- Comprendre la dérivée comme pente locale et taux de variation.
- Calculer des dérivées simples à la main, avec SymPy et numériquement.
- Comprendre les dérivées partielles et le gradient d'une fonction de plusieurs variables.
- Implémenter la descente de gradient et comprendre le rôle du taux d'apprentissage.
- Relier tout cela à l'entraînement d'un modèle : minimiser une fonction de perte.

## Concepts clés

- **Dérivée** (*derivative*) : f'(x), la pente de la tangente à la courbe en x ; la vitesse de variation de f.
- **Règle de dérivation en chaîne** (*chain rule*) : dérivée d'une composition de fonctions ; le cœur de la rétropropagation (module 6).
- **Dérivée partielle** (*partial derivative*) : dérivée par rapport à une variable, les autres étant fixées.
- **Gradient** : vecteur des dérivées partielles ; il pointe vers la direction de plus forte **montée**.
- **Fonction de perte** (*loss function*) : mesure l'erreur d'un modèle ; on cherche à la minimiser.
- **Descente de gradient** (*gradient descent*) : algorithme qui fait des petits pas dans la direction opposée au gradient.
- **Taux d'apprentissage** (*learning rate*, η) : taille des pas.

## Explication détaillée

### La dérivée, une pente locale

Pour une fonction affine, la pente est la même partout. Pour une courbe, la pente change : la dérivée f'(x) est la pente de la droite qui « touche » la courbe en x (la tangente). On l'approche par un taux d'accroissement :

```text
f'(x) ≈ (f(x + h) − f(x − h)) / (2h)   pour h très petit
```

Lecture : si f'(x) > 0, f augmente quand x augmente ; si f'(x) < 0, f diminue ; si f'(x) = 0, on est sur un plat — souvent un minimum ou un maximum.

Dérivées à connaître : (c)' = 0 ; (x)' = 1 ; (xⁿ)' = n·xⁿ⁻¹ ; (eˣ)' = eˣ ; (ln x)' = 1/x ; (a·f + b·g)' = a·f' + b·g'. Et la règle de la chaîne : si y = f(g(x)) alors y' = f'(g(x)) · g'(x). Par exemple ((3x + 1)²)' = 2(3x + 1) · 3. La sigmoïde a une dérivée remarquable : σ'(x) = σ(x)(1 − σ(x)).

### Plusieurs variables : le gradient

Un modèle a beaucoup de paramètres. Sa perte est une fonction de tous ces paramètres, par exemple L(w, b). La dérivée partielle ∂L/∂w indique comment varie la perte si l'on modifie seulement w. Le **gradient** ∇L = (∂L/∂w, ∂L/∂b) rassemble toutes ces informations : il pointe vers la direction où la perte augmente le plus vite. Pour diminuer la perte, il suffit donc d'aller dans la direction **opposée**.

### La descente de gradient

Imaginez-vous dans le brouillard, sur une colline, cherchant le fond de la vallée : vous tâtez la pente sous vos pieds et faites un pas vers le bas, puis recommencez. L'algorithme :

```text
répéter :
    calculer le gradient ∇L(θ) au point actuel θ
    θ ← θ − η · ∇L(θ)
```

Le taux d'apprentissage η est crucial :

- trop petit : convergence très lente ;
- bien choisi : la perte diminue régulièrement ;
- trop grand : on « saute » au-dessus du minimum, la perte oscille voire **diverge**.

Pour une régression linéaire avec la perte MSE, la vallée est un « bol » (fonction convexe) : il n'y a qu'un minimum et la descente de gradient le trouve. Pour un réseau de neurones, le paysage est plus accidenté, mais le principe est identique ; PyTorch calcule les gradients automatiquement (module 6).

**Standardiser les variables** aide beaucoup : si une variable va de 0 à 300 (m²) et une autre de 0 à 1, le bol est très allongé et la descente zigzague. Après standardisation, il est plus rond et la convergence est plus rapide.

## Intuition mathématique

Pour la régression linéaire ŷ = w·x + b avec la perte MSE sur n points :

```text
L(w, b)  = (1/n) Σ (w·xᵢ + b − yᵢ)²
∂L/∂w    = (2/n) Σ (w·xᵢ + b − yᵢ) · xᵢ
∂L/∂b    = (2/n) Σ (w·xᵢ + b − yᵢ)
```

On obtient ces formules par la règle de la chaîne : la dérivée de (erreur)² est 2·erreur multipliée par la dérivée de l'erreur par rapport au paramètre (xᵢ pour w, 1 pour b). En notation matricielle, avec X contenant une colonne de 1 : ∇L = (2/n) Xᵀ(Xθ − y).

## Applications concrètes

- Entraîner une régression linéaire ou logistique sur des millions d'exemples, quand la solution exacte serait trop coûteuse.
- Entraîner tous les réseaux de neurones (avec des variantes : SGD, Adam).
- Analyse de sensibilité : de combien varie une prédiction si une variable d'entrée change un peu ?

## Exemples de code Python

### Exemple 1 — dérivées symboliques et numériques

```python
import numpy as np
import sympy as sp

x = sp.symbols("x")
for expression in [x**3, sp.exp(2 * x), sp.log(x), (3 * x + 1) ** 2, 1 / (1 + sp.exp(-x))]:
    print(f"d/dx {str(expression):<18} = {sp.simplify(sp.diff(expression, x))}")


def numerical_derivative(f, x0, h=1e-5):
    return (f(x0 + h) - f(x0 - h)) / (2 * h)


sigmoid = lambda z: 1 / (1 + np.exp(-z))
x0 = 0.5
print("numerical sigmoid'(0.5):", round(numerical_derivative(sigmoid, x0), 6))
print("formula s(x)(1 - s(x)):  ", round(sigmoid(x0) * (1 - sigmoid(x0)), 6))
```

### Explication du code

`sp.diff` dérive symboliquement ; `sp.simplify` met le résultat sous une forme lisible. La dérivée de (3x + 1)² vaut 18x + 6, ce qui correspond à la règle de la chaîne : 2(3x + 1) · 3. Pour la sigmoïde, SymPy choisit l'écriture équivalente 1/(4·cosh²(x/2)) : c'est la même fonction que σ(x)(1 − σ(x)), ce que confirme la vérification numérique. La dérivée numérique par différence centrale retrouve la formule de la dérivée de la sigmoïde à 6 décimales près. Cette « vérification numérique du gradient » (*gradient checking*) est utilisée pour tester une implémentation de rétropropagation (exercice A01).

### Sortie attendue

```text output
d/dx x**3               = 3*x**2
d/dx exp(2*x)           = 2*exp(2*x)
d/dx log(x)             = 1/x
d/dx (3*x + 1)**2       = 18*x + 6
d/dx 1/(1 + exp(-x))    = 1/(4*cosh(x/2)**2)
numerical sigmoid'(0.5): 0.235004
formula s(x)(1 - s(x)):   0.235004
```

### Exemple 2 — descente de gradient en une dimension, effet du taux d'apprentissage

```python
def f(w):
    return (w - 3) ** 2 + 1


def grad_f(w):
    return 2 * (w - 3)


for learning_rate in [0.01, 0.1, 0.9, 1.1]:
    w = 0.0
    for step in range(25):
        w = w - learning_rate * grad_f(w)
    print(f"lr={learning_rate:<5} after 25 steps: w={w:10.4f}  f(w)={f(w):12.4f}")
```

### Explication du code

Le minimum de f est en w = 3 (f = 1). Avec lr = 0,01 on avance trop lentement ; avec lr = 0,1 on converge ; avec lr = 0,9 on oscille autour de 3 (on saute d'un côté à l'autre) mais on converge quand même ; avec lr = 1,1 chaque pas nous éloigne davantage : la descente **diverge**. Pour cette fonction, toute valeur de lr supérieure à 1 diverge, car le pas multiplie l'écart au minimum par |1 − 2·lr| > 1.

### Sortie attendue

```text output
lr=0.01  after 25 steps: w=    1.1896  f(w)=      4.2775
lr=0.1   after 25 steps: w=    2.9887  f(w)=      1.0001
lr=0.9   after 25 steps: w=    3.0113  f(w)=      1.0001
lr=1.1   after 25 steps: w=  289.1886  f(w)=  81904.9434
```

### Exemple 3 — régression linéaire par descente de gradient

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

houses = pd.read_csv("datasets/generated/house_prices.csv")
x_raw = houses["surface_m2"].to_numpy()
y = houses["price_keur"].to_numpy()
x = (x_raw - x_raw.mean()) / x_raw.std()  # standardize the input

w, b, learning_rate = 0.0, 0.0, 0.1
history = []
for epoch in range(100):
    error = w * x + b - y
    loss = np.mean(error ** 2)
    history.append(loss)
    grad_w = 2 * np.mean(error * x)
    grad_b = 2 * np.mean(error)
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

slope_per_m2 = w / x_raw.std()  # back to original units
intercept = b - w * x_raw.mean() / x_raw.std()
exact_slope, exact_intercept = np.polyfit(x_raw, y, deg=1)
print(f"gradient descent: price = {slope_per_m2:.3f} * surface + {intercept:.2f}")
print(f"exact solution:   price = {exact_slope:.3f} * surface + {exact_intercept:.2f}")
print(f"loss: first={history[0]:.0f}, after 10={history[10]:.0f}, last={history[-1]:.0f}")

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(history)
ax.set_xlabel("iteration")
ax.set_ylabel("MSE loss")
ax.set_yscale("log")
fig.tight_layout()
Path("outputs/module02").mkdir(parents=True, exist_ok=True)
fig.savefig("outputs/module02/gradient_descent_loss.png", dpi=100)
plt.close(fig)
```

### Explication du code

- On standardise la surface pour que la descente converge vite avec un taux d'apprentissage simple.
- Chaque itération calcule les erreurs, la perte, puis les deux dérivées partielles (formules de la section « Intuition mathématique ») et met à jour w et b.
- On reconvertit la pente dans les unités d'origine (k€ par m²) et on compare à la solution exacte des moindres carrés (`np.polyfit`) : les deux coïncident.
- La courbe de perte (`outputs/module02/gradient_descent_loss.png`) descend rapidement puis se stabilise : c'est une **courbe d'apprentissage**, que vous tracerez pour chaque réseau de neurones.

### Sortie attendue

```text output
gradient descent: price = 2.838 * surface + 22.69
exact solution:   price = 2.838 * surface + 22.69
loss: first=102331, after 10=9628, last=8547
```

## Erreurs fréquentes

- **Ajouter le gradient au lieu de le soustraire** : on monte la colline au lieu de descendre.
- **Taux d'apprentissage trop grand** : perte qui augmente ou devient `nan`/`inf`.
- **Variables non standardisées** : convergence très lente ou divergence.
- **Oublier de recalculer le gradient à chaque itération** avec les paramètres à jour.
- **Mettre à jour w puis utiliser le nouveau w pour calculer le gradient de b** dans la même itération : calculez tous les gradients d'abord.
- **Arrêter trop tôt sans regarder la courbe de perte**.

## Exercices guidés

1. Calculez à la main la dérivée de f(x) = 4x³ − 2x + 5, puis vérifiez avec SymPy et numériquement en x = 2.
2. Calculez le gradient de g(w₁, w₂) = w₁² + 3w₂² + w₁w₂ avec SymPy (`sp.diff(g, w1)`, `sp.diff(g, w2)`) et évaluez-le en (1, 2).
3. Minimisez g par descente de gradient en partant de (4, −3) avec lr = 0,1 pendant 100 itérations. Le minimum est (0, 0).
4. Dans l'exemple 3, relancez avec `learning_rate = 1.2`. Que se passe-t-il ?

Corrigés : [corrections.md](corrections.md), section « Leçon 2.5 ».

## Défis autonomes

- Ajoutez un critère d'arrêt : stopper quand la perte diminue de moins de 10⁻⁶ entre deux itérations. Combien d'itérations faut-il ?
- Implémentez la descente de gradient **stochastique par mini-lots** (*mini-batch SGD*) : à chaque étape, calculez le gradient sur 32 maisons tirées au hasard. Comparez les courbes de perte.
- Refaites l'exemple 3 **sans** standardiser la surface : quel taux d'apprentissage maximal évite la divergence ?

## Quiz

1. Que représente géométriquement f'(x) ?
2. Que vaut la dérivée de x⁵ ? Celle de e³ˣ ?
3. Dans quelle direction pointe le gradient ? Pourquoi soustrait-on le gradient ?
4. Que se passe-t-il si le taux d'apprentissage est trop grand ?
5. Pourquoi standardiser les variables avant une descente de gradient ?
6. Quel est le lien entre la règle de la chaîne et les réseaux de neurones ?

Réponses : [corrections.md](corrections.md), section « Leçon 2.5 ».

## Résumé

- La dérivée est une pente locale ; le gradient regroupe les dérivées partielles et pointe vers la montée la plus raide.
- Descente de gradient : θ ← θ − η∇L(θ), répétée jusqu'à convergence.
- Le taux d'apprentissage contrôle la taille des pas : trop petit = lent, trop grand = divergence.
- La règle de la chaîne permet de calculer les gradients de fonctions composées : c'est la rétropropagation.

## Pour aller plus loin

- Exercice **I04** (`exercises/intermediate/I04_gradient_descent`) : régression multivariée par descente de gradient.
- Module 6 : PyTorch calcule les gradients automatiquement (*autograd*).
- Module 5, leçon 5.2 : régression linéaire avec scikit-learn.
