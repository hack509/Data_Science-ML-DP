# Leçon 2.1 — Algèbre, fonctions et graphiques

## Prérequis

- Module 1, leçon 1.4 (NumPy).
- Les quatre opérations, les fractions et les puissances.

## Objectifs d'apprentissage

- Revoir les règles de calcul (priorités, puissances, racines, logarithmes) et les appliquer en Python.
- Résoudre des équations simples à la main puis avec SymPy.
- Comprendre la notion de fonction, la lire sur un graphique et la tracer avec Matplotlib.
- Connaître les fonctions qui reviennent en IA : affine, quadratique, exponentielle, logarithme, sigmoïde.

## Concepts clés

- **Expression** : combinaison de nombres, de variables et d'opérations, par exemple `3x + 2`.
- **Équation** : égalité entre deux expressions que l'on cherche à rendre vraie (`3x + 2 = 11`).
- **Fonction** (*function*) : règle qui associe à chaque entrée x une sortie unique f(x).
- **Pente** (*slope*) : variation de la sortie quand l'entrée augmente de 1, pour une fonction affine.
- **Ordonnée à l'origine** (*intercept*) : valeur de la fonction en x = 0.
- **Calcul symbolique** (*symbolic computation*) : manipulation exacte de formules (SymPy), par opposition au calcul numérique approché (NumPy).

## Explication détaillée

### Rappels de calcul

L'ordre des opérations est : parenthèses, puissances, multiplications et divisions, additions et soustractions. En Python, la puissance s'écrit `**` (et non `^`, qui est un « ou exclusif » binaire !).

Règles des puissances, valables pour a > 0 : aᵐ · aⁿ = aᵐ⁺ⁿ, (aᵐ)ⁿ = aᵐⁿ, a⁻ⁿ = 1/aⁿ, a^(1/2) = √a.

Le **logarithme** est l'opération inverse de l'exponentielle : si y = eˣ alors x = ln(y). Il transforme les produits en sommes : ln(a · b) = ln(a) + ln(b). C'est pour cela qu'en Machine Learning on maximise souvent le **log** d'une vraisemblance (un produit de probabilités devient une somme, plus stable numériquement). Le logarithme « écrase » aussi les grandes valeurs : on l'applique à des variables très asymétriques comme les prix ou les revenus.

### Résoudre une équation

Pour résoudre 3x + 2 = 11, on applique la même opération des deux côtés : on retire 2 (3x = 9), puis on divise par 3 (x = 3). Pour un **système** de deux équations à deux inconnues, on peut substituer ou combiner les équations ; on verra en leçon 2.2 que c'est un problème matriciel. SymPy résout ces équations **exactement** et peut aussi développer, factoriser et simplifier des expressions.

### Fonctions et graphiques

Une fonction f associe une sortie à chaque entrée. Son **graphique** est l'ensemble des points (x, f(x)). Lire un graphique, c'est répondre à : où la fonction croît-elle ? où atteint-elle un minimum ? à quelle vitesse change-t-elle ?

Fonctions à connaître :

| Fonction | Formule | Où la rencontre-t-on en IA ? |
|---|---|---|
| affine | f(x) = a·x + b | régression linéaire, neurone sans activation |
| quadratique | f(x) = (x − c)² | erreur quadratique, fonctions de perte |
| exponentielle | f(x) = eˣ | softmax, lois de probabilité |
| logarithme | f(x) = ln(x) | entropie croisée (*cross-entropy*), transformation de variables |
| sigmoïde | σ(x) = 1 / (1 + e⁻ˣ) | régression logistique, probabilité d'une classe |
| ReLU | f(x) = max(0, x) | activation des réseaux de neurones |

Pour une fonction affine, a est la pente et b l'ordonnée à l'origine. Si a = 2,5 (k€ par m²) et b = 30 (k€), un logement de 80 m² est estimé à 2,5 × 80 + 30 = 230 k€ : c'est déjà un modèle de prédiction !

## Intuition mathématique

La sigmoïde transforme n'importe quel réel en un nombre entre 0 et 1 :

```text
σ(x) = 1 / (1 + e^(−x))       σ(0) = 0,5     σ(x) → 1 quand x → +∞     σ(x) → 0 quand x → −∞
```

C'est exactement ce qu'il faut pour transformer un **score** (une somme pondérée qui peut valoir −3 ou 12) en **probabilité**. La fonction quadratique (x − 3)² est minimale en x = 3, là où sa courbe est « au fond de la vallée » : chercher les paramètres d'un modèle revient à chercher le fond d'une vallée de ce type (leçon 2.5).

## Applications concrètes

- Estimer un prix à partir d'une surface avec une fonction affine.
- Transformer des revenus très dispersés avec un logarithme avant une analyse.
- Convertir un score en probabilité avec la sigmoïde (détection de spam).
- Calculer des intérêts composés avec une exponentielle.

## Exemples de code Python

### Exemple 1 — calcul numérique et symbolique

```python
import math

import sympy as sp

print(2 + 3 * 4 ** 2, (2 + 3) * 4 ** 2)
print(math.sqrt(16), 16 ** 0.5, math.exp(1), math.log(math.e))
print("log(a*b) = log(a) + log(b):", math.isclose(math.log(6 * 7), math.log(6) + math.log(7)))

x, y = sp.symbols("x y")
print(sp.solve(sp.Eq(3 * x + 2, 11), x))
print(sp.solve([sp.Eq(2 * x + y, 7), sp.Eq(x - y, -1)], [x, y]))
expression = (x + 2) * (x - 3)
print(sp.expand(expression), "|", sp.factor(x**2 - 5 * x + 6))
print(sp.solve(x**2 - 5 * x + 6, x))
print(sp.Rational(1, 3) + sp.Rational(1, 6), "vs", 1 / 3 + 1 / 6)
```

### Explication du code

- Les priorités opératoires sont respectées : `3 * 4 ** 2` vaut 48, donc la première expression vaut 50.
- `sp.symbols` crée des variables symboliques ; `sp.Eq(gauche, droite)` construit une équation et `sp.solve` la résout exactement.
- Le système 2x + y = 7, x − y = −1 a pour solution x = 2, y = 3.
- `expand` développe, `factor` factorise : x² − 5x + 6 = (x − 2)(x − 3), dont les racines sont 2 et 3.
- `sp.Rational` calcule avec des fractions exactes : 1/3 + 1/6 vaut exactement 1/2. Avec les réels Python, on obtient bien 0.5 ici, mais 1/3 lui-même n'est stocké que comme une approximation (0.333...), et ces erreurs peuvent s'accumuler dans des calculs plus longs.

### Sortie attendue

```text output
50 80
4.0 4.0 2.718281828459045 1.0
log(a*b) = log(a) + log(b): True
[3]
{x: 2, y: 3}
x**2 - x - 6 | (x - 3)*(x - 2)
[2, 3]
1/2 vs 0.5
```

### Exemple 2 — tracer les fonctions usuelles de l'IA

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-6, 6, 400)
functions = {
    "affine 0.5x + 1": 0.5 * x + 1,
    "quadratique (x-1)²/4": (x - 1) ** 2 / 4,
    "sigmoïde": 1 / (1 + np.exp(-x)),
    "ReLU": np.maximum(0, x),
}
fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
for ax, (name, values) in zip(axes, functions.items()):
    ax.plot(x, values, color="tab:blue")
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.set_title(name)
fig.tight_layout()

output_dir = Path("outputs/module02")
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "usual_functions.png", dpi=100)
plt.close(fig)
print("saved:", (output_dir / "usual_functions.png").as_posix())

sigmoid = lambda z: 1 / (1 + np.exp(-z))
print("sigmoid(-4, 0, 4):", sigmoid(np.array([-4, 0, 4])).round(3))
```

### Explication du code

`np.linspace(-6, 6, 400)` crée 400 abscisses ; chaque fonction est évaluée sur tout le tableau d'un coup (vectorisation). `plt.subplots(1, 4)` crée une figure avec 4 graphiques côte à côte. La figure est **enregistrée** dans `outputs/module02/usual_functions.png` : ouvrez ce fichier dans VS Code pour la voir. Dans un notebook Jupyter, la figure s'affiche directement ; dans un script interactif, ajoutez `plt.show()`.

Ce que montre la figure : la droite affine (pente constante), la parabole (minimum en x = 1), la sigmoïde (courbe en S entre 0 et 1) et la ReLU (nulle pour x < 0, identité ensuite).

### Sortie attendue

```text output
saved: outputs/module02/usual_functions.png
sigmoid(-4, 0, 4): [0.018 0.5   0.982]
```

### Exemple 3 — un premier « modèle » affine

```python
import numpy as np

surfaces = np.array([30, 55, 80, 120])
slope, intercept = 2.5, 30.0  # k EUR per m2, fixed part in k EUR
prices = slope * surfaces + intercept
for s, p in zip(surfaces, prices):
    print(f"{s:>4} m2 -> {p:6.1f} k EUR")

# Inverse question: what surface for a budget of 250 k EUR?
budget = 250
print("surface for 250 k EUR:", (budget - intercept) / slope, "m2")

# Log transform compresses large values
incomes = np.array([12, 25, 40, 80, 400])
print("log10(incomes):", np.log10(incomes).round(2))
```

### Explication du code

La prédiction applique la même formule à toutes les surfaces. Pour la question inverse, on résout l'équation 2,5·s + 30 = 250, soit s = (250 − 30) / 2,5 = 88 m². Le logarithme en base 10 transforme des revenus allant de 12 à 400 (rapport 33) en valeurs entre 1,08 et 2,6 : les écarts entre petites valeurs deviennent visibles.

### Sortie attendue

```text output
  30 m2 ->  105.0 k EUR
  55 m2 ->  167.5 k EUR
  80 m2 ->  230.0 k EUR
 120 m2 ->  330.0 k EUR
surface for 250 k EUR: 88.0 m2
log10(incomes): [1.08 1.4  1.6  1.9  2.6 ]
```

## Erreurs fréquentes

- **Écrire `x^2` au lieu de `x**2`** en Python : `^` est un opérateur binaire, `3 ^ 2` vaut 1.
- **Oublier les parenthèses** : `1 / 1 + np.exp(-x)` n'est pas la sigmoïde.
- **Logarithme de 0 ou d'un nombre négatif** : `-inf` ou `nan` (avec un avertissement NumPy). Utilisez `np.log1p(x)` = ln(1 + x) pour des données contenant des zéros.
- **Débordement de l'exponentielle** : `np.exp(1000)` donne `inf`. Les bibliothèques utilisent des formules stables (on le verra pour softmax).
- **Confondre symbolique et numérique** : un objet SymPy n'est pas un tableau NumPy ; convertissez avec `sp.lambdify` si besoin.

## Exercices guidés

1. Résolvez à la main puis avec SymPy : 5x − 7 = 2x + 8.
2. Développez (2x − 1)² avec SymPy et vérifiez le résultat pour x = 3.
3. Tracez f(x) = eˣ et g(x) = ln(x) sur le même graphique pour x ∈ [0,1 ; 3] avec la droite y = x. Que remarquez-vous ? (*Indice : fonctions réciproques.*)
4. Pour quelle valeur de x la sigmoïde vaut-elle 0,9 ? Résolvez avec SymPy (`sp.solve(1/(1+sp.exp(-x)) - sp.Rational(9, 10), x)`).

Corrigés : [corrections.md](corrections.md), section « Leçon 2.1 ».

## Défis autonomes

- Un capital de 1000 € placé à 4 % par an vaut 1000 · 1,04ⁿ après n années. Utilisez un logarithme pour trouver n tel que le capital double, puis vérifiez avec une boucle.
- Tracez la sigmoïde σ(a·x) pour a = 0,5, 1 et 5. Comment le paramètre a modifie-t-il la courbe ? Quel lien avec la « confiance » d'un classifieur ?

## Quiz

1. Que vaut `2 + 3 * 4 ** 2` en Python ?
2. Quelle est la pente de f(x) = −2x + 7 ? Son ordonnée à l'origine ?
3. Pourquoi le logarithme est-il utile pour des variables comme les revenus ?
4. Entre quelles bornes varie la sigmoïde ? Que vaut σ(0) ?
5. Quelle est la différence entre SymPy et NumPy ?
6. Où la fonction (x − 4)² atteint-elle son minimum ?

Réponses : [corrections.md](corrections.md), section « Leçon 2.1 ».

## Résumé

- Python respecte les priorités opératoires ; la puissance s'écrit `**`.
- Le logarithme transforme produits en sommes et compresse les grandes valeurs.
- Une fonction affine a·x + b est déjà un modèle de prédiction ; la sigmoïde convertit un score en probabilité.
- SymPy calcule exactement (équations, développement, factorisation), NumPy calcule vite sur des tableaux.

## Pour aller plus loin

- Ouvrez `outputs/module02/usual_functions.png` et décrivez chaque courbe avec vos mots.
- Leçon suivante : [Vecteurs, matrices et algèbre linéaire](02_vecteurs_matrices.md).
