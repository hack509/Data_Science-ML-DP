# Corrigés — Module 2 : Mathématiques et statistiques

Réponses aux quiz et corrigés des exercices guidés. Le code est exécuté par les tests automatiques.

## Leçon 2.1

### Quiz

1. 50 : la puissance d'abord (4² = 16), puis la multiplication (48), puis l'addition.
2. Pente −2 (la fonction décroît de 2 quand x augmente de 1), ordonnée à l'origine 7.
3. Il compresse les grandes valeurs et rend les distributions asymétriques plus symétriques ; il transforme les rapports en différences.
4. Entre 0 et 1 (exclus) ; σ(0) = 0,5.
5. SymPy manipule des expressions exactes (symboles, fractions) ; NumPy calcule numériquement et rapidement sur des tableaux de nombres approchés.
6. En x = 4, où (x − 4)² = 0.

### Exercices guidés

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x = sp.symbols("x")
print(sp.solve(sp.Eq(5 * x - 7, 2 * x + 8), x))      # 3x = 15 -> x = 5

expanded = sp.expand((2 * x - 1) ** 2)
print(expanded, "| at x=3:", expanded.subs(x, 3), "| check:", (2 * 3 - 1) ** 2)

t = np.linspace(0.1, 3, 200)
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(t, np.exp(t), label="exp(x)")
ax.plot(t, np.log(t), label="ln(x)")
ax.plot(t, t, "--", label="y = x")
ax.set_ylim(-2.5, 5)
ax.legend()
Path("outputs/module02").mkdir(parents=True, exist_ok=True)
fig.savefig("outputs/module02/exp_log.png", dpi=80)
plt.close(fig)

solution = sp.solve(1 / (1 + sp.exp(-x)) - sp.Rational(9, 10), x)
print(solution, "~", [float(s) for s in solution])
```

```text output
[5]
4*x**2 - 4*x + 1 | at x=3: 25 | check: 25
[log(9)] ~ [2.1972245773362196]
```

Exercice 3 : les courbes de eˣ et ln(x) sont symétriques par rapport à la droite y = x, car ce sont des fonctions réciproques. Exercice 4 : σ(x) = 0,9 pour x = ln(9) ≈ 2,197.

## Leçon 2.2

### Quiz

1. `(100,)` ; `(100, 3)`.
2. Les vecteurs sont orthogonaux (perpendiculaires).
3. Parce qu'on divise le produit scalaire par le produit des normes : seule la direction (l'angle) compte.
4. A (la matrice identité est l'élément neutre du produit matriciel).
5. Pour intégrer l'ordonnée à l'origine b dans le vecteur de paramètres : sa « variable » vaut toujours 1.
6. La matrice XᵀX n'est pas inversible (colinéarité) : la solution n'est pas unique et les coefficients deviennent instables.

### Exercices guidés

```python
import numpy as np
import pandas as pd

A = np.array([[1, 2], [3, 4]])
print(A @ np.array([1, -1]))                # [1-2, 3-4] = [-1, -1]

d1, d2 = np.array([2, 1, 0, 1]), np.array([1, 1, 1, 0])
print(round(d1 @ d2 / (np.linalg.norm(d1) * np.linalg.norm(d2)), 4))

M = np.array([[1, 2, 1], [2, -1, 0], [0, 1, 3]])
print(np.linalg.solve(M, np.array([9, 0, 11])))

students = pd.read_csv("datasets/generated/students.csv")
y = students["final_score"].to_numpy()
for features in (["study_hours_week", "attendance_rate", "sleep_hours"],
                 ["study_hours_week", "attendance_rate", "sleep_hours", "previous_score"]):
    X = np.column_stack([np.ones(len(students)), students[features].to_numpy()])
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    rmse = np.sqrt(np.mean((X @ w - y) ** 2))
    print(len(features), "features -> RMSE", round(rmse, 3))
```

```text output
[-1 -1]
0.7071
[1.23076923 2.46153846 2.84615385]
3 features -> RMSE 2.416
4 features -> RMSE 1.738
```

Exercice 4 : le RMSE diminue nettement, car `previous_score` est un indicateur (bruité) du niveau de l'étudiant, la variable cachée qui influence toutes les notes.

## Leçon 2.3

### Quiz

1. Elle ne dépend que de l'ordre des valeurs, pas de leur amplitude : une valeur extrême ne la déplace pas.
2. La variance est en points² ; l'écart-type en points.
3. Les écarts sont mesurés par rapport à la moyenne de l'échantillon, qui est plus proche des données que la vraie moyenne : diviser par n sous-estimerait la variance. Diviser par n − 1 corrige ce biais.
4. Une relation linéaire décroissante forte.
5. Non : elle peut cacher une relation non linéaire (exemple quadratique de la leçon).
6. Pearson mesure une relation linéaire sur les valeurs ; Spearman mesure une relation monotone sur les rangs, et résiste mieux aux valeurs extrêmes.

### Exercices guidés

```python
import statistics

import numpy as np
import pandas as pd

data = [4, 8, 6, 5, 3, 8, 9, 8]
print(statistics.mean(data), statistics.median(data), statistics.mode(data))

values = np.array([2, 4, 4, 4, 5, 5, 7, 9])
print(np.var(values), round(np.var(values, ddof=1), 4))

students = pd.read_csv("datasets/generated/students.csv")
print(students.groupby("group")["study_hours_week"].agg(["mean", "median"]).round(2).to_string())
print(students[["math", "physics", "french"]].corr().round(2).to_string())
```

```text output
6.375 7.0 8
4.0 4.5714
        mean  median
group               
A      10.66     9.8
B       9.79     9.6
C       9.94     7.9
         math  physics  french
math     1.00     0.71    0.61
physics  0.71     1.00    0.58
french   0.61     0.58    1.00
```

Exercice 1 : triée, la série est 3, 4, 5, 6, 8, 8, 8, 9 ; médiane = (6 + 8)/2 = 7 ; mode = 8. Exercice 4 : les trois notes sont positivement corrélées entre elles (elles dépendent toutes du niveau de l'étudiant) ; math et physique le sont davantage car le générateur leur donne un poids plus fort pour ce niveau.

## Leçon 2.4

### Quiz

1. P(A | B) est la probabilité de A **parmi** les cas où B est vrai ; P(A et B) est la probabilité que les deux arrivent, sur l'ensemble des cas. P(A et B) = P(A | B) · P(B).
2. Une loi de Poisson.
3. Pour n assez grand, la moyenne d'un échantillon suit approximativement une loi normale centrée sur la vraie moyenne, d'écart-type σ/√n, quelle que soit la loi des données (si sa variance est finie).
4. Elle est divisée par 2 (√4 = 2).
5. Non. C'est la probabilité d'observer un résultat au moins aussi extrême **si** H₀ est vraie.
6. Parce qu'une différence peut être statistiquement significative mais minuscule en pratique (grands échantillons), ou importante mais non significative (petits échantillons).

### Exercices guidés

```python
import numpy as np
import pandas as pd
from scipy import stats

# 1. Dice: P(sum = 7) = 6/36 ; P(sum = 7 | first = 3) = P(second = 4) = 1/6
rng = np.random.default_rng(0)
dice = rng.integers(1, 7, size=(200_000, 2))
total = dice.sum(axis=1)
print("P(7) ~", round((total == 7).mean(), 3), "| P(7 | first=3) ~", round((total[dice[:, 0] == 3] == 7).mean(), 3))

# 2. Medical test (Bayes)
prevalence, sensitivity, false_positive = 0.01, 0.99, 0.05
p_positive = sensitivity * prevalence + false_positive * (1 - prevalence)
print("P(sick | positive) =", round(sensitivity * prevalence / p_positive, 3))

students = pd.read_csv("datasets/generated/students.csv")
group_b = students.loc[students["group"] == "B", "final_score"]
low, high = stats.t.interval(0.95, df=len(group_b) - 1, loc=group_b.mean(), scale=stats.sem(group_b))
print(f"group B: mean={group_b.mean():.2f}, CI=[{low:.2f}, {high:.2f}]")

passed_a = students.loc[students["group"] == "A", "passed"]
passed_c = students.loc[students["group"] == "C", "passed"]
test = stats.ttest_ind(passed_c, passed_a, equal_var=False)
print(f"pass rate A={passed_a.mean():.3f}, C={passed_c.mean():.3f}, p-value={test.pvalue:.3f}")
```

```text output
P(7) ~ 0.168 | P(7 | first=3) ~ 0.171
P(sick | positive) = 0.167
group B: mean=9.41, CI=[8.91, 9.92]
pass rate A=0.398, C=0.543, p-value=0.051
```

Exercice 2 : malgré un test « fiable à 99 % », seulement 17 % des personnes testées positives sont malades, car la maladie est rare : la plupart des positifs sont des faux positifs parmi les 99 % de personnes saines. Exercice 4 : la p-value est juste au-dessus de 0,05 ; le groupe n'ayant aucun effet dans le générateur, conclure à une différence serait une erreur.

## Leçon 2.5

### Quiz

1. La pente de la tangente à la courbe au point x, c'est-à-dire la vitesse de variation de f.
2. 5x⁴ ; 3e³ˣ (règle de la chaîne).
3. Vers la plus forte augmentation de la fonction. On le soustrait pour aller vers la plus forte **diminution** de la perte.
4. Les pas dépassent le minimum : la perte oscille ou augmente (divergence), jusqu'à `inf` ou `nan`.
5. Pour que toutes les directions aient des échelles comparables : la « vallée » est plus ronde, un même taux d'apprentissage convient à tous les paramètres et la convergence est plus rapide.
6. Un réseau est une composition de fonctions (couches) ; la rétropropagation applique la règle de la chaîne couche par couche pour obtenir le gradient de la perte par rapport à chaque poids.

### Exercices guidés

```python
import numpy as np
import sympy as sp

x = sp.symbols("x")
f = 4 * x**3 - 2 * x + 5
df = sp.diff(f, x)
numeric = lambda x0, h=1e-5: ((4 * (x0 + h) ** 3 - 2 * (x0 + h)) - (4 * (x0 - h) ** 3 - 2 * (x0 - h))) / (2 * h)
print(df, "| at 2:", df.subs(x, 2), "| numerical:", round(numeric(2.0), 4))

w1, w2 = sp.symbols("w1 w2")
g = w1**2 + 3 * w2**2 + w1 * w2
gradient = [sp.diff(g, w1), sp.diff(g, w2)]
print(gradient, "| at (1, 2):", [d.subs({w1: 1, w2: 2}) for d in gradient])

point = np.array([4.0, -3.0])
for _ in range(100):
    grad = np.array([2 * point[0] + point[1], 6 * point[1] + point[0]])
    point -= 0.1 * grad
print("minimum found:", point.round(5))
```

```text output
12*x**2 - 2 | at 2: 46 | numerical: 46.0
[2*w1 + w2, w1 + 6*w2] | at (1, 2): [4, 13]
minimum found: [ 0. -0.]
```

Exercice 4 : avec `learning_rate = 1.2`, la perte augmente à chaque itération et les paramètres explosent : la descente diverge. Pour cette perte sur une variable standardisée, il faut un taux inférieur à 1.
