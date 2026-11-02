# Leçon 1.4 — NumPy : calcul numérique vectorisé

## Prérequis

- Leçons 1.1 à 1.3.
- Notion de tableau à deux dimensions (lignes × colonnes), comme dans un tableur.

## Objectifs d'apprentissage

- Créer des tableaux NumPy (`ndarray`) et connaître leurs attributs (`shape`, `dtype`, `ndim`).
- Indexer, découper (*slicing*) et filtrer avec des masques booléens.
- Effectuer des calculs vectorisés et des agrégations selon un axe.
- Comprendre le *broadcasting* et l'utiliser pour standardiser des données.
- Générer des nombres aléatoires de façon reproductible.

## Concepts clés

- **ndarray** : tableau multidimensionnel homogène (tous les éléments ont le même type).
- **Forme** (*shape*) : le nombre d'éléments selon chaque dimension, par exemple `(300, 4)` = 300 lignes, 4 colonnes.
- **Vectorisation** (*vectorization*) : appliquer une opération à tout un tableau sans boucle Python explicite.
- **Axe** (*axis*) : la dimension le long de laquelle on agrège ; `axis=0` réduit les lignes (résultat par colonne), `axis=1` réduit les colonnes (résultat par ligne).
- **Broadcasting** : règle qui permet de combiner des tableaux de formes différentes mais compatibles.
- **Masque booléen** (*boolean mask*) : tableau de `True`/`False` utilisé pour sélectionner des éléments.
- **Graine aléatoire** (*random seed*) : valeur initiale qui rend une suite de nombres aléatoires reproductible.

## Explication détaillée

### Pourquoi NumPy ?

Une liste Python peut contenir n'importe quoi : chaque élément est un objet indépendant, dispersé en mémoire. Un tableau NumPy stocke des nombres du même type **côte à côte** dans un bloc de mémoire contigu, et les opérations sont exécutées par du code compilé (C). Résultat : des calculs 10 à 100 fois plus rapides, et une écriture beaucoup plus proche des mathématiques. `notes * 2` multiplie toutes les notes, sans boucle.

NumPy est la fondation de tout l'écosystème : pandas stocke ses colonnes dans des tableaux NumPy, scikit-learn attend des tableaux NumPy, et les tenseurs PyTorch fonctionnent presque à l'identique.

### Créer des tableaux

- `np.array([1, 2, 3])` à partir d'une liste ;
- `np.zeros((3, 4))`, `np.ones(5)`, `np.full((2, 2), 7)` ;
- `np.arange(0, 10, 2)` (comme `range`) et `np.linspace(0, 1, 5)` (5 points régulièrement espacés, bornes incluses) ;
- `rng = np.random.default_rng(42)` puis `rng.normal(...)`, `rng.integers(...)`, `rng.choice(...)` pour l'aléatoire.

Le `dtype` (type des éléments) compte : `int64`, `float64`, `bool`. Mélanger entiers et réels donne des réels ; diviser des entiers avec `/` donne aussi des réels.

### Indexation et découpage

Pour un tableau 2D `X` de forme `(n_lignes, n_colonnes)` :

- `X[0]` : première ligne ; `X[:, 1]` : deuxième colonne ;
- `X[2:5, :2]` : lignes 2 à 4, colonnes 0 et 1 ;
- `X[X[:, 0] > 10]` : lignes dont la première colonne dépasse 10 (masque booléen).

Les combinaisons de masques utilisent `&` (et), `|` (ou), `~` (non), avec des **parenthèses obligatoires** : `(a > 1) & (a < 5)`.

**Attention** : un découpage (`X[2:5]`) renvoie une **vue** (*view*) qui partage la mémoire avec le tableau d'origine ; la modifier modifie l'original. Utilisez `.copy()` pour obtenir une copie indépendante. L'indexation par masque ou par liste d'indices renvoie toujours une copie.

### Agrégations et axes

`X.mean()` calcule la moyenne de tous les éléments. `X.mean(axis=0)` calcule la moyenne de chaque **colonne** (on « écrase » l'axe des lignes) ; `X.mean(axis=1)` la moyenne de chaque **ligne**. Même logique pour `sum`, `std`, `min`, `max`, `argmax`.

### Broadcasting

Quand on combine deux tableaux de formes différentes, NumPy compare les dimensions en partant de la droite ; deux dimensions sont compatibles si elles sont égales ou si l'une vaut 1. Exemple : `X` de forme `(300, 4)` moins `X.mean(axis=0)` de forme `(4,)` : la moyenne de chaque colonne est soustraite à chaque ligne. C'est ainsi qu'on **standardise** des données en une ligne.

## Intuition mathématique

Un tableau 1D est un **vecteur**, un tableau 2D une **matrice**. La standardisation (*z-score*) d'une colonne x s'écrit :

```text
z = (x − μ) / σ      avec μ = moyenne de x, σ = écart-type de x
```

Après standardisation, chaque colonne a une moyenne 0 et un écart-type 1. Beaucoup d'algorithmes (k plus proches voisins, SVM, réseaux de neurones, descente de gradient) fonctionnent mal si les variables ont des échelles très différentes (des m² et des nombres de pièces, par exemple). Le produit matriciel `X @ w` calcule en une opération une somme pondérée par ligne : c'est la prédiction d'un modèle linéaire pour toutes les observations à la fois (module 2, leçon 2.2).

## Applications concrètes

- Standardiser les variables avant un modèle de Machine Learning.
- Calculer en une ligne la moyenne de chaque étudiant sur une matrice de notes.
- Simuler des milliers de lancers de dés ou d'échantillons pour estimer une probabilité.
- Représenter une image en niveaux de gris comme une matrice de pixels (module 9).

## Exemples de code Python

### Exemple 1 — créer, inspecter, vectoriser

```python
import time

import numpy as np

scores = np.array([12.5, 8.0, 15.5, 10.0, 17.0])
print(scores.shape, scores.dtype, scores.ndim)
print("+1 bonus:", scores + 1)
print("mean:", scores.mean(), "| std:", round(scores.std(), 3))
print("passed mask:", scores >= 10)
print("passed scores:", scores[scores >= 10])

# Speed comparison: Python loop versus vectorized NumPy.
values = np.arange(1_000_000, dtype=np.float64)
start = time.perf_counter()
total_loop = 0.0
for v in values.tolist():
    total_loop += v * v
loop_time = time.perf_counter() - start
start = time.perf_counter()
total_numpy = float(np.sum(values * values))
numpy_time = time.perf_counter() - start
print("same result:", np.isclose(total_loop, total_numpy))
print("NumPy is faster:", numpy_time < loop_time)
```

### Explication du code

`scores >= 10` produit un tableau booléen ; utilisé entre crochets, il sélectionne les notes correspondantes. Dans la deuxième partie, la boucle Python et l'expression vectorisée calculent la même somme des carrés ; la version NumPy est nettement plus rapide (en général plusieurs dizaines de fois). On n'affiche pas les durées car elles varient d'une machine à l'autre.

### Sortie attendue

```text output
(5,) float64 1
+1 bonus: [13.5  9.  16.5 11.  18. ]
mean: 12.6 | std: 3.338
passed mask: [ True False  True  True  True]
passed scores: [12.5 15.5 10.  17. ]
same result: True
NumPy is faster: True
```

### Exemple 2 — matrice de notes, axes et broadcasting

```python
import numpy as np

rng = np.random.default_rng(seed=0)
# 6 students x 3 subjects (math, physics, french)
grades = np.round(rng.normal(loc=11, scale=3, size=(6, 3)), 1).clip(0, 20)
print(grades)
print("shape:", grades.shape)
print("mean per subject (axis=0):", grades.mean(axis=0).round(2))
print("mean per student (axis=1):", grades.mean(axis=1).round(2))
print("best student index:", grades.mean(axis=1).argmax())

weights = np.array([4, 3, 3])
weighted = grades @ weights / weights.sum()
print("weighted averages:", weighted.round(2))

# Broadcasting: standardize each column.
z = (grades - grades.mean(axis=0)) / grades.std(axis=0)
print("column means after standardization:", z.mean(axis=0).round(6))
print("column std after standardization:", z.std(axis=0).round(6))
```

### Explication du code

- `default_rng(seed=0)` rend le tirage reproductible : vous obtenez exactement les mêmes notes que dans la sortie ci-dessous.
- `.clip(0, 20)` borne les valeurs.
- `grades @ weights` est un **produit matrice-vecteur** : pour chaque ligne, somme des notes × coefficients. On obtient la moyenne pondérée des 6 étudiants sans boucle.
- `grades - grades.mean(axis=0)` : forme `(6, 3)` moins forme `(3,)` → broadcasting ligne par ligne.
- Les moyennes après standardisation valent 0 (aux erreurs d'arrondi près, d'où `round(6)` ; `-0.` est un zéro très légèrement négatif) et les écarts-types 1.

### Sortie attendue

```text output
[[11.4 10.6 12.9]
 [11.3  9.4 12.1]
 [14.9 13.8  8.9]
 [ 7.2  9.1 11.1]
 [ 4.  10.3  7.3]
 [ 8.8  9.4 10.1]]
shape: (6, 3)
mean per subject (axis=0): [ 9.6  10.43 10.4 ]
mean per student (axis=1): [11.63 10.93 12.53  9.13  7.2   9.43]
best student index: 2
weighted averages: [11.61 10.97 12.77  8.94  6.88  9.37]
column means after standardization: [-0. -0. -0.]
column std after standardization: [1. 1. 1.]
```

### Exemple 3 — vues, copies et reshape

```python
import numpy as np

a = np.arange(12)
m = a.reshape(3, 4)
print(m)
view = m[:, :2]
view[0, 0] = 99          # modifies m and a as well!
print("a[0] after modifying the view:", a[0])
safe = m[:, :2].copy()
safe[0, 0] = -1          # does not modify m
print("m[0, 0] after modifying the copy:", m[0, 0])
print("flatten:", m.ravel()[:5], "| transpose shape:", m.T.shape)
```

### Explication du code

`reshape` ne copie pas les données : `m` est une autre façon de voir `a`. La sous-matrice `view` partage elle aussi la mémoire, donc la modifier change `a`. Avec `.copy()`, la modification reste locale. `m.T` est la transposée (lignes ↔ colonnes).

### Sortie attendue

```text output
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
a[0] after modifying the view: 99
m[0, 0] after modifying the copy: 99
flatten: [99  1  2  3  4] | transpose shape: (4, 3)
```

## Erreurs fréquentes

- **`and` / `or` au lieu de `&` / `|`** sur des tableaux : `ValueError: The truth value of an array ... is ambiguous`. Écrivez `(a > 1) & (a < 5)`.
- **Oublier les parenthèses** : `a > 1 & a < 5` est évalué comme `a > (1 & a) < 5`.
- **Confondre `axis=0` et `axis=1`** : retenez que l'axe indiqué est celui qui **disparaît**.
- **Modifier une vue par accident** : utilisez `.copy()` quand vous voulez travailler sur un extrait indépendant.
- **Formes incompatibles** : `operands could not be broadcast together with shapes (6,3) (6,)`. Pour soustraire une valeur par ligne, utilisez `x[:, None]` ou `keepdims=True`.
- **Utiliser `np.random.seed` et l'ancienne API** : préférez `np.random.default_rng(seed)`.

## Exercices guidés

1. Créez un tableau de 10 températures avec `np.linspace(12, 30, 10)` et convertissez-les en degrés Fahrenheit (F = C × 9/5 + 32) sans boucle.
2. À partir de la matrice `grades` de l'exemple 2, comptez les notes inférieures à 10 par matière. *Indice : `(grades < 10).sum(axis=0)`.*
3. Soustrayez à chaque **ligne** de `grades` la moyenne de cette ligne. *Indice : `keepdims=True`.*
4. Simulez 10 000 lancers de deux dés avec `rng.integers(1, 7, size=(10000, 2))` et estimez la probabilité que la somme vaille 7 (valeur théorique 1/6 ≈ 0,167).

Corrigés : [corrections.md](corrections.md), section « Leçon 1.4 ».

## Défis autonomes

- Écrivez une fonction `min_max_scale(X)` qui ramène chaque colonne entre 0 et 1, et gérez le cas d'une colonne constante (division par zéro).
- Calculez la matrice des distances euclidiennes entre 5 points 2D **sans boucle**, avec le broadcasting (`X[:, None, :] - X[None, :, :]`).

## Quiz

1. Quelle est la forme du résultat de `np.zeros((300, 4)).mean(axis=0)` ?
2. Pourquoi les opérations NumPy sont-elles plus rapides que des boucles Python ?
3. Que renvoie `a[a > 0]` ?
4. Un découpage `a[1:3]` est-il une copie ou une vue ?
5. Peut-on additionner un tableau `(5, 3)` et un tableau `(3,)` ? Et un tableau `(5, 3)` et un tableau `(5,)` ?
6. À quoi sert la graine d'un générateur aléatoire ?

Réponses : [corrections.md](corrections.md), section « Leçon 1.4 ».

## Résumé

- Un `ndarray` est homogène, compact et rapide ; `shape` et `dtype` le décrivent.
- On sélectionne avec des indices, des tranches et des masques booléens (`&`, `|`, `~`).
- `axis=0` → résultat par colonne ; `axis=1` → résultat par ligne.
- Le broadcasting permet de standardiser en une ligne : `(X - X.mean(0)) / X.std(0)`.
- `X @ w` calcule toutes les sommes pondérées d'un coup : la base des modèles linéaires.

## Pour aller plus loin

- Exercice **B06** (`exercises/beginner/B06_matrix_operations`) : produits matriciels à la main puis avec NumPy.
- Module 2, leçon 2.2 : vecteurs et matrices.
- Leçon suivante : [pandas : DataFrames, CSV et JSON](05_pandas_csv_json.md).
