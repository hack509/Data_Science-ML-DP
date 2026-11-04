# Leçon 2.2 — Vecteurs, matrices et algèbre linéaire

## Prérequis

- Leçon 1.4 (NumPy) et leçon 2.1.

## Objectifs d'apprentissage

- Représenter des données comme des vecteurs et des matrices.
- Calculer norme, produit scalaire, distance et similarité cosinus.
- Effectuer produit matrice-vecteur, produit matriciel, transposée et inverse.
- Résoudre un système linéaire et ajuster une droite par moindres carrés.
- Relier ces opérations aux modèles de Machine Learning.

## Concepts clés

- **Vecteur** (*vector*) : liste ordonnée de nombres ; une observation décrite par n variables est un vecteur de dimension n.
- **Matrice** (*matrix*) : tableau de nombres à m lignes et n colonnes ; un jeu de données de m observations et n variables.
- **Norme** (*norm*) : longueur d'un vecteur, ‖v‖ = √(v₁² + ... + vₙ²).
- **Produit scalaire** (*dot product*) : u · v = Σ uᵢvᵢ ; mesure à quel point deux vecteurs « vont dans la même direction ».
- **Transposée** (*transpose*) : Aᵀ échange lignes et colonnes.
- **Matrice identité** (*identity matrix*) : I, avec des 1 sur la diagonale ; A·I = A.
- **Inverse** (*inverse*) : A⁻¹ telle que A·A⁻¹ = I (n'existe que pour certaines matrices carrées).
- **Moindres carrés** (*least squares*) : méthode qui choisit les paramètres minimisant la somme des erreurs au carré.

## Explication détaillée

### Les données sont des matrices

Le fichier `students.csv` contient 300 étudiants et plusieurs variables numériques. Si l'on garde 3 colonnes (heures d'étude, assiduité, sommeil), on obtient une matrice X de forme 300 × 3 : chaque **ligne** est un étudiant (un vecteur de dimension 3), chaque **colonne** une variable. Presque tous les algorithmes de ML manipulent ce type de matrice.

### Opérations sur les vecteurs

- Addition et multiplication par un nombre : composante par composante.
- Norme euclidienne : la longueur, par Pythagore.
- Distance entre deux points : ‖u − v‖. C'est la base des k plus proches voisins et de k-means (module 5).
- Produit scalaire : u · v = ‖u‖ ‖v‖ cos(θ), où θ est l'angle entre les vecteurs. Si le produit scalaire est nul, les vecteurs sont **orthogonaux** (perpendiculaires).
- Similarité cosinus : cos(θ) = u · v / (‖u‖ ‖v‖), entre −1 et 1. Elle compare des directions indépendamment de la longueur ; on l'utilise pour comparer des textes (module 8).

### Produit matrice-vecteur : la prédiction linéaire

Un modèle linéaire prédit ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b. Pour toutes les observations à la fois : **ŷ = X·w + b**. Chaque élément de ŷ est le produit scalaire d'une ligne de X avec le vecteur de poids w. Une matrice de forme (m, n) multipliée par un vecteur de dimension n donne un vecteur de dimension m.

### Produit matriciel

Le produit C = A·B n'est défini que si le nombre de colonnes de A égale le nombre de lignes de B : (m, n)·(n, p) → (m, p). L'élément Cᵢⱼ est le produit scalaire de la ligne i de A et de la colonne j de B. Attention : en général **A·B ≠ B·A**. Une couche de réseau de neurones est exactement un produit matriciel suivi d'une fonction d'activation (module 6).

En NumPy, `A @ B` est le produit matriciel, alors que `A * B` est le produit **élément par élément** : ne les confondez pas.

### Systèmes linéaires et inverse

Le système 2x + y = 7, x − y = −1 s'écrit A·v = c avec A = [[2, 1], [1, −1]], v = (x, y) et c = (7, −1). Si A est inversible, v = A⁻¹·c. En pratique, on n'inverse pas la matrice : on utilise `np.linalg.solve(A, c)`, plus rapide et plus précis. Une matrice n'est pas inversible lorsque ses colonnes sont **colinéaires** (une colonne est combinaison des autres) : en ML, c'est le problème de la **multicolinéarité** (deux variables qui disent la même chose, comme une surface en m² et la même surface en pieds²).

### Moindres carrés

Avec plus d'observations que d'inconnues (300 étudiants, 4 paramètres), le système X·w = y n'a généralement pas de solution exacte. On cherche alors le w qui minimise la somme des carrés des erreurs ‖X·w − y‖². La solution vérifie les **équations normales** XᵀX·w = Xᵀy. `np.linalg.lstsq` la calcule de façon stable. C'est exactement la régression linéaire (module 5).

## Intuition mathématique

```text
Produit scalaire :   u · v = Σᵢ uᵢ vᵢ = ‖u‖ ‖v‖ cos(θ)
Prédiction :         ŷ = X w + b            (m × n) · (n) → (m)
Produit matriciel :  (AB)ᵢⱼ = Σₖ Aᵢₖ Bₖⱼ    (m × n) · (n × p) → (m × p)
Moindres carrés :    w* = argmin ‖Xw − y‖²   ⇔   XᵀX w* = Xᵀ y
```

Astuce pratique : on ajoute à X une colonne de 1 pour intégrer l'ordonnée à l'origine b dans w. Ainsi ŷ = X·w suffit.

## Applications concrètes

- Systèmes de recommandation : similarité cosinus entre profils d'utilisateurs.
- Moteurs de recherche : comparaison de vecteurs de documents (TF-IDF, module 8).
- Réseaux de neurones : chaque couche est un produit matriciel.
- Images : une image en niveaux de gris est une matrice de pixels (module 9).

## Exemples de code Python

### Exemple 1 — vecteurs, normes, distances, similarité

```python
import numpy as np

u = np.array([3.0, 4.0])
v = np.array([4.0, 3.0])
w = np.array([-4.0, 3.0])

print("norm(u):", np.linalg.norm(u))
print("distance(u, v):", round(np.linalg.norm(u - v), 4))
print("u . v =", u @ v, "| u . w =", u @ w, "(orthogonal)")


def cosine_similarity(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


print("cos(u, v):", round(cosine_similarity(u, v), 3))
print("cos(u, 2u):", round(cosine_similarity(u, 2 * u), 3))

# Two students described by (study hours, attendance %, sleep hours)
alice = np.array([12, 95, 7.5])
bruno = np.array([4, 60, 6.0])
print("raw distance:", round(np.linalg.norm(alice - bruno), 2))
```

### Explication du code

- ‖(3, 4)‖ = √(9 + 16) = 5 : le triangle 3-4-5.
- u · w = 3 × (−4) + 4 × 3 = 0 : u et w sont orthogonaux.
- La similarité cosinus entre u et 2u vaut 1 : même direction, longueur différente.
- La distance entre Alice et Bruno est dominée par l'assiduité (écart de 35 points) alors que l'écart d'heures d'étude (8 h) est pédagogiquement plus important : il faut **standardiser** les variables avant de calculer des distances (leçon 1.4).

### Sortie attendue

```text output
norm(u): 5.0
distance(u, v): 1.4142
u . v = 24.0 | u . w = 0.0 (orthogonal)
cos(u, v): 0.96
cos(u, 2u): 1.0
raw distance: 35.93
```

### Exemple 2 — produit matriciel et système linéaire

```python
import numpy as np

A = np.array([[2, 1], [1, -1]])
B = np.array([[1, 2], [0, 1]])
print("A @ B =\n", A @ B)
print("B @ A =\n", B @ A)
print("A * B (element-wise) =\n", A * B)
print("A.T =\n", A.T)

c = np.array([7, -1])
solution = np.linalg.solve(A, c)
print("solution (x, y):", solution)
print("check A @ solution:", A @ solution)
print("det(A):", round(np.linalg.det(A), 6))
A_inv = np.linalg.inv(A)
print("A @ A_inv =\n", (A @ A_inv).round(10))

singular = np.array([[1, 2], [2, 4]])  # second column = 2 x first column
print("det(singular):", np.linalg.det(singular))
try:
    np.linalg.solve(singular, c)
except np.linalg.LinAlgError as exc:
    print("LinAlgError:", exc)
```

### Explication du code

On vérifie que A·B ≠ B·A et que `*` calcule autre chose que `@`. `np.linalg.solve` trouve (x, y) = (2, 3), et le produit A·solution redonne bien le second membre. Le déterminant non nul indique que A est inversible. La matrice `singular` a des colonnes proportionnelles : déterminant nul, pas de solution unique, et NumPy lève une `LinAlgError`.

### Sortie attendue

```text output
A @ B =
 [[2 5]
 [1 1]]
B @ A =
 [[ 4 -1]
 [ 1 -1]]
A * B (element-wise) =
 [[ 2  2]
 [ 0 -1]]
A.T =
 [[ 2  1]
 [ 1 -1]]
solution (x, y): [2. 3.]
check A @ solution: [ 7. -1.]
det(A): -3.0
A @ A_inv =
 [[1. 0.]
 [0. 1.]]
det(singular): 0.0
LinAlgError: Singular matrix
```

### Exemple 3 — moindres carrés sur les données des étudiants

```python
import numpy as np
import pandas as pd

students = pd.read_csv("datasets/generated/students.csv")
features = ["study_hours_week", "attendance_rate", "sleep_hours"]
X = students[features].to_numpy()
y = students["final_score"].to_numpy()

X_design = np.column_stack([np.ones(len(X)), X])  # column of ones -> intercept
print("X_design shape:", X_design.shape)

w, residuals, rank, _ = np.linalg.lstsq(X_design, y, rcond=None)
for name, value in zip(["intercept"] + features, w):
    print(f"{name:<18} {value:8.3f}")

w_normal = np.linalg.solve(X_design.T @ X_design, X_design.T @ y)
print("same as normal equations:", np.allclose(w, w_normal))

predictions = X_design @ w
rmse = np.sqrt(np.mean((predictions - y) ** 2))
print("RMSE:", round(rmse, 3), "| first predictions:", predictions[:3].round(2), "| true:", y[:3])
```

### Explication du code

- `np.column_stack` ajoute la colonne de 1 : le premier coefficient devient l'ordonnée à l'origine.
- `lstsq` renvoie les coefficients qui minimisent l'erreur quadratique. Chaque coefficient se lit « toutes choses égales par ailleurs » : une heure d'étude de plus par semaine est associée à une augmentation d'environ 0,18 point de la note finale.
- Les équations normales donnent le même résultat (mais sont moins stables numériquement sur des problèmes difficiles).
- L'erreur moyenne (RMSE) d'environ 2,4 points montre que ces trois variables n'expliquent qu'une partie de la note : il manque le « niveau » de l'étudiant, une variable cachée. Rappel : ces données sont **synthétiques**, les coefficients ne disent rien de vrais étudiants.

### Sortie attendue

```text output
X_design shape: (300, 4)
intercept             0.883
study_hours_week      0.178
attendance_rate       5.733
sleep_hours           0.304
same as normal equations: True
RMSE: 2.416 | first predictions: [ 8.31  9.04 11.11] | true: [ 8.33 11.63 11.26]
```

## Erreurs fréquentes

- **Confondre `*` et `@`** : produit élément par élément contre produit matriciel.
- **Formes incompatibles** : `matmul: Input operand 1 has a mismatch in its core dimension`. Vérifiez `A.shape` et `B.shape` : (m, n) @ (n, p).
- **Inverser une matrice pour résoudre un système** : préférez `np.linalg.solve` ou `lstsq`.
- **Oublier la colonne de 1** : le modèle est alors forcé de passer par l'origine.
- **Calculer des distances sur des variables d'échelles différentes** sans les standardiser.
- **Ignorer la multicolinéarité** : coefficients instables et difficiles à interpréter.

## Exercices guidés

1. Calculez à la main puis avec NumPy le produit de A = [[1, 2], [3, 4]] et du vecteur (1, −1).
2. Calculez la similarité cosinus entre les vecteurs de mots (fréquences) d1 = (2, 1, 0, 1) et d2 = (1, 1, 1, 0).
3. Résolvez le système x + 2y + z = 9, 2x − y = 0, y + 3z = 11 avec `np.linalg.solve`.
4. Refaites l'exemple 3 en ajoutant la variable `previous_score`. Le RMSE diminue-t-il ? Pourquoi ?

Corrigés : [corrections.md](corrections.md), section « Leçon 2.2 ».

L'exercice **B06** (`exercises/beginner/B06_matrix_operations`) vous fait implémenter ces opérations sans NumPy.

## Défis autonomes

- Implémentez le produit matriciel avec trois boucles imbriquées et comparez le résultat à `@` sur des matrices aléatoires 50 × 40 et 40 × 30.
- Montrez numériquement que (A·B)ᵀ = Bᵀ·Aᵀ pour des matrices aléatoires.
- Créez une colonne `study_minutes = 60 * study_hours_week` et ajoutez-la à X. Calculez `np.linalg.matrix_rank(X_design)` et expliquez.

## Quiz

1. Quelle est la forme du résultat d'un produit (100, 5) @ (5,) ? Et (100, 5) @ (5, 3) ?
2. Que signifie un produit scalaire nul ?
3. Pourquoi la similarité cosinus est-elle insensible à la longueur des vecteurs ?
4. Que vaut A·I ?
5. Pourquoi ajoute-t-on une colonne de 1 à la matrice des variables ?
6. Que se passe-t-il si deux colonnes de X sont proportionnelles ?

Réponses : [corrections.md](corrections.md), section « Leçon 2.2 ».

## Résumé

- Un jeu de données numérique est une matrice : lignes = observations, colonnes = variables.
- Norme, distance, produit scalaire et cosinus mesurent tailles, écarts et ressemblances.
- ŷ = X·w (+ b) : la prédiction d'un modèle linéaire est un produit matrice-vecteur.
- `np.linalg.solve` résout les systèmes carrés, `np.linalg.lstsq` les problèmes de moindres carrés.

## Pour aller plus loin

- Exercice **B06** — opérations matricielles.
- Module 5, leçon 5.2 : la même régression avec scikit-learn.
- Leçon suivante : [Statistiques descriptives, corrélation et covariance](03_statistiques_descriptives.md).
