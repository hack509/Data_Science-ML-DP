# B06 — Opérations matricielles

- **Identifiant** : B06
- **Niveau** : débutant
- **Module** : 2 (leçon 2.2)
- **Durée indicative** : 1 h

## Objectifs d'apprentissage

- Comprendre le produit scalaire, le produit matrice-vecteur et le produit matriciel en les implémentant.
- Vérifier la compatibilité des formes et produire des messages d'erreur clairs.
- Résoudre un système 2 × 2 avec la règle de Cramer.

## Énoncé

Les vecteurs sont des listes de nombres et les matrices des listes de lignes (`[[1, 2], [3, 4]]`). Complétez `exercise.py` **sans NumPy** :

1. `shape(A)` : `(lignes, colonnes)` ; `ValueError` si la matrice est vide ou si les lignes n'ont pas toutes la même longueur.
2. `dot(u, v)` : produit scalaire ; `ValueError` si les longueurs diffèrent.
3. `norm(v)` : norme euclidienne.
4. `transpose(A)`.
5. `matvec(A, v)` : produit matrice-vecteur ; `ValueError` si les formes sont incompatibles.
6. `matmul(A, B)` : produit matriciel ; `ValueError` si les formes sont incompatibles.
7. `identity(n)` : matrice identité de taille n (avec des `1.0` et `0.0`).
8. `solve_2x2(A, b)` : solution `[x, y]` de A·(x, y) = b par la règle de Cramer ; `ValueError` si le déterminant est nul.

## Données d'entrée

Petites matrices dans les tests, plus des matrices aléatoires comparées à NumPy.

## Sortie attendue

```text
>>> matmul([[2, 1], [1, -1]], [[1, 2], [0, 1]])
[[2, 5], [1, 1]]
>>> solve_2x2([[2, 1], [1, -1]], [7, -1])
[2.0, 3.0]
```

## Contraintes

- Python standard uniquement dans `exercise.py`.
- Les fonctions ne modifient pas leurs arguments.

## Indices

- **Indice 1** : l'élément (i, j) de A·B est le produit scalaire de la ligne i de A et de la colonne j de B, c'est-à-dire de la ligne j de Bᵀ.
- **Indice 2** : `zip(*A)` donne les colonnes de A (pensez à les convertir en listes).
- **Indice 3** : Cramer : det = a·d − b·c, x = (e·d − b·f)/det, y = (a·f − e·c)/det pour le système a·x + b·y = e, c·x + d·y = f.

## Vérification

```bash
python -m pytest exercises/beginner/B06_matrix_operations -q
```

Solution : `solutions/beginner/B06_matrix_operations/solution.py`.
