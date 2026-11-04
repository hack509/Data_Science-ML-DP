# I09 — k-means implémenté de zéro

- **Identifiant** : I09
- **Niveau** : intermédiaire
- **Module** : 5 (leçon 5.5)
- **Durée indicative** : 2 h

## Objectifs d'apprentissage

- Implémenter les deux étapes de k-means (affectation, mise à jour) de façon vectorisée.
- Gérer les cas particuliers (groupe vide, convergence).
- Comprendre l'effet de l'initialisation et l'intérêt de plusieurs lancements.

## Énoncé

Complétez `exercise.py` avec NumPy (sans scikit-learn) :

1. `init_centroids(X, k, seed=0)` : k observations **distinctes** tirées au hasard avec `np.random.default_rng(seed)` ; `ValueError` si k n'est pas entre 1 et n.
2. `assign(X, centroids)` : tableau des indices du centroïde le plus proche (distance euclidienne), sans boucle Python sur les observations.
3. `update(X, labels, centroids)` : nouveaux centroïdes = moyennes des groupes ; un groupe vide **garde** son ancien centroïde.
4. `inertia(X, centroids, labels)` : somme des distances au carré à son centroïde (`float`).
5. `kmeans(X, k, n_iter=100, tol=1e-6, seed=0)` → `(centroids, labels, inertia, n_iterations)` ; arrêt quand le plus grand déplacement d'une coordonnée de centroïde est inférieur à `tol`.
6. `best_of(X, k, n_init=10, seed=0)` : lance `kmeans` avec les graines `seed`, `seed + 1`, ... et renvoie le résultat de plus faible inertie.

## Données d'entrée

`datasets/generated/customers.csv` standardisé (6 variables), et de petits exemples dans les tests.

## Sortie attendue

Avec k = 4 sur les clients standardisés, `best_of` retrouve les segments du générateur (indice de Rand ajusté > 0,95) avec une inertie proche de celle de scikit-learn (≈ 629,5).

## Contraintes

- `assign` doit être vectorisée (broadcasting) : `X[:, None, :] - centroids[None, :, :]`.
- Ne pas utiliser `sklearn.cluster`.

## Indices

- **Indice 1** : le tableau des distances au carré a la forme (n, k) ; `argmin(axis=1)` donne les affectations.
- **Indice 2** : `centroids[labels]` donne, pour chaque observation, le centroïde de son groupe.
- **Indice 3** : copiez les centroïdes avant de les modifier dans `update`.

## Vérification

```bash
python -m pytest exercises/intermediate/I09_kmeans_from_scratch -q
```

Solution : `solutions/intermediate/I09_kmeans_from_scratch/solution.py`.
