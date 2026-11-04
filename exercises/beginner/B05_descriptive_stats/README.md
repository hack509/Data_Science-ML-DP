# B05 — Statistiques descriptives sans bibliothèque

- **Identifiant** : B05
- **Niveau** : débutant
- **Module** : 2 (leçon 2.3)
- **Durée indicative** : 1 h 30

## Objectifs d'apprentissage

- Comprendre ce que calculent `np.mean`, `np.median`, `np.var`, `np.percentile`, `np.corrcoef` en les recodant.
- Maîtriser la différence entre variance de population (`ddof=0`) et d'échantillon (`ddof=1`).
- Gérer les cas limites (liste vide, écart-type nul).

## Énoncé

Complétez `exercise.py` **sans NumPy, pandas ni module `statistics`** (seuls `math` et `collections` sont autorisés) :

| Fonction | Résultat |
|---|---|
| `mean(values)` | moyenne |
| `median(values)` | médiane (moyenne des deux valeurs centrales si n est pair) |
| `modes(values)` | liste triée de **toutes** les valeurs les plus fréquentes |
| `variance(values, ddof=0)` | Σ(x − x̄)² / (n − ddof) |
| `std(values, ddof=0)` | racine de la variance |
| `quantile(values, q)` | quantile par interpolation linéaire (méthode par défaut de NumPy) |
| `iqr(values)` | Q3 − Q1 |
| `covariance(x, y, ddof=1)` | covariance |
| `correlation(x, y)` | corrélation de Pearson |
| `zscores(values)` | liste des (x − x̄) / σ (avec ddof=0) |

Toutes les fonctions lèvent `ValueError` pour une liste vide. `variance` lève `ValueError` si n − ddof ≤ 0, `quantile` si q n'est pas dans [0, 1], `covariance` si les longueurs diffèrent, `zscores` si l'écart-type est nul.

## Données d'entrée

Listes de nombres (dans les tests) et les colonnes de `datasets/generated/students.csv`.

## Sortie attendue

```text
>>> modes([1, 2, 2, 3, 3])
[2, 3]
>>> variance([2, 4, 4, 4, 5, 5, 7, 9])
4.0
>>> quantile([1, 2, 3, 4], 0.25)
1.75
```

## Contraintes

- Pas de NumPy/pandas/statistics dans `exercise.py` (les tests, eux, les utilisent pour vérifier vos résultats).
- Ne modifiez pas les listes reçues (utilisez `sorted()`, pas `.sort()`).

## Indices

- **Indice 1** : pour le quantile, la position (éventuellement fractionnaire) dans la liste triée est q × (n − 1). Interpolez entre l'élément d'indice `floor(position)` et le suivant.
- **Indice 2** : `collections.Counter(values).most_common()` donne les fréquences.
- **Indice 3** : réutilisez vos fonctions : `correlation` peut appeler `covariance` et `std`.

## Vérification

```bash
python -m pytest exercises/beginner/B05_descriptive_stats -q
```

Solution : `solutions/beginner/B05_descriptive_stats/solution.py`.
