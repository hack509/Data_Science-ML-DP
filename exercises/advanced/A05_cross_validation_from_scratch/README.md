# A05 — Validation croisée de zéro et fuite de données

- **Identifiant** : A05
- **Niveau** : avancé
- **Module** : 5 (leçons 5.1 et 5.6)
- **Durée indicative** : 2 h 30

## Objectifs d'apprentissage

- Comprendre exactement comment sont construits les plis d'une validation croisée.
- Réentraîner une copie neuve du modèle à chaque pli (`sklearn.base.clone`).
- Mesurer expérimentalement l'effet d'une fuite de données.

## Énoncé

Complétez `exercise.py` :

1. `kfold_indices(n, k, shuffle=False, seed=None)` → liste de k couples `(indices_entraînement, indices_validation)` **identiques** à ceux de `KFold(n_splits=k, shuffle=shuffle, random_state=seed)` de scikit-learn. `ValueError` si k n'est pas entre 2 et n.
2. `cross_validate(model, X, y, k=5, shuffle=False, seed=None)` → tableau NumPy des k accuracies (méthode `score` du modèle), en entraînant à chaque pli une **copie neuve** du modèle.
3. `selection_cv(X, y, n_features=20, k=5, leak=False)` → accuracy moyenne d'une régression logistique précédée d'une sélection `SelectKBest(f_classif, k=n_features)` :
   - `leak=True` : la sélection est faite **une fois sur toutes les données**, puis on valide (incorrect) ;
   - `leak=False` : la sélection fait partie d'un pipeline, refaite dans chaque pli (correct).

## Données d'entrée

- `binary_classification.csv` pour comparer avec `cross_val_score`.
- Des données de bruit pur (100 lignes, 2000 variables, étiquettes aléatoires) pour mesurer la fuite.

## Sortie attendue

- Mêmes plis et mêmes scores que scikit-learn.
- Sur du bruit pur : accuracy avec fuite > 0,75 alors que la version correcte reste proche de 0,5 (entre 0,3 et 0,7).

## Contraintes

- Ne pas utiliser `KFold` ni `cross_val_score` dans `exercise.py` (les tests les utilisent pour comparer).
- Avec `n` non divisible par `k`, les premiers plis ont un élément de plus (comme scikit-learn).

## Indices

- **Indice 1** : tailles des plis : `n // k` pour tous, plus 1 pour les `n % k` premiers.
- **Indice 2** : scikit-learn mélange avec `np.random.RandomState(seed).shuffle(indices)`.
- **Indice 3** : `clone(model)` crée un modèle non entraîné avec les mêmes hyperparamètres.

## Vérification

```bash
python -m pytest exercises/advanced/A05_cross_validation_from_scratch -q
```

Solution : `solutions/advanced/A05_cross_validation_from_scratch/solution.py`.
