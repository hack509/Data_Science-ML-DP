# I07 — Pipeline de régression sur les prix immobiliers

- **Identifiant** : I07
- **Niveau** : intermédiaire
- **Module** : 5 (leçons 5.2 et 5.6)
- **Durée indicative** : 2 h

## Objectifs d'apprentissage

- Construire un `ColumnTransformer` qui traite colonnes numériques, nominales et ordinales.
- Comparer plusieurs modèles par validation croisée, sans toucher au jeu de test.
- Calculer les métriques de régression sur le jeu de test.

## Énoncé

Complétez `exercise.py` :

1. `split(df, seed=0)` → `X_train, X_test, y_train, y_test` (80/20, `random_state=seed`, cible `price_keur`).
2. `build_pipeline(regressor)` → `Pipeline` à deux étapes nommées `"preprocess"` et `"model"`. Le `ColumnTransformer` contient trois blocs nommés :
   - `"num"` pour `NUMERIC` : imputation médiane + standardisation ;
   - `"nom"` pour `NOMINAL` (`neighborhood`) : imputation du mode + one-hot avec `handle_unknown="ignore"` ;
   - `"ord"` pour `ORDINAL` (`energy_class`) : imputation du mode + `OrdinalEncoder` avec l'ordre A < B < ... < G.
3. `evaluate(pipeline, X_test, y_test)` → `{"mae", "rmse", "r2"}` (des `float`).
4. `candidate_models()` → dictionnaire `{"linear", "ridge", "random_forest"}` de régresseurs non entraînés.
5. `compare_models(X_train, y_train, cv=5)` → DataFrame indexé par le nom du modèle, colonnes `mean_mae` et `std_mae` (MAE de validation croisée avec `KFold(n_splits=cv, shuffle=True, random_state=0)`), trié par `mean_mae` croissante.

## Données d'entrée

`datasets/generated/house_prices.csv` (800 logements).

## Sortie attendue

- La régression linéaire dans le pipeline obtient un R² de test supérieur à 0,9.
- Le pipeline accepte un logement avec un quartier inconnu et des valeurs manquantes sans erreur.
- `compare_models` renvoie trois lignes ; la meilleure MAE moyenne est inférieure à 30 k€.

## Contraintes

- Le jeu de test n'est utilisé que dans `evaluate`.
- Aucun prétraitement en dehors du pipeline.

## Indices

- **Indice 1** : `OrdinalEncoder(categories=[list("ABCDEFG")])` (une liste par colonne).
- **Indice 2** : `cross_val_score(..., scoring="neg_mean_absolute_error")` renvoie des valeurs négatives : prenez l'opposé.
- **Indice 3** : réutilisez la structure de l'exemple 1 de la leçon 5.6.

## Vérification

```bash
python -m pytest exercises/intermediate/I07_regression_pipeline -q
```

Solution : `solutions/intermediate/I07_regression_pipeline/solution.py`.
