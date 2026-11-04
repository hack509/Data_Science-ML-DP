# I04 — Descente de gradient pour la régression linéaire

- **Identifiant** : I04
- **Niveau** : intermédiaire
- **Module** : 2 (leçon 2.5), jalon du module
- **Durée indicative** : 2 h

## Objectifs d'apprentissage

- Écrire la perte MSE et son gradient sous forme matricielle.
- Implémenter une descente de gradient complète avec historique et critère d'arrêt.
- Vérifier une implémentation de gradient par différences finies (*gradient checking*).
- Retrouver la solution exacte des moindres carrés.

## Énoncé

Complétez `exercise.py` (NumPy autorisé) :

1. `standardize(X)` : renvoie `(X_standardisé, moyennes, écarts_types)` colonne par colonne (`ddof=0`) ; `ValueError` si une colonne est constante.
2. `predict(X, w, b)` : X·w + b.
3. `mse(y, y_pred)` : erreur quadratique moyenne (renvoyer un `float`).
4. `gradients(X, y, w, b)` : `(grad_w, grad_b)` de la MSE : grad_w = (2/n)·Xᵀ(ŷ − y), grad_b = (2/n)·Σ(ŷ − y).
5. `fit_gradient_descent(X, y, learning_rate=0.1, n_iter=1000, tol=None)` : part de w = 0, b = 0 ; à chaque itération, ajoute la perte **courante** à `history`, puis met à jour w et b. S'arrête si la perte devient infinie/NaN (lever `FloatingPointError`) ou si `tol` est donné et que la perte a diminué de moins de `tol` depuis l'itération précédente. Renvoie `(w, b, history)`.
6. `to_original_units(w, b, mean, std)` : convertit des coefficients appris sur données standardisées en coefficients sur données d'origine.

## Données d'entrée

`datasets/generated/students.csv` : variables `study_hours_week`, `attendance_rate`, `sleep_hours`, `previous_score` ; cible `final_score`.

## Sortie attendue

Sur ces données standardisées, avec `learning_rate=0.1` et 1000 itérations, la MSE finale vaut environ **3.02** (RMSE ≈ 1,74) et les coefficients reconvertis égalent ceux de `np.linalg.lstsq` à 10⁻⁴ près. La perte doit décroître à chaque itération.

## Contraintes

- Pas de scikit-learn.
- Calculer tous les gradients **avant** de mettre à jour les paramètres.
- Implémentation vectorisée : pas de boucle sur les observations.

## Indices

- **Indice 1** : avec X de forme (n, p) et l'erreur de forme (n,), `X.T @ error` a la forme (p,).
- **Indice 2** : si x_std = (x − m)/s alors w·x_std = (w/s)·x − w·m/s : la pente d'origine est w/s et l'ordonnée à l'origine diminue de Σ w·m/s.
- **Indice 3** : pour tester votre gradient, perturbez un poids de ±10⁻⁶ et comparez (L(w+h) − L(w−h)) / 2h à votre formule.

## Vérification

```bash
python -m pytest exercises/intermediate/I04_gradient_descent -q
```

Solution : `solutions/intermediate/I04_gradient_descent/solution.py`.
