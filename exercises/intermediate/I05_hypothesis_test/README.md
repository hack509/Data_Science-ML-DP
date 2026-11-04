# I05 — Intervalle de confiance et test d'hypothèse

- **Identifiant** : I05
- **Niveau** : intermédiaire
- **Module** : 2 (leçon 2.4)
- **Durée indicative** : 2 h

## Objectifs d'apprentissage

- Calculer un intervalle de confiance de Student et un intervalle bootstrap.
- Implémenter le test t de Welch à la main et le comparer à SciPy.
- Rapporter une taille d'effet (d de Cohen) en plus de la p-value.

## Énoncé

Complétez `exercise.py` :

1. `mean_confidence_interval(sample, confidence=0.95)` → `(moyenne, borne_basse, borne_haute)` avec la loi de Student à n − 1 degrés de liberté ; `ValueError` si moins de 2 valeurs.
2. `bootstrap_ci(sample, n_boot=5000, confidence=0.95, seed=0)` → `(borne_basse, borne_haute)` par la méthode des percentiles, avec `np.random.default_rng(seed)`.
3. `welch_t_test(a, b)` → `(t, ddl, p_value)` **calculés à la main** (bilatéral). Seule la fonction `stats.t.sf` (ou `cdf`) de SciPy est autorisée pour obtenir la p-value.
4. `cohens_d(a, b)` → (moyenne_a − moyenne_b) / écart-type combiné (*pooled*).
5. `compare_groups(df, group_col, value_col, group_a, group_b, alpha=0.05)` → dictionnaire avec `mean_a, mean_b, difference, t, df, p_value, cohens_d, reject_h0`.

## Données d'entrée

`datasets/generated/students.csv` : comparez `final_score` entre les groupes `C` et `A`, puis `study_hours_week` entre étudiants admis (`passed == 1`) et non admis.

## Sortie attendue

```text
compare_groups(students, "group", "final_score", "C", "A")
difference ≈ 0.70, t ≈ 1.80, p_value ≈ 0.074, reject_h0 = False
```

Votre `welch_t_test` doit donner exactement les mêmes t et p-value que `scipy.stats.ttest_ind(a, b, equal_var=False)`.

## Contraintes

- `welch_t_test` et `cohens_d` sans fonction de test toute faite.
- Utilisez `ddof=1` pour les variances d'échantillon.

## Indices

- **Indice 1** : t = (x̄a − x̄b) / √(sa²/na + sb²/nb).
- **Indice 2** : ddl de Welch-Satterthwaite = (va + vb)² / (va²/(na − 1) + vb²/(nb − 1)) avec va = sa²/na.
- **Indice 3** : p-value bilatérale = 2 × P(T > |t|) = `2 * stats.t.sf(abs(t), ddl)`.

## Vérification

```bash
python -m pytest exercises/intermediate/I05_hypothesis_test -q
```

Solution : `solutions/intermediate/I05_hypothesis_test/solution.py`. Questions de réflexion (réponses dans la solution) : la différence entre admis et non admis pour les heures d'étude est-elle significative ? Est-ce une preuve que travailler plus *cause* la réussite ?
