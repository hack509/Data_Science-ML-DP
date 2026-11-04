# I02 — Valeurs manquantes et valeurs aberrantes

- **Identifiant** : I02
- **Niveau** : intermédiaire
- **Module** : 3 (leçons 3.1 et 3.2)
- **Durée indicative** : 2 h

## Objectifs d'apprentissage

- Écrire une fonction de nettoyage complète, reproductible et accompagnée d'un rapport chiffré.
- Imputer de façon conditionnelle (par groupe) en gardant la trace des valeurs imputées.
- Implémenter deux détecteurs de valeurs aberrantes (IQR, z-score).

## Énoncé

Complétez `exercise.py` :

1. `iqr_outliers(series, k=1.5)` : Series booléenne, `True` pour les valeurs hors de [Q1 − k·IQR ; Q3 + k·IQR].
2. `zscore_outliers(series, threshold=3.0)` : Series booléenne, `True` si |z| > seuil (écart-type pandas, `ddof=1`).
3. `clean_students(raw)` renvoie `(df_propre, rapport)` en appliquant **dans cet ordre** :
   1. suppression des doublons exacts ;
   2. normalisation de `group` (`strip` + majuscules) ;
   3. conversion de `attendance_rate` en réel entre 0 et 1 (`"90.4%"` → 0.904) ;
   4. colonne booléenne `math_absent`, puis conversion de `math` en nombre (`"absent"` → `NaN`, **non imputé**) ;
   5. les valeurs de `study_hours_week` hors de [0, 80] deviennent `NaN` ;
   6. colonnes `study_hours_week_was_missing`, `sleep_hours_was_missing`, `physics_was_missing` ;
   7. imputation de `study_hours_week` par la **médiane de son groupe**, de `sleep_hours` et `physics` par la médiane globale ;
   8. tri par `student_id` et index renuméroté.

   Le rapport est un dictionnaire : `rows_before`, `duplicates_removed`, `math_absent`, `impossible_values`, `imputed` (dictionnaire colonne → nombre de valeurs imputées) et `rows_after`.

## Données d'entrée

`datasets/raw/students_raw.csv` (voir la leçon 3.1 pour le diagnostic).

## Sortie attendue

```text
{'rows_before': 308, 'duplicates_removed': 8, 'math_absent': 6, 'impossible_values': 3,
 'imputed': {'study_hours_week': 21, 'sleep_hours': 15, 'physics': 9}, 'rows_after': 300}
```

Pour les valeurs **non imputées**, le résultat doit coïncider avec le fichier propre `datasets/generated/students.csv` (à l'arrondi de l'assiduité près).

## Contraintes

- Ne pas modifier `raw`.
- Aucune boucle sur les lignes.

## Indices

- **Indice 1** : `groupby("group")["col"].transform("median")` renvoie une Series alignée sur le DataFrame, parfaite pour `fillna`.
- **Indice 2** : créez les colonnes `*_was_missing` **avant** d'imputer.
- **Indice 3** : `~between(0, 80)` est aussi vrai pour les `NaN` ; combinez avec `notna()` pour compter les seules valeurs impossibles.

## Vérification

```bash
python -m pytest exercises/intermediate/I02_missing_values_outliers -q
```

Solution : `solutions/intermediate/I02_missing_values_outliers/solution.py`.
