# I03 — Fusion de tables et agrégations

- **Identifiant** : I03
- **Niveau** : intermédiaire
- **Module** : 3 (leçons 3.3 et 3.4)
- **Durée indicative** : 1 h 30

## Objectifs d'apprentissage

- Réaliser une jointure sûre (cardinalité vérifiée, aucune ligne perdue ni dupliquée).
- Repérer les clés orphelines.
- Produire des tableaux de synthèse (`groupby`, `pivot_table`).

## Énoncé

Complétez `exercise.py` :

1. `orphan_ids(students, activities)` : liste **triée** des `student_id` présents dans `activities` mais absents de `students`.
2. `merge_activities(students, activities)` : jointure **gauche** qui vérifie la relation un-à-un (`validate`) ; les étudiants sans activité reçoivent `club = "inconnu"`. Le résultat doit avoir exactement autant de lignes que `students`.
3. `summary_by_club(merged)` : une ligne par club (index `club`), colonnes `n`, `pass_rate`, `mean_final`, `mean_study_hours`, arrondies à 3 décimales, triées par `pass_rate` décroissant (puis `n` décroissant).
4. `pivot_pass_rate(merged)` : table pivot du taux de réussite (lignes : `group`, colonnes : `club`), arrondie à 3 décimales.

## Données d'entrée

- `datasets/generated/students.csv` (300 lignes)
- `datasets/generated/student_activities.csv` (260 lignes, dont 10 identifiants inconnus)

## Sortie attendue

```text
orphan_ids -> ['S0901', 'S0902', ..., 'S0910']   (10 identifiants)
summary_by_club -> premier club : 'aucun' (n = 61, pass_rate = 0.574)
```

## Contraintes

- Aucune ligne d'étudiant ne doit être perdue ni dupliquée.
- `merge_activities` doit lever une erreur si la table des activités contient un identifiant en double.

## Indices

- **Indice 1** : opérations d'ensembles : `set(a) - set(b)`.
- **Indice 2** : `merge(..., how="left", validate="one_to_one")` lève `pandas.errors.MergeError` en cas de doublon.
- **Indice 3** : la moyenne de `passed` (0/1) est un taux.

## Vérification

```bash
python -m pytest exercises/intermediate/I03_groupby_merge -q
```

Solution : `solutions/intermediate/I03_groupby_merge/solution.py`.
