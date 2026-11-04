# B01 — Moyennes d'étudiants

- **Identifiant** : B01
- **Niveau** : débutant
- **Module** : 1 (leçon 1.1 et 1.2)
- **Durée indicative** : 45 min

## Objectifs d'apprentissage

- Manipuler des listes et des dictionnaires.
- Écrire des fonctions qui valident leurs entrées et lèvent des exceptions explicites.
- Calculer des moyennes simples et pondérées.

## Énoncé

Complétez les fonctions du fichier `exercise.py` :

1. `average(scores)` : moyenne d'une liste de notes. Lève `ValueError` si la liste est vide.
2. `weighted_average(scores, weights)` : moyenne pondérée. `scores` et `weights` sont des dictionnaires `{matière: valeur}`. Lève `KeyError` si une matière de `scores` n'a pas de coefficient, `ValueError` si la somme des coefficients utilisés n'est pas strictement positive.
3. `student_averages(records, weights=None)` : `records` est une liste de dictionnaires contenant au moins `student_id`, `math`, `physics`, `french` (les notes peuvent être des chaînes, comme dans un CSV). Renvoie `{student_id: moyenne arrondie à 2 décimales}`. Sans `weights`, toutes les matières ont le même poids.
4. `mention(avg)` : renvoie `"ajourné"` (< 10), `"passable"` (< 12), `"assez bien"` (< 14), `"bien"` (< 16) ou `"très bien"`. Lève `ValueError` si `avg` n'est pas dans [0, 20].
5. `class_report(records, weights=None)` : renvoie un dictionnaire avec les clés `averages`, `best_student`, `pass_rate` (proportion de moyennes ≥ 10, arrondie à 3 décimales) et `mentions`.

## Données d'entrée

- Exemples en ligne dans les tests.
- `datasets/generated/students.csv` (300 étudiants). Avec les coefficients `{"math": 4, "physics": 3, "french": 3}`, votre moyenne doit être égale à la colonne `final_score` (à 0,01 près).

## Sortie attendue

```text
>>> average([12, 14, 16])
14.0
>>> weighted_average({"math": 10, "french": 16}, {"math": 2, "french": 1})
12.0
>>> mention(13.99)
'assez bien'
```

Sur `students.csv` avec les coefficients 4/3/3 : meilleur étudiant `S0081`, taux de réussite `0.44`.

## Contraintes

- N'utilisez ni NumPy ni pandas : uniquement Python standard (le module `csv` est autorisé).
- Ne modifiez pas les noms ni les signatures des fonctions.

## Indices

- **Indice 1** : Pour la moyenne pondérée : accumulez `weights[m] * score` et `weights[m]` dans deux variables.
- **Indice 2** : `max(d, key=d.get)` renvoie la clé de plus grande valeur.
- **Indice 3** : Convertissez les notes avec `float()` : dans un CSV, tout est texte.

## Vérification

```bash
python -m pytest exercises/beginner/B01_student_averages -q
```

La solution de référence se trouve dans `solutions/beginner/B01_student_averages/solution.py` : ne l'ouvrez qu'après avoir essayé.
