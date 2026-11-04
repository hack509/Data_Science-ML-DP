# Exercices

21 exercices en trois niveaux. Chaque dossier contient :

| Fichier | Rôle |
|---|---|
| `README.md` | identifiant, objectifs, énoncé, données, sortie attendue, contraintes, indices |
| `exercise.py` | fichier à compléter (les fonctions lèvent `NotImplementedError`) |
| `test_exercise.py` | tests automatiques |

## Méthode

1. Lisez l'énoncé, puis complétez `exercise.py`.
2. Lancez les tests depuis la racine du dépôt :

```bash
python -m pytest exercises/beginner/B01_student_averages -q
```

3. Tous les tests passent ? Comparez avec la solution (`solutions/<niveau>/<identifiant>/solution.py`), puis marquez l'exercice comme terminé : `python scripts/progress.py exercise B01`.

Les tests échouent tant que l'exercice n'est pas fait : c'est normal. Les solutions sont volontairement séparées : essayez d'abord, utilisez les indices ensuite, la solution en dernier.

## Liste

| Niveau | Exercices |
|---|---|
| Débutant | B01 moyennes, B02 ventes, B03 nettoyage CSV, B04 JSON, B05 statistiques, B06 matrices |
| Intermédiaire | I01 script réutilisable, I02 valeurs manquantes, I03 fusion, I04 descente de gradient, I05 test d'hypothèse, I06 rapport visuel, I07 pipeline de régression, I08 métriques, I09 k-means |
| Avancé | A01 réseau NumPy, A02 boucle PyTorch, A03 TF-IDF, A04 convolution, A05 validation croisée, A06 attention |

A02, A04 et A06 nécessitent PyTorch.
