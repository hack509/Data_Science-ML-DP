# I08 — Métriques de classification codées à la main

- **Identifiant** : I08
- **Niveau** : intermédiaire
- **Module** : 5 (leçon 5.3)
- **Durée indicative** : 1 h 30

## Objectifs d'apprentissage

- Maîtriser les définitions de la matrice de confusion, de l'accuracy, de la précision, du rappel et du F1.
- Gérer les cas limites (aucune prédiction positive : division par zéro).
- Comprendre l'AUC comme probabilité de bon classement d'une paire positif/négatif.

## Énoncé

Complétez `exercise.py` **sans scikit-learn ni NumPy** :

1. `confusion_counts(y_true, y_pred, positive=1)` → `{"tp", "fp", "fn", "tn"}` ; `ValueError` si les longueurs diffèrent.
2. `accuracy(y_true, y_pred)` ; `ValueError` si l'entrée est vide.
3. `precision`, `recall`, `f1` (paramètre `positive=1`) : renvoient `0.0` quand le dénominateur est nul.
4. `confusion_matrix(y_true, y_pred, labels)` : liste de listes, lignes = vraies classes, colonnes = classes prédites, dans l'ordre de `labels`.
5. `roc_auc(y_true, scores, positive=1)` : proportion des paires (positif, négatif) où le positif a le score le plus élevé, les égalités comptant pour 1/2. `ValueError` s'il manque une des deux classes.

## Données d'entrée

Petites listes dans les tests, puis les prédictions d'une régression logistique sur `binary_classification.csv`, comparées à scikit-learn.

## Sortie attendue

```text
>>> confusion_counts([1, 0, 1, 1, 0], [1, 1, 0, 1, 0])
{'tp': 2, 'fp': 1, 'fn': 1, 'tn': 1}
>>> roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
0.75
```

## Contraintes

- Python standard uniquement dans `exercise.py`.
- `roc_auc` doit rester rapide pour 10 000 exemples (évitez la double boucle sur toutes les paires si possible).

## Indices

- **Indice 1** : une seule boucle sur `zip(y_true, y_pred)` suffit pour compter VP, FP, FN, VN.
- **Indice 2** : l'AUC se calcule avec les rangs : triez les scores, attribuez à chaque score son rang (rang moyen en cas d'égalité), additionnez les rangs des positifs R, puis AUC = (R − n₊(n₊ + 1)/2) / (n₊ · n₋).
- **Indice 3** : commencez par une version simple en double boucle, vérifiez-la, puis optimisez.

## Vérification

```bash
python -m pytest exercises/intermediate/I08_classification_metrics -q
```

Solution : `solutions/intermediate/I08_classification_metrics/solution.py`.
