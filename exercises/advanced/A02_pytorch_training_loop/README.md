# A02 — Boucle d'entraînement PyTorch réutilisable

- **Identifiant** : A02
- **Niveau** : avancé
- **Module** : 6 (leçons 6.3 et 6.4)
- **Durée indicative** : 2 h 30

## Objectifs d'apprentissage

- Écrire des fonctions d'entraînement et d'évaluation génériques, utilisables pour n'importe quel modèle de classification.
- Préparer les données sans fuite (standardisation avec les seules statistiques d'entraînement).
- Implémenter l'arrêt précoce avec restauration des meilleurs poids.
- Sauvegarder et recharger des poids en toute sécurité.

## Énoncé

Complétez `exercise.py` :

1. `make_loaders(X, y, batch_size=32, val_fraction=0.2, seed=0)` : mélange les indices avec `np.random.default_rng(seed).permutation`, prend les `round(n × val_fraction)` premiers pour la validation, standardise avec la moyenne et l'écart-type **de l'entraînement** (remplacer un écart-type nul par 1), convertit X en `float32` et y en `int64`. Renvoie `(train_loader, val_loader, (mean, std))` ; seul le chargeur d'entraînement mélange les données.
2. `train_one_epoch(model, loader, loss_fn, optimizer)` → perte moyenne pondérée par la taille des lots.
3. `evaluate(model, loader, loss_fn)` → `(perte moyenne, accuracy)`, en mode évaluation et sans gradient.
4. `fit(model, train_loader, val_loader, loss_fn, optimizer, epochs=100, patience=10)` → dictionnaire `train_loss`, `val_loss`, `val_accuracy` (listes, une valeur par époque effectuée) et `best_epoch` (indice de l'époque de meilleure perte de validation). Arrêt quand `patience` époques se sont écoulées depuis la meilleure ; à la fin, le modèle contient les **meilleurs** poids.
5. `save_checkpoint(model, path)` (crée les dossiers, renvoie le chemin) et `load_checkpoint(model, path)` (charge avec `weights_only=True`, passe en mode évaluation, renvoie le modèle).

## Données d'entrée

`datasets/generated/multiclass_classification.csv` (900 exemples, 5 variables, 3 classes).

## Sortie attendue

Un petit réseau 5 → 32 → 3 entraîné avec Adam (`lr=1e-2`) atteint une accuracy de validation supérieure à 0,85 en quelques secondes sur CPU ; après rechargement, les prédictions sont identiques.

## Contraintes

- Aucune fuite : les statistiques de standardisation ne voient pas la validation.
- `fit` doit fonctionner avec `patience=None` (pas d'arrêt précoce).
- Copiez l'état du modèle avec `copy.deepcopy(model.state_dict())`.

## Indices

- **Indice 1** : `torch.Generator().manual_seed(seed)` passé au `DataLoader` rend le mélange reproductible.
- **Indice 2** : `(logits.argmax(dim=1) == yb).sum().item()` compte les bonnes réponses d'un lot.
- **Indice 3** : réutilisez la fonction `train` de l'exemple 2 de la leçon 6.4.

## Vérification

```bash
python -m pytest exercises/advanced/A02_pytorch_training_loop -q
```

Solution : `solutions/advanced/A02_pytorch_training_loop/solution.py`.
