# A06 — Attention par produit scalaire et attention multi-têtes en NumPy

- **Identifiant** : A06
- **Niveau** : avancé
- **Module** : 8 (leçon 8.4)
- **Durée indicative** : 2 h

## Objectifs d'apprentissage

- Implémenter le cœur des Transformers : softmax stable, attention mise à l'échelle, masque causal.
- Découper les dimensions en têtes et recombiner leurs sorties.
- Vérifier une implémentation contre la référence PyTorch.

## Énoncé

Complétez `exercise.py` avec NumPy :

1. `softmax(x, axis=-1)` numériquement stable (soustraire le maximum) : fonctionne même pour des valeurs de l'ordre de 1000.
2. `causal_mask(n)` : matrice booléenne (n × n), `True` là où l'attention est autorisée (j ≤ i).
3. `scaled_dot_product_attention(Q, K, V, mask=None)` → `(sortie, poids)` avec poids = softmax(QKᵀ / √d_k) ; là où le masque vaut `False`, le poids doit être **exactement** 0.
4. `multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads, mask=None)` : auto-attention ; Q = XW_q, K = XW_k, V = XW_v ; la tête h utilise les colonnes `[h·d_head, (h+1)·d_head[` ; les sorties des têtes sont concaténées puis multipliées par W_o. `ValueError` si d_model n'est pas divisible par `n_heads`.

## Données d'entrée

Matrices aléatoires générées dans les tests.

## Sortie attendue

- Résultats identiques (à 10⁻⁶ près) à `torch.nn.functional.scaled_dot_product_attention` et à `torch.nn.MultiheadAttention(..., bias=False)` avec les mêmes poids.
- Chaque ligne de poids somme à 1.

## Contraintes

- NumPy uniquement dans `exercise.py` (les tests utilisent PyTorch comme référence).
- Convention : les poids multiplient à droite (`X @ W`), alors que PyTorch stocke des matrices transposées (`x @ W.T`). Les tests font la conversion.

## Indices

- **Indice 1** : remplacer les scores masqués par `-np.inf` donne un poids exactement nul après le softmax.
- **Indice 2** : `np.tril(np.ones((n, n), dtype=bool))`.
- **Indice 3** : chaque tête travaille sur `d_head = d_model // n_heads` colonnes ; la mise à l'échelle utilise `√d_head`.

## Vérification

```bash
python -m pytest exercises/advanced/A06_attention -q
```

Solution : `solutions/advanced/A06_attention/solution.py`.
