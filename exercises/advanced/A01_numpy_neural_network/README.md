# A01 — Réseau de neurones en NumPy avec rétropropagation

- **Identifiant** : A01
- **Niveau** : avancé
- **Module** : 6 (leçon 6.2), 7 (leçon 7.1)
- **Durée indicative** : 3 h

## Objectifs d'apprentissage

- Généraliser la rétropropagation à un nombre quelconque de couches cachées.
- Structurer un modèle en classe (`forward`, `backward`, `step`, `fit`, `predict`).
- Vérifier ses gradients par différences finies.
- Résoudre le problème XOR, impossible pour un neurone seul.

## Énoncé

Complétez `exercise.py` :

1. `sigmoid(z)` et `binary_cross_entropy(p, y)` (écrêter p dans [1e-12, 1 − 1e-12]).
2. La classe `MLP(layer_sizes, seed=0)` pour la classification binaire :
   - `layer_sizes` du type `[n_entrées, h1, ..., 1]` ; `ValueError` sinon (moins de 2 tailles ou dernière taille ≠ 1) ;
   - attributs `weights` (liste de matrices (n_in, n_out), initialisation de He avec `np.random.default_rng(seed)`) et `biases` (vecteurs nuls) ;
   - `forward(X)` : ReLU pour les couches cachées, sigmoïde en sortie ; renvoie P de forme (n, 1) ;
   - `loss(X, y)` : entropie croisée binaire (y de forme (n,) ou (n, 1)) ;
   - `backward(X, y)` → `(grads_W, grads_b)`, listes de mêmes formes que `weights` et `biases` ;
   - `step(grads_W, grads_b, learning_rate)` ;
   - `fit(X, y, epochs=1000, learning_rate=0.5)` → historique des pertes (perte **avant** chaque mise à jour) ;
   - `predict_proba(X)` (forme (n,)) et `predict(X)` (0/1).
3. `numerical_gradient(model, X, y, layer, index, h=1e-6)` : différence centrée de la perte par rapport à `model.weights[layer][index]` (le poids doit retrouver sa valeur d'origine).

## Données d'entrée

- XOR : X = [[0,0],[0,1],[1,0],[1,1]], y = [0,1,1,0].
- `make_moons(n_samples=400, noise=0.2, random_state=0)` (généré localement par scikit-learn).

## Sortie attendue

- Erreur relative entre gradients analytiques et numériques < 10⁻⁵ (réseau 3 → 5 → 4 → 1).
- XOR résolu (4/4) avec `MLP([2, 8, 1], seed=0)`, 3000 époques, `learning_rate=0.5`.
- Précision d'entraînement > 0,9 sur les croissants.

## Contraintes

- NumPy uniquement ; aucune boucle sur les exemples (seulement sur les couches).

## Indices

- **Indice 1** : mémorisez dans `forward` les entrées de chaque couche et les pré-activations z ; `backward` en a besoin.
- **Indice 2** : en sortie, δ = (P − y) / n ; puis pour la couche i : dW = Aᵢᵀ δ, db = Σ δ, et δ ← (δ Wᵢᵀ) ⊙ 1[zᵢ₋₁ > 0].
- **Indice 3** : écrivez d'abord le test du gradient, puis la rétropropagation.

## Vérification

```bash
python -m pytest exercises/advanced/A01_numpy_neural_network -q
```

Solution : `solutions/advanced/A01_numpy_neural_network/solution.py`.
