# A04 — Convolution 2D et max-pooling en NumPy

- **Identifiant** : A04
- **Niveau** : avancé
- **Module** : 9 (leçon 9.2)
- **Durée indicative** : 2 h 30

## Objectifs d'apprentissage

- Maîtriser le calcul exact d'une convolution (corrélation croisée) avec remplissage et pas.
- Écrire une version vectorisée, sans boucle Python sur les pixels.
- Gérer plusieurs canaux d'entrée et de sortie comme `nn.Conv2d`.
- Implémenter le max-pooling par remodelage de tableaux.

## Énoncé

Complétez `exercise.py` avec NumPy :

1. `output_size(n, kernel, padding=0, stride=1)` : ⌊(n + 2p − k)/s⌋ + 1 ; `ValueError` si le résultat est ≤ 0.
2. `conv2d(image, kernel, padding=0, stride=1)` : corrélation croisée d'une image 2D par un filtre 2D, avec remplissage de zéros et pas.
3. `conv2d_multi(x, weights, bias, padding=0, stride=1)` : x de forme (C_in, H, W), weights (C_out, C_in, k, k), bias (C_out,) → sortie (C_out, H_out, W_out), comme `torch.nn.functional.conv2d` sur une seule image.
4. `max_pool2d(x, size=2)` : max-pooling sans recouvrement sur les deux derniers axes (fonctionne aussi avec un axe de canaux devant) ; les lignes ou colonnes restantes (si la taille n'est pas divisible) sont ignorées, comme dans PyTorch.

## Données d'entrée

Tableaux aléatoires et images de `datasets/generated/shapes.npz`.

## Sortie attendue

Résultats identiques (à 10⁻⁵ près) à `torch.nn.functional.conv2d` et `torch.nn.functional.max_pool2d`. La convolution d'une image 28 × 28 par un filtre 3 × 3 sans remplissage donne 26 × 26 ; avec `padding=1, stride=2`, 14 × 14.

## Contraintes

- Pas de boucle Python sur les pixels dans `conv2d` et `conv2d_multi` (les tests vérifient la vitesse sur une image 256 × 256).
- NumPy uniquement dans `exercise.py`.

## Indices

- **Indice 1** : `np.pad(image, padding)` ajoute des zéros autour.
- **Indice 2** : `np.lib.stride_tricks.sliding_window_view(padded, (k, k))` crée toutes les fenêtres sans copie ; `[::stride, ::stride]` applique le pas.
- **Indice 3** : `np.einsum("ijkl,kl->ij", windows, kernel)` fait la somme des produits pour chaque fenêtre.
- **Indice 4** : pour le pooling, remodelez (H, W) en (H/2, 2, W/2, 2) puis prenez le maximum sur les axes de taille 2.

## Vérification

```bash
python -m pytest exercises/advanced/A04_convolution_2d -q
```

Solution : `solutions/advanced/A04_convolution_2d/solution.py`.
