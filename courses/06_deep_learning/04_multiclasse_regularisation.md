# Leçon 6.4 — Classification multiclasse et régularisation

## Prérequis

- Leçon 6.3 (boucle d'entraînement PyTorch).

## Objectifs d'apprentissage

- Passer de la classification binaire à la classification multiclasse : softmax et `nn.CrossEntropyLoss`.
- Reconnaître le surapprentissage sur les courbes d'apprentissage.
- Utiliser la pénalité sur les poids (*weight decay*), le **dropout** et la **normalisation par lots** (*batch normalization*).
- Mettre en place l'arrêt précoce (*early stopping*) avec sauvegarde des meilleurs poids.

## Concepts clés

- **Softmax** : transforme un vecteur de K scores (logits) en K probabilités positives de somme 1.
- **Entropie croisée multiclasse** (*cross-entropy*) : −log(probabilité attribuée à la bonne classe).
- **Régularisation** : toute technique qui réduit le surapprentissage.
- **Pénalité L2 / weight decay** : pousse les poids vers 0.
- **Dropout** : met aléatoirement à zéro une fraction des activations pendant l'entraînement.
- **Normalisation par lots** (*BatchNorm*) : standardise les activations d'une couche sur chaque mini-lot, puis les réajuste avec deux paramètres appris.
- **Arrêt précoce** (*early stopping*) : arrêter quand la perte de validation ne s'améliore plus et garder les meilleurs poids.

## Explication détaillée

### Softmax et entropie croisée

Pour K classes, la dernière couche produit K logits z₁, ..., z_K. Le softmax les transforme en probabilités : pₖ = exp(zₖ) / Σⱼ exp(zⱼ). La perte est −log(p_bonne_classe) : faible si le modèle donne une forte probabilité à la bonne classe, très forte s'il est confiant et faux. En PyTorch, `nn.CrossEntropyLoss` prend directement les **logits** (elle applique le softmax de façon stable) et des étiquettes **entières** (`int64`) de forme (n,). La classe prédite est `logits.argmax(dim=1)`.

### Reconnaître le surapprentissage

On trace la perte d'entraînement et la perte de validation à chaque époque. Au début, les deux baissent. Si ensuite la perte d'entraînement continue de baisser alors que la perte de validation **remonte**, le réseau mémorise l'entraînement. Plus le réseau est grand et les données petites, plus ce phénomène arrive tôt. Pour l'observer clairement, cette leçon entraîne volontairement un grand réseau sur seulement 120 exemples.

### Les outils de régularisation

- **Weight decay** (paramètre `weight_decay` de l'optimiseur, de préférence avec `AdamW`) : ajoute λ‖w‖² à la perte, ce qui garde les poids petits et la fonction plus lisse.
- **Dropout** (`nn.Dropout(p=0.3)`) : à chaque mini-lot, 30 % des activations sont mises à zéro au hasard (et les autres multipliées par 1/0,7). Le réseau ne peut pas compter sur un neurone particulier ; c'est un peu comme entraîner un ensemble de sous-réseaux. **En mode évaluation**, le dropout est désactivé : d'où l'importance de `model.eval()`.
- **Normalisation par lots** (`nn.BatchNorm1d`) : stabilise et accélère l'entraînement, permet des taux d'apprentissage plus élevés, avec un léger effet régularisant. En entraînement elle utilise les statistiques du lot ; en évaluation, des moyennes mobiles apprises (encore `model.eval()`).
- **Arrêt précoce** : on mémorise les poids de la meilleure époque de validation et on arrête après `patience` époques sans amélioration.
- **Plus de données** et **augmentation de données** (module 9) restent les remèdes les plus puissants.

## Intuition mathématique

```text
softmax(z)ₖ = exp(zₖ − max z) / Σⱼ exp(zⱼ − max z)    (soustraire max z évite le débordement)
CE = −(1/n) Σᵢ log softmax(zᵢ)_{yᵢ}
weight decay : L' = L + λ ‖w‖²   ⇒   w ← w − η(∇L + 2λw)
dropout (entraînement) : h' = h ⊙ m / (1 − p),  m ~ Bernoulli(1 − p)
```

## Applications concrètes

- Reconnaître des chiffres, des objets, des catégories de documents (multiclasse).
- Tout entraînement réel de réseau : la régularisation et l'arrêt précoce sont systématiques.

## Exemples de code Python

### Exemple 1 — softmax et entropie croisée à la main

```python
import torch
from torch import nn

logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 3.0]])
targets = torch.tensor([0, 1])
probabilities = torch.softmax(logits, dim=1)
print(probabilities.round(decimals=3), "| row sums:", probabilities.sum(dim=1))
manual = -torch.log(probabilities[torch.arange(2), targets]).mean()
print("manual cross-entropy:", round(manual.item(), 4), "| nn.CrossEntropyLoss:", round(nn.CrossEntropyLoss()(logits, targets).item(), 4))
print("predicted classes:", logits.argmax(dim=1).tolist())
```

### Explication du code

Le premier exemple est bien classé (classe 0, probabilité élevée) ; le second est mal classé (le modèle mise sur la classe 2 alors que la vraie est 1) et contribue fortement à la perte. Le calcul manuel donne la même valeur que `nn.CrossEntropyLoss`, qui reçoit les logits bruts.

### Sortie attendue

```text output
tensor([[0.7860, 0.1750, 0.0390],
        [0.0490, 0.0540, 0.8960]]) | row sums: tensor([1., 1.])
manual cross-entropy: 1.5755 | nn.CrossEntropyLoss: 1.5755
predicted classes: [0, 2]
```

### Exemple 2 — provoquer le surapprentissage, puis le réduire

```python
import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv("datasets/generated/multiclass_classification.csv")
X = df.drop(columns="target").to_numpy(dtype=np.float32)
y = df["target"].to_numpy(dtype=np.int64)
X_small, X_rest, y_small, y_rest = train_test_split(X, y, train_size=120, stratify=y, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=0)
mean, std = X_small.mean(axis=0), X_small.std(axis=0)
to_tensor = lambda a: torch.from_numpy((a - mean) / std)
X_tr_t, X_val_t, X_te_t = to_tensor(X_small), to_tensor(X_val), to_tensor(X_test)
y_tr_t, y_val_t, y_te_t = map(torch.from_numpy, (y_small, y_val, y_test))
loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=16, shuffle=True)


def make_model(dropout=0.0, batchnorm=False):
    layers = []
    for n_in, n_out in [(5, 256), (256, 256)]:
        layers.append(nn.Linear(n_in, n_out))
        if batchnorm:
            layers.append(nn.BatchNorm1d(n_out))
        layers += [nn.ReLU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(256, 3))
    return nn.Sequential(*layers)


def train(model, epochs=150, weight_decay=0.0, patience=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history, best = [], (float("inf"), None, 0)
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(X_tr_t), y_tr_t).item()
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        history.append((train_loss, val_loss))
        if val_loss < best[0]:
            best = (val_loss, copy.deepcopy(model.state_dict()), epoch)
        elif patience is not None and epoch - best[2] >= patience:
            break
    if patience is not None:
        model.load_state_dict(best[1])  # restore the best weights
    return history, best[2]


@torch.no_grad()
def accuracy(model, X_t, y_t):
    model.eval()
    return (model(X_t).argmax(dim=1) == y_t).float().mean().item()


results = {}
for name, kwargs, train_kwargs in [
    ("no regularization", {}, {}),
    ("dropout 0.5 + weight decay", {"dropout": 0.5}, {"weight_decay": 0.05}),
    ("early stopping", {}, {"patience": 15}),
    ("batchnorm + dropout + early stop", {"dropout": 0.3, "batchnorm": True}, {"patience": 15}),
]:
    torch.manual_seed(0)
    model = make_model(**kwargs)
    history, best_epoch = train(model, **train_kwargs)
    results[name] = history
    min_val = min(v for _, v in history)
    print(f"{name:<34} epochs={len(history):>3}  final train loss={history[-1][0]:.3f}  "
          f"min val loss={min_val:.3f}  final val loss={history[-1][1]:.3f}  test acc={accuracy(model, X_te_t, y_te_t):.3f}")
```

### Explication du code

- On n'utilise que 120 exemples d'entraînement et un réseau d'environ 68 000 paramètres (5×256 + 256 + 256×256 + 256 + 256×3 + 3 = 68 099) : idéal pour voir le surapprentissage.
- Sans régularisation, la perte d'entraînement tombe presque à zéro tandis que la perte de validation finale est bien plus haute que son minimum : le réseau a mémorisé l'entraînement.
- Avec dropout et weight decay, l'écart entre entraînement et validation se réduit.
- Avec l'arrêt précoce, l'entraînement s'arrête une quinzaine d'époques après le meilleur score de validation et les meilleurs poids sont restaurés (`copy.deepcopy` du `state_dict`, sinon on garderait une référence aux poids qui continuent de changer).
- Pour les lignes avec arrêt précoce, « final val loss » est la perte à l'époque d'arrêt ; le modèle évalué ensuite sur le test est bien celui de la **meilleure** époque, restauré.
- Les précisions de test sont calculées sur un jeu jamais utilisé pour décider quoi que ce soit.

### Sortie attendue

```text output
no regularization                  epochs=150  final train loss=0.001  min val loss=0.295  final val loss=0.804  test acc=0.895
dropout 0.5 + weight decay         epochs=150  final train loss=0.038  min val loss=0.296  final val loss=0.512  test acc=0.915
early stopping                     epochs= 24  final train loss=0.085  min val loss=0.295  final val loss=0.363  test acc=0.923
batchnorm + dropout + early stop   epochs= 35  final train loss=0.136  min val loss=0.274  final val loss=0.305  test acc=0.926
```

### Exemple 3 — tracer les courbes d'apprentissage

```python
from pathlib import Path

import matplotlib.pyplot as plt

from lab.plotting import CATEGORICAL, apply_style

apply_style()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, name in zip(axes, ["no regularization", "dropout 0.5 + weight decay"]):
    history = np.array(results[name])
    ax.plot(history[:, 0], color=CATEGORICAL[0], label="entraînement")
    ax.plot(history[:, 1], color=CATEGORICAL[1], label="validation")
    ax.axvline(history[:, 1].argmin(), color="#8a8a85", linestyle="--", linewidth=1)
    ax.set_title(name)
    ax.set_xlabel("époque")
axes[0].set_ylabel("entropie croisée")
axes[0].legend()
Path("outputs/module06").mkdir(parents=True, exist_ok=True)
fig.savefig("outputs/module06/learning_curves_regularization.png", bbox_inches="tight")
plt.close(fig)
print("best validation epoch without regularization:", int(np.array(results["no regularization"])[:, 1].argmin()))
```

### Explication du code

La figure `outputs/module06/learning_curves_regularization.png` montre, à gauche, la courbe de validation qui remonte après son minimum (ligne pointillée) pendant que la courbe d'entraînement continue de descendre : c'est la signature du surapprentissage. À droite, la régularisation maintient les deux courbes plus proches.

### Sortie attendue

```text output
best validation epoch without regularization: 8
```

## Erreurs fréquentes

- **Appliquer softmax avant `CrossEntropyLoss`** : la perte attend des logits.
- **Étiquettes en `float` ou en one-hot** pour `CrossEntropyLoss` : elle attend des entiers `int64` de forme (n,).
- **Oublier `model.eval()`** : dropout et BatchNorm restent en mode entraînement et les résultats varient d'un appel à l'autre.
- **BatchNorm avec des lots de taille 1** : erreur ou statistiques absurdes.
- **Arrêt précoce sans restaurer les meilleurs poids**, ou sans `deepcopy`.
- **Régler la régularisation sur le jeu de test**.

## Exercices guidés

1. Calculez à la main le softmax de (1, 2, 3) et vérifiez avec `torch.softmax`.
2. Entraînez le modèle sans régularisation sur **tout** le jeu d'entraînement disponible (X_rest moins validation et test, ou un découpage 70/15/15). Le surapprentissage est-il aussi marqué ?
3. Faites varier le taux de dropout (0 ; 0,2 ; 0,5 ; 0,8). Que se passe-t-il pour 0,8 ?
4. Montrez qu'un modèle avec dropout donne des sorties différentes à deux appels en mode `train()` et identiques en mode `eval()`.

Corrigés : [corrections.md](corrections.md), section « Leçon 6.4 ».

## Défis autonomes

- Ajoutez un planificateur de taux d'apprentissage (`torch.optim.lr_scheduler.ReduceLROnPlateau`) et observez son effet.
- Écrivez une classe `EarlyStopping` réutilisable (c'est une partie de l'exercice A02).

## Quiz

1. Que garantit la fonction softmax ?
2. Quel type et quelle forme d'étiquettes attend `nn.CrossEntropyLoss` ?
3. Comment reconnaître le surapprentissage sur des courbes d'apprentissage ?
4. Pourquoi le dropout doit-il être désactivé à l'évaluation ?
5. Que fait la normalisation par lots ?
6. Pourquoi restaurer les meilleurs poids en fin d'arrêt précoce ?

Réponses : [corrections.md](corrections.md), section « Leçon 6.4 ».

## Résumé

- Multiclasse : K logits, `CrossEntropyLoss` (softmax intégré), étiquettes entières, prédiction par `argmax`.
- Surapprentissage visible quand la validation remonte alors que l'entraînement baisse.
- Régularisation : weight decay, dropout, BatchNorm, arrêt précoce avec meilleurs poids, plus de données.

## Pour aller plus loin

- Exercice **A02** (`exercises/advanced/A02_pytorch_training_loop`).
- Module 7 : architectures, initialisation et expérimentations.
