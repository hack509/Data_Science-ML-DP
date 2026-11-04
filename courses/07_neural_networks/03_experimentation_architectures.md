# Leçon 7.3 — Expérimenter : comparer des architectures et des hyperparamètres

## Prérequis

- Leçons 7.1 et 7.2 ; leçon 5.6 (validation, choix d'hyperparamètres) ; leçon 2.4 (variabilité, incertitude).

## Objectifs d'apprentissage

- Concevoir une expérience comparative équitable : mêmes données, même budget, plusieurs graines.
- Mesurer performance, nombre de paramètres et temps d'entraînement.
- Distinguer une vraie différence du bruit dû à l'aléa de l'initialisation et des lots.
- Explorer un espace d'hyperparamètres de façon méthodique et consigner les résultats.
- Rédiger une conclusion prudente et reproductible.

## Concepts clés

- **Architecture** : structure du réseau (nombre de couches, largeur, activations, régularisation).
- **Budget** : nombre d'époques, de mises à jour ou temps de calcul alloué à chaque variante.
- **Graine aléatoire** (*random seed*) : fixe l'initialisation et l'ordre des lots ; la faire varier mesure la variabilité.
- **Ablation** (*ablation study*) : retirer un élément à la fois pour mesurer sa contribution.
- **Journal d'expériences** (*experiment log*) : table où chaque essai est consigné avec ses réglages et résultats.

## Explication détaillée

### Une expérience équitable

Comparer un réseau entraîné 200 époques à un autre entraîné 20 époques ne dit rien des architectures. Les règles :

1. **Mêmes données** et même découpage entraînement / validation / test.
2. **Même budget** (nombre d'époques ou de mises à jour) et même procédure de sélection (par exemple, meilleure époque de validation).
3. **Plusieurs graines** (au moins 3, idéalement 5 ou plus) pour chaque variante ; on rapporte la **moyenne** et l'**écart-type**.
4. **Sélection sur la validation**, rapport final sur le test une seule fois.
5. **Consigner tout** : graine, hyperparamètres, nombre de paramètres, temps, métriques, version du code.

### Le bruit des graines

Deux entraînements identiques avec des graines différentes ne donnent pas le même score : l'initialisation et l'ordre des lots changent la trajectoire de l'optimisation. Sur un petit jeu de validation (quelques centaines d'exemples), un écart d'1 point de précision est souvent du bruit. Règle pratique : si l'écart entre deux moyennes est inférieur à l'écart-type entre graines, ne concluez pas.

### Profondeur, largeur et taille

- **Plus large** (plus de neurones par couche) : plus de capacité, risque de surapprentissage sur de petites données.
- **Plus profond** : représentations plus riches, mais optimisation plus délicate (leçon 7.2) ; sur des données tabulaires simples, 1 à 3 couches cachées suffisent généralement.
- **Plus de paramètres** ne signifie pas meilleur : au-delà d'une certaine capacité, la performance de validation plafonne alors que le temps et la mémoire augmentent.
- Comparez toujours à une **référence** simple (régression logistique, forêt aléatoire) : un réseau qui ne la bat pas n'est pas justifié sur ce problème.

### Explorer les hyperparamètres

Ordre de priorité habituel : taux d'apprentissage > taille et profondeur > régularisation (dropout, weight decay) > taille de lot. Méthodes : grille grossière puis fine ; recherche aléatoire (souvent plus efficace qu'une grille quand peu d'hyperparamètres comptent vraiment) ; échelle **logarithmique** pour les taux et les pénalités (10⁻⁴, 10⁻³, 10⁻²...).

## Intuition mathématique

```text
moyenne sur s graines :      m = (1/s) Σ scoreᵢ
écart-type :                 sd = √(Σ (scoreᵢ − m)² / (s − 1))
erreur standard de m :       sd / √s
nombre de paramètres d'un MLP [d₀, d₁, ..., d_L] = Σₗ (d_{l−1} · d_l + d_l)
```

## Applications concrètes

- Choisir une architecture à mettre en production en justifiant le choix.
- Rédiger la section « expériences » d'un rapport ou d'un projet (projet 5 et 6).
- Éviter de publier ou déployer une « amélioration » qui n'est que du bruit.

## Exemples de code Python

### Exemple 1 — comparer 5 architectures avec 3 graines

```python
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch import nn

df = pd.read_csv("datasets/generated/multiclass_classification.csv")
X = df.drop(columns="target").to_numpy(dtype=np.float32)
y = df["target"].to_numpy(dtype=np.int64)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=0)
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
Xtr, Xva, Xte = (torch.from_numpy((a - mean) / std) for a in (X_train, X_val, X_test))
ytr, yva, yte = (torch.from_numpy(a) for a in (y_train, y_val, y_test))

baseline = LogisticRegression(max_iter=1000).fit((X_train - mean) / std, y_train)
print(f"baseline logistic regression: val accuracy={baseline.score((X_val - mean) / std, y_val):.3f}")


def build(hidden_sizes, dropout=0.0):
    layers, n_in = [], 5
    for h in hidden_sizes:
        layers += [nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(dropout)]
        n_in = h
    layers.append(nn.Linear(n_in, 3))
    return nn.Sequential(*layers)


def run(hidden_sizes, seed, lr=1e-2, epochs=150, batch_size=64, dropout=0.0):
    """Train with mini-batches, keep the epoch with the best validation accuracy."""
    torch.manual_seed(seed)
    model = build(hidden_sizes, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_val, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(Xtr))
        for start in range(0, len(Xtr), batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_acc = (model(Xva).argmax(1) == yva).float().mean().item()
        if val_acc > best_val:
            best_val, best_state = val_acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_val


architectures = {"linear (no hidden)": [], "1 x 8": [8], "1 x 64": [64], "2 x 64": [64, 64], "3 x 128": [128, 128, 128]}
rows = []
for name, hidden in architectures.items():
    start = time.perf_counter()
    scores = [run(hidden, seed)[1] for seed in (0, 1, 2)]
    seconds = (time.perf_counter() - start) / 3
    n_params = sum(p.numel() for p in build(hidden).parameters())
    rows.append({"architecture": name, "params": n_params, "val_mean": np.mean(scores), "val_std": np.std(scores, ddof=1), "sec_per_run": seconds})
results = pd.DataFrame(rows).set_index("architecture")
print(results.drop(columns="sec_per_run").round(3).to_string())
print("largest model is the slowest:", results["sec_per_run"].idxmax() == "3 x 128")
```

### Explication du code

- Données découpées 60 / 20 / 20 ; standardisation avec les statistiques d'entraînement.
- La référence (régression logistique) donne le niveau à battre.
- Chaque architecture est entraînée 3 fois (graines 0, 1, 2) avec le même budget (150 époques) et la même règle de sélection (meilleure époque de validation).
- Résultat instructif : **toutes les architectures se valent** (environ 95 à 96 % de précision de validation), y compris le réseau sans couche cachée, qui n'est autre qu'une régression logistique entraînée par PyTorch. Le jeu de validation compte 180 exemples : un exemple de plus ou de moins correspond à 0,56 point, si bien que les écarts du tableau représentent un ou deux exemples. Ce ne sont pas des différences démontrées.
- Ce problème est presque **linéairement séparable** (`make_classification` avec des classes bien séparées) : ajouter des couches et des milliers de paramètres n'apporte rien de mesurable, et coûte du temps (le plus gros modèle est le plus lent). Conclusion prudente : un modèle simple suffit ici. Sur un problème non linéaire (les croissants de la leçon 6.2), le tableau serait très différent : toujours mesurer plutôt que supposer.
- Attention : choisir la meilleure époque sur la validation rend le score de validation légèrement optimiste ; c'est pourquoi le test final (exemple 3) reste indispensable.

### Sortie attendue

```text output
baseline logistic regression: val accuracy=0.944
                    params  val_mean  val_std
architecture                                 
linear (no hidden)      18     0.956    0.000
1 x 8                   75     0.956    0.000
1 x 64                 579     0.954    0.003
2 x 64                4739     0.952    0.003
3 x 128              34179     0.956    0.000
largest model is the slowest: True
```

### Exemple 2 — un petit balayage d'hyperparamètres consigné dans un journal

```python
from pathlib import Path

log = []
for lr in [1e-3, 1e-2, 1e-1]:
    for dropout in [0.0, 0.3]:
        scores = [run([64], seed, lr=lr, dropout=dropout, epochs=80)[1] for seed in (0, 1, 2)]
        log.append({"hidden": "1 x 64", "lr": lr, "dropout": dropout, "epochs": 80, "seeds": "0,1,2",
                    "val_mean": round(float(np.mean(scores)), 4), "val_std": round(float(np.std(scores, ddof=1)), 4)})
journal = pd.DataFrame(log).sort_values("val_mean", ascending=False)
Path("outputs/module07").mkdir(parents=True, exist_ok=True)
journal.to_csv("outputs/module07/experiment_log.csv", index=False)
print(journal.to_string(index=False))
```

### Explication du code

Chaque combinaison (taux d'apprentissage × dropout) est évaluée sur 3 graines et **consignée** dans `outputs/module07/experiment_log.csv`, avec tous les réglages nécessaires pour la reproduire. Le taux d'apprentissage le plus faible (10⁻³) est légèrement en retrait, signe qu'il converge plus lentement en 80 époques ; le dropout ne change presque rien sur ce problème qui ne surapprend pas. Toutes les lignes sont à moins d'un point les unes des autres, soit deux exemples de validation : plusieurs réglages sont équivalents, et l'on choisit le plus simple.

### Sortie attendue

```text output
hidden    lr  dropout  epochs seeds  val_mean  val_std
1 x 64 0.100      0.0      80 0,1,2    0.9593   0.0032
1 x 64 0.100      0.3      80 0,1,2    0.9593   0.0032
1 x 64 0.010      0.3      80 0,1,2    0.9556   0.0000
1 x 64 0.010      0.0      80 0,1,2    0.9537   0.0032
1 x 64 0.001      0.3      80 0,1,2    0.9519   0.0032
1 x 64 0.001      0.0      80 0,1,2    0.9500   0.0056
```

### Exemple 3 — conclusion et évaluation finale sur le test

```python
torch.manual_seed(0)
final_scores = []
for seed in (0, 1, 2):
    model, _ = run([64], seed, lr=1e-2)
    with torch.no_grad():
        final_scores.append((model(Xte).argmax(1) == yte).float().mean().item())
test_logistic = baseline.score((X_test - mean) / std, y_test)
print(f"chosen MLP 1 x 64, test accuracy over 3 seeds: {np.mean(final_scores):.3f} +/- {np.std(final_scores, ddof=1):.3f}")
print(f"logistic regression baseline, test accuracy: {test_logistic:.3f}")
```

### Explication du code

L'architecture retenue (« 1 × 64 », simple et parmi les meilleures) est évaluée **une fois** sur le jeu de test, en rapportant la moyenne et l'écart-type sur 3 graines, à côté de la référence. Le MLP dépasse la régression logistique d'environ 1,5 point, soit moins de 3 exemples sur 180 : une différence trop faible pour être affirmée. Une conclusion honnête ressemble à : « sur ces données synthétiques presque linéairement séparables, un MLP à une couche cachée fait au moins aussi bien que la régression logistique ; augmenter la profondeur ou la largeur n'apporte pas de gain mesurable. La régression logistique, plus simple et plus interprétable, est un choix raisonnable. »

### Sortie attendue

```text output
chosen MLP 1 x 64, test accuracy over 3 seeds: 0.937 +/- 0.006
logistic regression baseline, test accuracy: 0.922
```

## Erreurs fréquentes

- **Une seule graine par variante**, et conclure sur un écart de 0,5 point.
- **Budgets différents** entre variantes.
- **Choisir sur le test**.
- **Oublier la référence simple**.
- **Ne rien consigner** : impossible de reproduire le meilleur résultat une semaine plus tard.
- **Explorer les taux d'apprentissage sur une échelle linéaire** (0,01 ; 0,02 ; 0,03) au lieu de logarithmique.

## Exercices guidés

1. Calculez à la main le nombre de paramètres de l'architecture « 2 × 64 » (5 entrées, 3 sorties).
2. Ajoutez l'architecture « 1 × 16 » au tableau de l'exemple 1. Où se situe-t-elle ?
3. Refaites l'exemple 1 avec 5 graines au lieu de 3 pour « 1 × 64 » et « 2 × 64 ». La conclusion change-t-elle ?
4. Ajoutez la taille de lot (16, 64, 256) au balayage de l'exemple 2 pour lr = 1e-2 sans dropout.

Corrigés : [corrections.md](corrections.md), section « Leçon 7.3 ».

## Défis autonomes

- Faites une **ablation** sur le modèle « 2 × 64 avec dropout 0,3 et weight decay » : retirez un élément à la fois et mesurez l'effet.
- Remplacez la grille de l'exemple 2 par 12 tirages aléatoires (lr log-uniforme entre 10⁻⁴ et 10⁻¹, dropout uniforme entre 0 et 0,5) et comparez les meilleurs résultats.
- Rédigez un rapport d'une page (Markdown) présentant votre protocole, vos tableaux et votre conclusion.

## Quiz

1. Pourquoi utiliser plusieurs graines ?
2. Que signifie « même budget » ?
3. Deux architectures : 0,912 ± 0,010 et 0,918 ± 0,012. Que concluez-vous ?
4. Pourquoi explorer le taux d'apprentissage sur une échelle logarithmique ?
5. Qu'est-ce qu'une étude d'ablation ?
6. Que doit contenir un journal d'expériences ?

Réponses : [corrections.md](corrections.md), section « Leçon 7.3 ».

## Résumé

- Comparaison équitable : mêmes données, même budget, plusieurs graines, sélection sur validation, test une fois.
- Rapporter moyenne ± écart-type, nombre de paramètres et temps ; comparer à une référence simple.
- Plus grand n'est pas meilleur : chercher le modèle le plus simple parmi les meilleurs.
- Tout consigner dans un journal pour pouvoir reproduire.

## Pour aller plus loin

- Module 8 : réseaux pour le texte (embeddings, attention).
- Projet 5 (`projects/05_image_classification`) : protocole expérimental appliqué à un CNN.
