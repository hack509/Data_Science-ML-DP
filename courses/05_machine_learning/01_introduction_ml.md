# Leçon 5.1 — Introduction au Machine Learning et méthodologie d'évaluation

## Prérequis

- Modules 1 à 3 ; leçon 2.5 (fonction de perte, descente de gradient).

## Objectifs d'apprentissage

- Situer IA, Machine Learning et Deep Learning les uns par rapport aux autres.
- Reconnaître un problème de régression, de classification ou de clustering.
- Découper des données en jeux d'entraînement, de validation et de test, et savoir pourquoi.
- Reconnaître et éviter la fuite de données.
- Diagnostiquer sous-apprentissage et surapprentissage et comprendre le compromis biais-variance.

## Concepts clés

- **Intelligence artificielle** (*Artificial Intelligence*, AI) : ensemble des techniques qui permettent à une machine d'accomplir des tâches que l'on associe à l'intelligence (raisonner, planifier, percevoir, comprendre le langage).
- **Apprentissage automatique** (*Machine Learning*, ML) : sous-domaine de l'IA où le comportement n'est pas programmé règle par règle mais **appris à partir d'exemples**.
- **Apprentissage profond** (*Deep Learning*, DL) : sous-domaine du ML utilisant des réseaux de neurones à nombreuses couches, qui apprennent eux-mêmes des représentations des données brutes (pixels, texte).
- **Variables** (*features*, X) et **cible** (*target*, y).
- **Apprentissage supervisé** (*supervised learning*) : on connaît la cible des exemples ; **non supervisé** (*unsupervised*) : pas de cible, on cherche une structure.
- **Généralisation** : capacité à bien prédire sur des données **jamais vues**.
- **Surapprentissage** (*overfitting*) / **sous-apprentissage** (*underfitting*).
- **Fuite de données** (*data leakage*) : information de l'évaluation qui contamine l'entraînement.
- **Hyperparamètre** : réglage choisi avant l'entraînement (profondeur d'un arbre), par opposition aux **paramètres** appris (poids).

## Explication détaillée

### IA, ML, DL

Imaginez trois cercles emboîtés. L'**IA** est le plus grand : elle inclut aussi des systèmes à règles écrites à la main (un système expert, un moteur d'échecs classique). Le **ML** est un cercle intérieur : au lieu d'écrire « si le message contient GRATUIT et un numéro surtaxé, alors spam », on donne au programme des milliers de messages étiquetés et il apprend lui-même la règle. Le **DL** est un cercle encore plus petit : des réseaux de neurones profonds, particulièrement efficaces sur les données non structurées (images, sons, textes) mais gourmands en données et en calcul. Pour des données **tabulaires** (des tableaux comme ceux de ce cours), les méthodes classiques de ML (forêts aléatoires, gradient boosting, modèles linéaires) restent souvent les meilleures.

### Les familles de problèmes

| Problème | Cible | Exemple dans ce cours | Algorithmes |
|---|---|---|---|
| régression | nombre continu | prix d'une maison | régression linéaire, arbres, forêts, k-NN |
| classification | catégorie | spam / non spam | logistique, k-NN, Naive Bayes, SVM, arbres |
| clustering | aucune | segments de clients | k-means |
| réduction de dimension | aucune | visualiser 6 variables en 2D | ACP |

### Pourquoi séparer les données ?

Un modèle peut **mémoriser** les exemples qu'il a vus. Mesurer sa performance sur ces mêmes exemples revient à noter un élève sur les exercices dont il a appris les corrigés par cœur. On sépare donc :

- le **jeu d'entraînement** (*train set*, souvent 60 à 80 %) : le modèle apprend dessus ;
- le **jeu de validation** (*validation set*) : on compare des modèles et on règle les hyperparamètres ;
- le **jeu de test** (*test set*, 15 à 20 %) : utilisé **une seule fois**, à la fin, pour estimer la performance réelle.

Si l'on choisit le modèle d'après son score de test, ce score n'est plus une estimation honnête : on a « appris » sur le test. La **validation croisée** (leçon 5.6) remplace avantageusement un jeu de validation unique quand les données sont peu nombreuses.

En classification, on utilise un découpage **stratifié** (`stratify=y`) pour conserver la proportion de chaque classe dans chaque jeu. Pour des données temporelles, on découpe **dans le temps** (entraînement sur le passé, test sur le futur), jamais au hasard.

### La fuite de données

Exemples classiques :

- **Prétraitement avant le découpage** : standardiser ou imputer avec des moyennes calculées sur toutes les données, test compris.
- **Variable qui contient la réponse** : prédire `final_score` à partir de `math`, `physics`, `french` qui la composent ; prédire un défaut de paiement avec une variable « relance envoyée » qui n'existe qu'après le défaut.
- **Doublons** entre entraînement et test (les SMS identiques du jeu de données spam !).
- **Futur dans le passé** : utiliser des informations postérieures à la date de prédiction.

Symptôme typique : un score « trop beau pour être vrai ». La protection principale est le **pipeline** scikit-learn (leçon 5.6), qui apprend chaque transformation uniquement sur les données d'entraînement.

### Sous-apprentissage, surapprentissage et biais-variance

- Un modèle **trop simple** (une droite pour une relation courbe) se trompe même sur l'entraînement : **sous-apprentissage**, ou **biais** élevé.
- Un modèle **trop complexe** (un polynôme de degré 15 sur 30 points) épouse le bruit : erreur d'entraînement minuscule, erreur de validation élevée : **surapprentissage**, ou **variance** élevée (il change beaucoup si l'on change un peu les données).
- Entre les deux, la complexité optimale minimise l'erreur de validation.

Le diagnostic se lit en comparant erreur d'entraînement et erreur de validation : toutes deux élevées → sous-apprentissage ; grand écart → surapprentissage. Remèdes au surapprentissage : plus de données, modèle plus simple, régularisation, moins de variables. Remèdes au sous-apprentissage : modèle plus riche, meilleures variables.

## Intuition mathématique

Un modèle supervisé est une fonction f_θ de paramètres θ. L'entraînement minimise une perte moyenne sur les données d'entraînement :

```text
θ* = argmin_θ (1/n) Σ L(yᵢ, f_θ(xᵢ))
```

Mais ce que l'on veut vraiment, c'est une faible erreur sur de **nouvelles** données (erreur de généralisation), qu'on estime sur le jeu de test. Pour la perte quadratique, l'erreur attendue sur un nouveau point se décompose :

```text
erreur attendue = biais² + variance + bruit irréductible
```

Le bruit irréductible ne dépend pas du modèle (dans nos données, c'est le bruit ajouté par le générateur). Diminuer le biais (modèle plus flexible) augmente généralement la variance : c'est le **compromis biais-variance**.

## Applications concrètes

- Régression : prévoir une consommation d'énergie, un délai de livraison, un prix.
- Classification : filtrer des spams, détecter des fraudes, diagnostiquer à partir d'images.
- Clustering : segmenter une clientèle, regrouper des documents similaires.
- Dans tous les cas, une mauvaise méthodologie d'évaluation conduit à déployer des modèles qui échouent en production.

## Exemples de code Python

### Exemple 1 — découper les données correctement

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("datasets/generated/binary_classification.csv")
X = df.drop(columns="target")
y = df["target"]
print("class balance:", y.value_counts(normalize=True).round(3).to_dict())

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
for name, part in [("train", y_train), ("validation", y_val), ("test", y_test)]:
    print(f"{name:<10} n={len(part):>3}  share of class 1={part.mean():.3f}")
```

### Explication du code

Deux appels successifs à `train_test_split` produisent 60 % / 20 % / 20 % (0,25 × 0,8 = 0,2). `stratify` garantit la même proportion de classe 1 dans chaque partie ; `random_state` rend le découpage reproductible.

### Sortie attendue

```text output
class balance: {0: 0.598, 1: 0.402}
train      n=360  share of class 1=0.403
validation n=120  share of class 1=0.400
test       n=120  share of class 1=0.400
```

### Exemple 2 — sous-apprentissage et surapprentissage

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from lab.plotting import CATEGORICAL, apply_style

rng = np.random.default_rng(0)
x = np.sort(rng.uniform(-3, 3, 40))
y = np.sin(x) + rng.normal(0, 0.3, 40)
x_val = np.sort(rng.uniform(-3, 3, 200))
y_val = np.sin(x_val) + rng.normal(0, 0.3, 200)

apply_style()
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
grid = np.linspace(-3, 3, 300)
for ax, degree in zip(axes, [1, 4, 15]):
    model = make_pipeline(PolynomialFeatures(degree), StandardScaler(), LinearRegression())
    model.fit(x.reshape(-1, 1), y)
    train_mse = mean_squared_error(y, model.predict(x.reshape(-1, 1)))
    val_mse = mean_squared_error(y_val, model.predict(x_val.reshape(-1, 1)))
    print(f"degree {degree:>2}: train MSE={train_mse:.3f}  validation MSE={val_mse:.3f}")
    ax.scatter(x, y, s=14, color=CATEGORICAL[0], label="entraînement")
    ax.plot(grid, np.clip(model.predict(grid.reshape(-1, 1)), -3, 3), color=CATEGORICAL[1], label="modèle")
    ax.set_title(f"degré {degree}")
axes[0].legend()
Path("outputs/module05").mkdir(parents=True, exist_ok=True)
fig.savefig("outputs/module05/under_over_fitting.png", bbox_inches="tight")
plt.close(fig)
```

### Explication du code

- Les données suivent une sinusoïde plus du bruit (écart-type 0,3, donc un bruit irréductible de MSE ≈ 0,09).
- Degré 1 (une droite) : erreurs d'entraînement et de validation élevées → **sous-apprentissage**.
- Degré 4 : les deux erreurs sont proches du bruit irréductible → bon compromis.
- Degré 15 : l'erreur d'entraînement est encore plus basse mais l'erreur de validation remonte → **surapprentissage**. La figure `outputs/module05/under_over_fitting.png` montre la courbe de degré 15 qui oscille entre les points.

### Sortie attendue

```text output
degree  1: train MSE=0.250  validation MSE=0.301
degree  4: train MSE=0.106  validation MSE=0.113
degree 15: train MSE=0.062  validation MSE=0.147
```

### Exemple 3 — une fuite de données qui embellit les scores

```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(1)
X = rng.normal(size=(100, 2000))      # pure noise: 2000 random features
y = rng.integers(0, 2, size=100)      # random labels, nothing to learn

# WRONG: select the "best" features using all the data, then cross-validate.
selected = SelectKBest(f_classif, k=20).fit(X, y).transform(X)
leaky = cross_val_score(LogisticRegression(max_iter=1000), selected, y, cv=5).mean()

# RIGHT: the selection is part of the pipeline, refit inside each training fold.
pipeline = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
honest = cross_val_score(pipeline, X, y, cv=5).mean()
print(f"leaky accuracy:  {leaky:.2f}")
print(f"honest accuracy: {honest:.2f}  (chance level = 0.50)")
```

### Explication du code

Les étiquettes sont **aléatoires** : aucun modèle ne peut faire mieux que 50 % en moyenne. Pourtant, en sélectionnant les 20 variables les plus corrélées à y **sur toutes les données**, puis en validant, on obtient une précision bien supérieure : les variables ont été choisies en regardant les réponses du jeu de validation. Placée dans un pipeline, la sélection est refaite à l'intérieur de chaque pli d'entraînement et le score retombe au niveau du hasard. (Ici 0,39 : avec seulement 100 exemples, un score de hasard fluctue facilement de ±0,1 autour de 0,50. Un score honnête *inférieur* au hasard n'a rien d'anormal sur des données sans signal.) C'est l'exemple le plus parlant de fuite de données.

### Sortie attendue

```text output
leaky accuracy:  0.88
honest accuracy: 0.39  (chance level = 0.50)
```

## Erreurs fréquentes

- **Évaluer sur les données d'entraînement**.
- **Régler les hyperparamètres sur le jeu de test** : il devient un jeu de validation, et l'estimation finale est optimiste.
- **Prétraiter avant de découper** : fuite.
- **Découpage aléatoire de données temporelles** : le modèle « voit » le futur.
- **Ne pas stratifier** une classification déséquilibrée : un jeu de test peut ne presque pas contenir la classe rare.
- **Ne pas comparer à une référence** (*baseline*) : 90 % de précision n'a rien d'impressionnant si 90 % des exemples sont de la même classe.

## Exercices guidés

1. Découpez `house_prices.csv` en 80 % / 20 % avec `random_state=0` et vérifiez que les prix moyens des deux parties sont proches.
2. Dans l'exemple 2, ajoutez les degrés 2, 3, 8 et 12 et tracez l'erreur d'entraînement et de validation en fonction du degré. Où se trouve le minimum de validation ?
3. Calculez la précision d'un classifieur qui prédit toujours la classe majoritaire sur `binary_classification.csv` (`DummyClassifier`).
4. Identifiez la fuite : on veut prédire `passed` dans `students.csv` en utilisant toutes les autres colonnes numériques. Quel problème ?

Corrigés : [corrections.md](corrections.md), section « Leçon 5.1 ».

## Défis autonomes

- Refaites l'exemple 3 avec 200, 1000 puis 5000 variables de bruit. Comment évolue le score « avec fuite » ? Pourquoi ?
- Trouvez dans `sms_spam.csv` combien de messages apparaissent plusieurs fois. Quel risque pour l'évaluation d'un filtre anti-spam, et comment l'éviter ?

## Quiz

1. Quelle différence entre IA, ML et DL ?
2. Prédire le nombre de jours d'hospitalisation : régression ou classification ?
3. À quoi servent respectivement les jeux de validation et de test ?
4. Erreur d'entraînement 0,01, erreur de validation 0,85 : diagnostic ? remèdes ?
5. Pourquoi standardiser avant le découpage est-il une fuite de données ?
6. Qu'est-ce qu'un hyperparamètre ? Donnez un exemple.

Réponses : [corrections.md](corrections.md), section « Leçon 5.1 ».

## Résumé

- ML = apprendre une fonction à partir d'exemples ; DL = réseaux profonds, surtout pour les données non structurées.
- Régression (nombre), classification (catégorie), clustering (structure sans cible).
- Entraînement / validation / test ; le test ne sert qu'une fois.
- Fuite de données = score trop optimiste ; les pipelines la préviennent.
- Biais (trop simple) contre variance (trop complexe) : on cherche le minimum de l'erreur de validation.

## Pour aller plus loin

- Exercice **A05** (`exercises/advanced/A05_cross_validation_from_scratch`).
- Leçon suivante : [Régression : régression linéaire et métriques](02_regression.md).
