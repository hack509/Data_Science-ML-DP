# Leçon 5.4 — Arbres de décision et forêts aléatoires

## Prérequis

- Leçons 5.1 à 5.3.

## Objectifs d'apprentissage

- Comprendre comment un arbre de décision découpe l'espace par une suite de questions.
- Contrôler la complexité d'un arbre (profondeur, taille des feuilles) pour éviter le surapprentissage.
- Comprendre pourquoi une forêt aléatoire est plus robuste qu'un arbre seul.
- Interpréter les importances de variables, et connaître leurs limites (importance par permutation).
- Utiliser arbres et forêts en classification multiclasse et en régression.

## Concepts clés

- **Arbre de décision** (*decision tree*) : modèle en forme d'arbre ; chaque nœud pose une question (« surface ≤ 92 m² ? »), chaque feuille donne une prédiction.
- **Impureté de Gini** (*Gini impurity*) / **entropie** : mesures du mélange des classes dans un nœud.
- **Profondeur** (*max_depth*), **taille minimale d'une feuille** (*min_samples_leaf*) : hyperparamètres de complexité.
- **Ensemble** (*ensemble*) : combinaison de plusieurs modèles.
- **Bagging** (*bootstrap aggregating*) : entraîner chaque modèle sur un échantillon bootstrap et moyenner.
- **Forêt aléatoire** (*random forest*) : bagging d'arbres + tirage aléatoire des variables candidates à chaque nœud.
- **Importance des variables** (*feature importance*) : contribution de chaque variable aux prédictions.
- **Score hors sac** (*out-of-bag score*) : évaluation sur les exemples non tirés par chaque arbre.

## Explication détaillée

### Fiche algorithme : arbre de décision

1. **Intuition** : comme un jeu de « 20 questions ». À chaque étape, on choisit la question qui sépare le mieux les exemples : « heures d'étude > 13 ? », puis « note précédente > 11 ? »... jusqu'à des groupes homogènes.
2. **Idée mathématique** : à chaque nœud, l'algorithme teste toutes les variables et tous les seuils possibles, et retient la coupure qui diminue le plus l'impureté (Gini en classification : 1 − Σ pₖ² ; variance en régression). C'est une recherche **gloutonne** (*greedy*) : la meilleure coupure locale, sans retour en arrière.
3. **Implémentation** : `DecisionTreeClassifier`, `DecisionTreeRegressor` ; `export_text` affiche les règles, `plot_tree` les dessine.
4. **Données** : `multiclass_classification.csv` (900 exemples, 5 variables, 3 classes) et `house_prices.csv`.
5. **Entraînement** : aucune standardisation nécessaire (les coupures ne dépendent que de l'ordre des valeurs).
6. **Évaluation** : accuracy, matrice de confusion 3 × 3 ; écart entraînement / test pour diagnostiquer le surapprentissage.
7. **Interprétation** : un arbre peu profond se lit comme une suite de règles « si... alors » : c'est le modèle le plus explicable.
8. **Erreurs fréquentes** : laisser l'arbre pousser sans limite (il apprend chaque exemple par cœur, 100 % sur l'entraînement) ; instabilité (un petit changement des données change tout l'arbre) ; frontières « en escalier » parallèles aux axes.
9-10. **Exercices et corrigé** : voir plus bas et `corrections.md`.

### Fiche algorithme : forêt aléatoire

1. **Intuition** : la « sagesse des foules ». Beaucoup d'arbres un peu différents, chacun imparfait, votent ; leurs erreurs se compensent en partie.
2. **Idée mathématique** : chaque arbre est entraîné sur un échantillon bootstrap (tirage avec remise) et, à chaque nœud, ne considère qu'un sous-ensemble aléatoire de variables (`max_features`). Ce double hasard **décorrèle** les arbres ; la moyenne de modèles peu corrélés a une variance bien plus faible que chaque modèle. Le biais reste celui d'arbres profonds (faible).
3. **Implémentation** : `RandomForestClassifier`, `RandomForestRegressor` ; `n_estimators` (nombre d'arbres, plus = plus stable mais plus lent), `max_features`, `min_samples_leaf`, `n_jobs=-1` pour utiliser tous les cœurs du processeur, `oob_score=True`.
4. à 6. Mêmes données ; le score hors sac donne une estimation de validation « gratuite ».
7. **Interprétation** : `feature_importances_` (diminution moyenne d'impureté) est rapide mais biaisée en faveur des variables continues ou à nombreuses valeurs ; l'**importance par permutation** (`permutation_importance`), calculée sur un jeu de test, mesure la perte de performance quand on mélange une variable : plus fiable.
8. **Erreurs fréquentes** : croire qu'une forêt ne surapprend jamais (elle peut, avec du bruit) ; extrapolation impossible en régression (une forêt ne prédit jamais au-delà des valeurs vues) ; modèles lourds en mémoire avec des centaines d'arbres profonds ; importances interprétées comme des causes.

### Et le gradient boosting ?

Les méthodes de **boosting** (`HistGradientBoostingClassifier`, et les bibliothèques XGBoost, LightGBM) construisent les arbres **séquentiellement**, chacun corrigeant les erreurs des précédents. Elles sont souvent les plus performantes sur des données tabulaires. `HistGradientBoosting` est inclus dans scikit-learn et fonctionne hors ligne ; il est utilisé dans le projet 6.

## Intuition mathématique

```text
Gini(nœud) = 1 − Σₖ pₖ²          (0 = pur ; 0,5 = deux classes à 50/50)
gain d'une coupure = Gini(parent) − [nG/n · Gini(gauche) + nD/n · Gini(droite)]
variance de la moyenne de B modèles de variance σ² et de corrélation ρ :  ρσ² + (1 − ρ)σ²/B
```

La dernière formule explique la forêt : augmenter B fait disparaître le second terme ; seul le terme ρσ² subsiste, d'où l'intérêt de **décorréler** les arbres (ρ petit) avec le tirage des variables.

## Applications concrètes

- Octroi de crédit avec règles explicables (arbre peu profond).
- Prédiction de churn, de fraude, de prix : forêts et boosting sont des choix par défaut solides pour les tableaux de données.
- Repérage des variables qui influencent le plus une prédiction (avec prudence).

## Exemples de code Python

### Exemple 1 — un arbre, sa complexité et ses règles

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

df = pd.read_csv("datasets/generated/multiclass_classification.csv")
X, y = df.drop(columns="target"), df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

for depth in [1, 2, 4, 8, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_train, y_train)
    print(f"max_depth={str(depth):<4} leaves={tree.get_n_leaves():>3}  train={tree.score(X_train, y_train):.3f}  test={tree.score(X_test, y_test):.3f}")

small_tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
print(export_text(small_tree, feature_names=list(X.columns), decimals=2))
```

### Explication du code

- `max_depth=1` (une seule question) sous-apprend. Sans limite (`None`), l'arbre atteint 100 % sur l'entraînement avec de nombreuses feuilles, mais le score de test ne suit pas : surapprentissage typique. Une profondeur intermédiaire donne le meilleur test.
- `export_text` affiche les règles de l'arbre de profondeur 2 : lisibles par un humain, c'est le grand atout des arbres.

### Sortie attendue

```text output
max_depth=1    leaves=  2  train=0.655  test=0.640
max_depth=2    leaves=  4  train=0.904  test=0.898
max_depth=4    leaves= 14  train=0.938  test=0.920
max_depth=8    leaves= 52  train=0.990  test=0.871
max_depth=None leaves= 62  train=1.000  test=0.876
|--- x4 <= 0.02
|   |--- x2 <= -2.39
|   |   |--- class: 2
|   |--- x2 >  -2.39
|   |   |--- class: 0
|--- x4 >  0.02
|   |--- x2 <= 0.13
|   |   |--- class: 2
|   |--- x2 >  0.13
|   |   |--- class: 1
```

### Exemple 2 — forêt aléatoire, score hors sac, matrice de confusion

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, oob_score=True, n_jobs=-1, random_state=0)
forest.fit(X_train, y_train)
print(f"random forest: train={forest.score(X_train, y_train):.3f}  OOB={forest.oob_score_:.3f}  test={forest.score(X_test, y_test):.3f}")
print(confusion_matrix(y_test, forest.predict(X_test)))

for n in [1, 10, 100, 300]:
    model = RandomForestClassifier(n_estimators=n, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(X_train, y_train)
    print(f"n_estimators={n:<4} test={model.score(X_test, y_test):.3f}")
```

### Explication du code

- La forêt obtient un meilleur score de test que le meilleur arbre seul, et le score hors sac (OOB), calculé uniquement sur l'entraînement, en donne une bonne estimation.
- La matrice de confusion 3 × 3 montre quelles classes sont confondues (lignes : vraies classes, colonnes : prédictions).
- Avec 1 arbre, la forêt est un arbre bootstrap moyen ; le score se stabilise quand le nombre d'arbres augmente : plus d'arbres ne cause pas de surapprentissage, seulement du temps de calcul.

### Sortie attendue

```text output
random forest: train=0.979  OOB=0.921  test=0.929
[[72  1  1]
 [ 3 69  4]
 [ 0  7 68]]
n_estimators=1    test=0.840
n_estimators=10   test=0.929
n_estimators=100  test=0.920
n_estimators=300  test=0.929
```

### Exemple 3 — régression et importance des variables

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

houses = pd.read_csv("datasets/generated/house_prices.csv")
houses["energy_rank"] = houses["energy_class"].map({c: i for i, c in enumerate("ABCDEFG", start=1)})
houses["random_noise"] = np.random.default_rng(0).normal(size=len(houses))
Xh = pd.get_dummies(houses.drop(columns=["price_keur", "energy_class"]), columns=["neighborhood"], dtype=float)
yh = houses["price_keur"]
Xh_train, Xh_test, yh_train, yh_test = train_test_split(Xh, yh, test_size=0.2, random_state=0)

rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(Xh_train, yh_train)
print("random forest MAE:", round(mean_absolute_error(yh_test, rf.predict(Xh_test)), 1))

impurity = pd.Series(rf.feature_importances_, index=Xh.columns)
permutation = permutation_importance(rf, Xh_test, yh_test, n_repeats=10, random_state=0, n_jobs=-1)
importance = pd.DataFrame({"impurity": impurity, "permutation (R2 drop)": permutation.importances_mean})
print(importance.sort_values("impurity", ascending=False).round(3).to_string())
```

### Explication du code

- On ajoute volontairement une variable `random_noise` sans aucun lien avec le prix.
- La forêt fait moins bien que la régression linéaire sur log(prix) de la leçon 5.2 : la relation est presque multiplicative et lisse, ce que des arbres approchent « en escalier ». Encore une fois : comparer.
- L'importance par permutation se lit comme la baisse du R² sur le test quand on mélange la variable (score par défaut d'un régresseur). L'importance par impureté attribue un poids non nul à la variable de bruit (une variable continue offre beaucoup de seuils de coupure possibles) ; l'importance par permutation, calculée sur le test, la ramène près de zéro. La surface domine dans les deux cas.

### Sortie attendue

```text output
random forest MAE: 26.7
                          impurity  permutation (R2 drop)
surface_m2                   0.643                       1.235
distance_center_km           0.189                       0.224
neighborhood_Centre          0.076                       0.132
neighborhood_Residentiel     0.024                       0.022
age_years                    0.021                       0.015
neighborhood_Rural           0.012                       0.014
random_noise                 0.009                      -0.000
energy_rank                  0.008                       0.006
rooms                        0.008                      -0.002
neighborhood_Peripherie      0.006                       0.002
has_garden                   0.004                      -0.000
```

## Erreurs fréquentes

- **Arbre sans limite de profondeur** : mémorisation de l'entraînement.
- **Lire l'importance par impureté comme une vérité** : biaisée vers les variables continues ; préférez la permutation.
- **Importance = causalité** : non.
- **Forêt en régression hors de la plage d'entraînement** : elle plafonne aux valeurs vues.
- **Standardiser pour un arbre** : inutile (pas nuisible).
- **Variables très corrélées** : l'importance se répartit entre elles et chacune paraît moins importante.

## Exercices guidés

1. Calculez à la main l'impureté de Gini d'un nœud contenant 30 exemples de la classe A et 10 de la classe B, puis le gain d'une coupure qui donne (25 A, 2 B) à gauche et (5 A, 8 B) à droite.
2. Pour l'arbre de classification, faites varier `min_samples_leaf` (1, 5, 20, 50) avec `max_depth=None`. Que constatez-vous ?
3. Comparez sur `binary_classification.csv` une forêt aléatoire et le SVM de la leçon 5.3 (même découpage, `random_state=0`).
4. Entraînez un `HistGradientBoostingRegressor` sur les données de l'exemple 3 et comparez sa MAE à celle de la forêt.

Corrigés : [corrections.md](corrections.md), section « Leçon 5.4 ».

## Défis autonomes

- Dessinez l'arbre de profondeur 3 avec `sklearn.tree.plot_tree` et enregistrez-le en PNG. Écrivez en français les trois règles les plus fréquentes.
- Montrez qu'une forêt ne sait pas extrapoler : entraînez-la sur les maisons de moins de 150 m² et prédisez des maisons de 250 m². Comparez avec la régression linéaire.

## Quiz

1. Comment un arbre choisit-il la question posée à un nœud ?
2. Pourquoi un arbre sans limite surapprend-il ?
3. Quels sont les deux sources de hasard d'une forêt aléatoire ?
4. Ajouter des arbres à une forêt provoque-t-il du surapprentissage ?
5. Qu'est-ce que le score hors sac (OOB) ?
6. Pourquoi l'importance par permutation est-elle plus fiable que l'importance par impureté ?

Réponses : [corrections.md](corrections.md), section « Leçon 5.4 ».

## Résumé

- Arbre : questions successives choisies pour réduire l'impureté ; lisible mais instable et prompt à surapprendre.
- Forêt : moyenne d'arbres décorrélés (bootstrap + variables aléatoires) ; robuste, peu de réglages, pas de standardisation.
- Importances : utiles pour comprendre, à vérifier par permutation, jamais causales.

## Pour aller plus loin

- Projet 6 (`projects/06_end_to_end_ml_project`) : gradient boosting dans un pipeline complet.
- Leçon suivante : [Apprentissage non supervisé : k-means et ACP](05_non_supervise.md).
