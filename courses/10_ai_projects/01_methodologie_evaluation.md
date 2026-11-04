# Leçon 10.1 — Mener un projet de Machine Learning et l'évaluer honnêtement

## Prérequis

- Module 5 complet ; leçon 2.4 (intervalles de confiance) ; leçon 7.3 (expérimentation).

## Objectifs d'apprentissage

- Suivre les étapes d'un projet de ML, du cadrage à la mise à disposition.
- Traduire un objectif métier en tâche, en cible et en métrique, et fixer une référence.
- Quantifier l'incertitude d'une métrique (intervalle de confiance bootstrap).
- Mener une analyse d'erreurs et une évaluation par sous-groupe.
- Organiser le code d'un projet pour qu'il soit reproductible et testable.

## Concepts clés

- **Cadrage** (*problem framing*) : définir l'objectif, l'utilisateur, la décision que le modèle aide à prendre, et le coût des erreurs.
- **Référence** (*baseline*) : solution simple à battre (règle métier, moyenne, modèle linéaire).
- **Métrique principale et métriques de garde-fou** (*guardrail metrics*) : ce que l'on optimise, et ce qui ne doit pas se dégrader.
- **Intervalle de confiance bootstrap** : incertitude d'une métrique obtenue en rééchantillonnant le jeu de test.
- **Analyse d'erreurs** (*error analysis*) : examen systématique des erreurs pour comprendre leurs causes.
- **Évaluation par sous-groupe** (*slice-based evaluation*) : mesurer la performance séparément pour chaque groupe.
- **Reproductibilité** : obtenir les mêmes résultats en relançant le même code sur les mêmes données.

## Explication détaillée

### Les étapes d'un projet

1. **Cadrer** : quel problème, pour qui, quelle décision ? Un modèle de détection de spam sert à éviter des arnaques **sans** perdre de messages importants : le coût d'un faux positif est élevé.
2. **Données** : d'où viennent-elles, que représentent-elles, quels biais de collecte, quelles autorisations (vie privée) ? Explorer et nettoyer (modules 3 et 4).
3. **Métrique et référence** avant de modéliser : sinon on choisit la métrique qui flatte le modèle.
4. **Découpage** entraînement / validation / test (ou validation croisée), sans fuite.
5. **Modéliser** de façon itérative : référence → modèle simple → modèles plus complexes, en consignant chaque essai.
6. **Évaluer** honnêtement : test utilisé une fois, incertitude, sous-groupes, erreurs.
7. **Livrer** : modèle sauvegardé, fonction de prédiction, documentation, tests.
8. **Surveiller** et réentraîner (leçon 10.2).

Ce cycle n'est pas linéaire : l'analyse d'erreurs renvoie souvent à l'étape 2 (des données mal étiquetées, une variable manquante).

### Choisir la métrique

La métrique traduit le coût des erreurs : MAE si l'erreur moyenne en euros parle aux utilisateurs ; RMSE si les grosses erreurs sont particulièrement graves ; rappel si manquer un cas positif est grave ; précision si les fausses alertes coûtent cher ; AUC pour comparer des classements indépendamment du seuil. On ajoute des **garde-fous** : temps de prédiction, performance minimale dans chaque sous-groupe, taux de fausses alertes.

### L'incertitude d'une métrique

Un score de test est calculé sur un échantillon : avec 160 maisons de test, une MAE de 25,2 k€ pourrait tout aussi bien valoir 22 ou 29 sur un autre échantillon (c'est l'intervalle calculé dans l'exemple 1). Le **bootstrap** estime cette incertitude : on tire avec remise 1 000 jeux de test de même taille, on recalcule la métrique sur chacun, et l'on prend les percentiles 2,5 % et 97,5 %. Deux modèles dont les intervalles se recouvrent largement ne sont pas clairement différents. Pour comparer deux modèles sur le **même** jeu de test, on calcule plutôt l'intervalle de la **différence** (bootstrap apparié), plus précis.

### Analyse d'erreurs et sous-groupes

Un score global peut cacher de fortes disparités. On calcule la métrique par sous-groupe (quartier, tranche d'âge, catégorie de produit) et on lit les plus grosses erreurs. Questions à se poser : les erreurs sont-elles concentrées sur un groupe ? Les données de ce groupe sont-elles rares ? Une variable importante manque-t-elle ? Les étiquettes sont-elles fiables (leçon 8.2) ? Les groupes très petits ont des métriques très incertaines : indiquez toujours leur effectif.

### Organiser un projet reproductible

La structure utilisée dans `projects/` :

```text
projects/NN_nom/
├── README.md            énoncé, objectifs, étapes, défis
├── src/                 code de la solution de référence (fonctions testables)
├── tests/               tests pytest du projet
├── run.py               point d'entrée : python projects/NN_nom/run.py
└── outputs/             résultats générés (ignorés par Git)
```

Principes : graines aléatoires fixées, chemins relatifs à la racine, paramètres regroupés (dictionnaire de configuration), fonctions courtes testées, résultats écrits dans des fichiers (métriques en JSON, figures en PNG, rapport en Markdown).

## Intuition mathématique

```text
bootstrap : pour b = 1..B : tirer n indices avec remise, calculer m_b = métrique(y[indices], ŷ[indices])
IC à 95 % ≈ [percentile 2,5 % ; percentile 97,5 %] des m_b
comparaison appariée : d_b = métrique_A(indices_b) − métrique_B(indices_b) ; si l'IC de d exclut 0, la différence est nette
```

## Applications concrètes

- Rédiger le rapport d'évaluation d'un modèle avant sa mise en service.
- Décider entre deux modèles avec une base chiffrée et honnête.
- Repérer qu'un modèle fonctionne mal pour un groupe d'utilisateurs avant qu'ils ne s'en plaignent.

## Exemples de code Python

### Exemple 1 — référence, modèle et intervalle de confiance bootstrap

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

houses = pd.read_csv("datasets/generated/house_prices.csv")
X, y = houses.drop(columns="price_keur"), houses["price_keur"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
numeric = ["surface_m2", "rooms", "age_years", "distance_center_km", "has_garden"]
preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("hood", OneHotEncoder(handle_unknown="ignore"), ["neighborhood"]),
    ("energy", OrdinalEncoder(categories=[list("ABCDEFG")]), ["energy_class"]),
])
baseline = DummyRegressor(strategy="median").fit(X_train, y_train)
model = Pipeline([("prep", preprocess), ("reg", LinearRegression())])
model.fit(X_train, np.log(y_train))  # log target, as found in lesson 5.2
pred_model = np.exp(model.predict(X_test))
pred_base = baseline.predict(X_test)


def bootstrap_ci(y_true, y_pred, metric, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    values = [metric(y_true[idx], y_pred[idx]) for idx in rng.integers(0, len(y_true), size=(n_boot, len(y_true)))]
    return np.percentile(values, [2.5, 97.5])


for name, pred in [("baseline (median)", pred_base), ("linear on log(price)", pred_model)]:
    low, high = bootstrap_ci(y_test, pred, mean_absolute_error)
    print(f"{name:<22} MAE={mean_absolute_error(y_test, pred):6.1f} k EUR  95% CI=[{low:.1f}, {high:.1f}]")
```

### Explication du code

- La référence (prédire la médiane des prix) fixe le niveau à battre ; le modèle linéaire sur le logarithme du prix reprend la meilleure approche trouvée en leçon 5.2, ici dans un pipeline propre.
- L'intervalle bootstrap montre l'incertitude de chaque MAE : avec 160 maisons de test, la MAE du modèle n'est connue qu'à quelques k€ près. Les intervalles du modèle et de la référence sont très éloignés : la supériorité du modèle est nette.

### Sortie attendue

```text output
baseline (median)      MAE= 112.5 k EUR  95% CI=[97.9, 129.0]
linear on log(price)   MAE=  25.2 k EUR  95% CI=[21.8, 28.8]
```

### Exemple 2 — évaluation par sous-groupe et analyse des plus grosses erreurs

```python
evaluation = X_test.assign(price=y_test, prediction=pred_model)
evaluation["abs_error"] = (evaluation["price"] - evaluation["prediction"]).abs()
evaluation["relative_error"] = evaluation["abs_error"] / evaluation["price"]

by_group = evaluation.groupby("neighborhood").agg(
    n=("price", "size"), mae=("abs_error", "mean"), mean_relative_error=("relative_error", "mean"), mean_price=("price", "mean"),
)
print(by_group.round(3).sort_values("mae", ascending=False).to_string())

worst = evaluation.nlargest(3, "abs_error")[["surface_m2", "neighborhood", "age_years", "price", "prediction", "abs_error"]]
print(worst.round(1).to_string())
```

### Explication du code

- La MAE varie fortement selon le quartier, mais il faut la lire avec le prix moyen : en erreur **relative**, les écarts entre quartiers sont bien plus faibles. Une erreur de 40 k€ sur une maison à 600 k€ n'a pas la même gravité que sur une maison à 150 k€.
- Les plus grosses erreurs concernent des maisons chères (plus de 400 k€), peu nombreuses dans les données ; la pire est une très grande maison (193 m²) surestimée de 146 k€. Un point de départ pour l'amélioration : plus d'exemples de ce type, ou des variables supplémentaires.
- Les effectifs `n` rappellent que les métriques des petits groupes sont incertaines.

### Sortie attendue

```text output
               n     mae  mean_relative_error  mean_price
neighborhood                                             
Centre        35  32.733                0.089     419.294
Residentiel   62  27.210                0.085     306.529
Peripherie    39  21.954                0.111     231.467
Rural         24  14.376                0.091     151.038
     surface_m2 neighborhood  age_years  price  prediction  abs_error
570       193.1  Residentiel         22  538.3       684.3      146.0
156        81.5       Centre         39  408.9       310.4       98.5
682       120.7       Centre         21  557.8       469.6       88.2
```

### Exemple 3 — comparer deux modèles de façon appariée

```python
from sklearn.ensemble import RandomForestRegressor

forest = Pipeline([("prep", preprocess), ("reg", RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1))])
pred_forest = forest.fit(X_train, y_train).predict(X_test)

rng = np.random.default_rng(1)
y_arr, a, b = y_test.to_numpy(), pred_model, pred_forest
differences = []
for idx in rng.integers(0, len(y_arr), size=(1000, len(y_arr))):
    differences.append(mean_absolute_error(y_arr[idx], b[idx]) - mean_absolute_error(y_arr[idx], a[idx]))
low, high = np.percentile(differences, [2.5, 97.5])
print(f"MAE forest={mean_absolute_error(y_arr, b):.1f}  MAE linear(log)={mean_absolute_error(y_arr, a):.1f}")
print(f"paired difference (forest - linear): 95% CI=[{low:.1f}, {high:.1f}] -> {'clear' if low > 0 or high < 0 else 'not clear'}")
```

### Explication du code

On rééchantillonne **les mêmes indices** pour les deux modèles et on calcule la différence de MAE à chaque tirage : les variations dues à l'échantillon touchent les deux modèles de la même façon et se compensent. Si l'intervalle de la différence ne contient pas 0, l'écart est net ; sinon, on ne peut pas départager les modèles avec ce jeu de test.

### Sortie attendue

```text output
MAE forest=25.8  MAE linear(log)=25.2
paired difference (forest - linear): 95% CI=[-3.3, 5.1] -> not clear
```

## Erreurs fréquentes

- **Choisir la métrique après avoir vu les résultats**.
- **Ne pas définir de référence**.
- **Présenter un score sans incertitude**, surtout sur un petit jeu de test.
- **Ne regarder que le score global** : des groupes entiers peuvent être mal servis.
- **Comparer des MAE de groupes aux prix très différents** sans erreur relative.
- **Projet non reproductible** : graines non fixées, chemins absolus, étapes manuelles.

## Exercices guidés

1. Calculez l'intervalle bootstrap du R² du modèle linéaire de l'exemple 1.
2. Ajoutez à l'évaluation par sous-groupe une découpe par classe énergétique. Quelle classe a la plus forte erreur relative ? Quel est son effectif ?
3. Refaites la comparaison appariée avec `n_boot = 200` puis `2000`. L'intervalle change-t-il beaucoup ?
4. Pour le projet de détection de spam, proposez une métrique principale et deux garde-fous, en justifiant.

Corrigés : [corrections.md](corrections.md), section « Leçon 10.1 ».

## Défis autonomes

- Écrivez une fonction `evaluation_report(y_true, y_pred, groups)` qui renvoie un tableau Markdown des métriques globales et par groupe, avec intervalles bootstrap.
- Appliquez la méthodologie complète au projet 2 et rédigez une page de conclusions.

## Quiz

1. Pourquoi définir la métrique et la référence avant de modéliser ?
2. Qu'est-ce qu'un garde-fou ? Donnez un exemple.
3. Comment calcule-t-on un intervalle de confiance bootstrap ?
4. Pourquoi une comparaison appariée est-elle plus précise ?
5. Pourquoi évaluer par sous-groupe ?
6. Citez trois pratiques qui rendent un projet reproductible.

Réponses : [corrections.md](corrections.md), section « Leçon 10.1 ».

## Résumé

- Cadrer, explorer, fixer métrique et référence, découper, itérer, évaluer, livrer, surveiller.
- Toujours accompagner un score de son incertitude (bootstrap) et de son détail par sous-groupe.
- Lire les erreurs pour améliorer données et modèle.
- Organiser le code pour qu'un autre (ou vous dans six mois) puisse tout relancer.

## Pour aller plus loin

- Projets 1 à 6 du dossier `projects/`.
- Leçon suivante : [Déploiement local, surveillance et IA responsable](02_deploiement_ia_responsable.md).
