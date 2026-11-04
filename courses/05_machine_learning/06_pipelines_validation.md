# Leçon 5.6 — Pipelines, validation croisée, hyperparamètres et persistance

## Prérequis

- Leçons 5.1 à 5.5 ; leçon 3.1 (imputation) et 3.2 (encodage).

## Objectifs d'apprentissage

- Construire un pipeline de prétraitement complet avec `Pipeline` et `ColumnTransformer`.
- Évaluer un modèle par validation croisée et lire la dispersion des scores.
- Régler des hyperparamètres avec `GridSearchCV` / `RandomizedSearchCV` sans toucher au jeu de test.
- Sélectionner des variables de façon sûre.
- Sauvegarder et recharger un modèle avec `joblib`, en connaissant les précautions associées.

## Concepts clés

- **Pipeline** : suite d'étapes (transformateurs puis modèle) qui s'entraîne et prédit comme un seul objet.
- **Transformateur** (*transformer*) : objet avec `fit` (apprendre des paramètres, ex. moyennes) et `transform` (les appliquer).
- **ColumnTransformer** : applique des transformations différentes selon les colonnes (numériques / catégorielles).
- **Validation croisée à k plis** (*k-fold cross-validation*) : k entraînements, chacun évalué sur un pli différent.
- **Recherche sur grille** (*grid search*) : essayer toutes les combinaisons d'hyperparamètres, chacune évaluée par validation croisée.
- **Sélection de variables** (*feature selection*) : garder les variables utiles (filtres statistiques, modèles, élimination récursive).
- **Persistance** (*model persistence*) : sauvegarder un modèle entraîné pour le réutiliser.

## Explication détaillée

### Pourquoi des pipelines ?

Le prétraitement fait partie du modèle : les médianes d'imputation, les moyennes de standardisation, la liste des catégories connues sont des **paramètres appris**. Ils doivent être appris sur l'entraînement seulement et appliqués tels quels aux nouvelles données. Un `Pipeline` le garantit automatiquement : `pipeline.fit(X_train, y_train)` appelle `fit_transform` sur chaque étape puis `fit` sur le modèle ; `pipeline.predict(X_new)` appelle `transform` puis `predict`. Avantages :

- **pas de fuite** pendant la validation croisée (le prétraitement est refait dans chaque pli) ;
- **un seul objet** à sauvegarder et à déployer ;
- les hyperparamètres du prétraitement se règlent comme ceux du modèle (`"preprocess__num__imputer__strategy"`).

`ColumnTransformer` traite chaque type de colonne : numériques → imputation par la médiane + standardisation ; catégorielles → imputation par le mode + `OneHotEncoder(handle_unknown="ignore")` (une catégorie jamais vue à l'entraînement ne fait pas planter la prédiction).

### Validation croisée

Un seul découpage entraînement/validation donne un score qui dépend du hasard du découpage. La validation croisée à k plis découpe l'entraînement en k parties ; chaque partie sert une fois de validation. On obtient k scores : leur **moyenne** estime la performance, leur **écart-type** sa stabilité. Deux modèles dont les moyennes diffèrent moins que l'écart-type ne sont pas clairement différents. Variantes : `StratifiedKFold` (classification), `GroupKFold` (quand plusieurs lignes concernent la même personne : elles doivent rester dans le même pli), `TimeSeriesSplit` (données temporelles).

### Régler les hyperparamètres

`GridSearchCV(pipeline, param_grid, cv=5)` essaie chaque combinaison, l'évalue par validation croisée **sur l'entraînement**, puis réentraîne la meilleure sur tout l'entraînement (`refit=True`). Le jeu de test n'intervient qu'à la toute fin. Quand la grille est grande, `RandomizedSearchCV` tire un nombre fixe de combinaisons au hasard, souvent aussi efficace pour bien moins de calcul. Attention : le meilleur score de validation croisée est légèrement optimiste (on a choisi le maximum parmi beaucoup d'essais), d'où l'importance du test final.

### Sélection de variables

Moins de variables = modèles plus simples, plus rapides, parfois meilleurs. Méthodes : **filtres** (`SelectKBest` avec un test statistique), **modèles** (`SelectFromModel` avec les coefficients d'un Lasso ou les importances d'une forêt), **élimination récursive** (`RFECV`). La règle d'or de la leçon 5.1 s'applique : la sélection doit être **dans le pipeline**, sinon la validation croisée est faussée.

### Persistance

`joblib.dump(pipeline, "model.joblib")` enregistre le pipeline complet ; `joblib.load` le recharge. Précautions :

- recharger avec les **mêmes versions** de Python et de scikit-learn (enregistrez-les à côté du modèle) ;
- ne **jamais** charger un fichier joblib/pickle d'origine inconnue : le chargement peut exécuter du code arbitraire ;
- enregistrer aussi les métadonnées : date, données utilisées, métriques, liste des colonnes attendues.

## Intuition mathématique

```text
score CV = (1/k) Σⱼ score(modèle entraîné sans le pli j, évalué sur le pli j)
incertitude ≈ écart-type des k scores (indicatif : les plis ne sont pas indépendants)
coût d'une grille : (nombre de combinaisons) × k entraînements
```

## Applications concrètes

- Tout projet de ML destiné à être utilisé : le pipeline est l'unité de déploiement.
- Comparaison équitable de plusieurs familles de modèles.
- Mise à jour périodique d'un modèle avec un script reproductible (projet 6).

## Exemples de code Python

### Exemple 1 — un pipeline complet sur des données brutes

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

students = pd.read_csv("datasets/generated/students.csv")
rng = np.random.default_rng(0)
students.loc[rng.choice(len(students), 25, replace=False), "sleep_hours"] = np.nan  # simulate missing values

numeric = ["study_hours_week", "attendance_rate", "sleep_hours", "previous_score"]
categorical = ["group"]
X = students[numeric + categorical]
y = students["passed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
])
model = Pipeline([("preprocess", preprocess), ("classifier", LogisticRegression(max_iter=1000))])

scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
print("CV ROC-AUC per fold:", scores.round(3))
print(f"mean = {scores.mean():.3f} +/- {scores.std():.3f}")
model.fit(X_train, y_train)
print("learned medians:", model.named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"].statistics_.round(2))
print("output features:", list(model.named_steps["preprocess"].get_feature_names_out()))
```

### Explication du code

- On simule 25 valeurs manquantes pour montrer l'imputation dans le pipeline.
- Note : on n'utilise **pas** les notes `math`, `physics`, `french` qui composent la cible (fuite, leçon 5.1).
- `cross_val_score` refait tout le prétraitement dans chaque pli. La dispersion des 5 scores donne une idée de l'incertitude.
- Après `fit`, on peut inspecter les paramètres appris : les médianes d'imputation (calculées sur l'entraînement seul) et les noms des colonnes produites.

### Sortie attendue

```text output
CV ROC-AUC per fold: [0.831 0.917 0.799 0.776 0.902]
mean = 0.845 +/- 0.056
learned medians: [9.45 0.85 7.2  9.7 ]
output features: ['num__study_hours_week', 'num__attendance_rate', 'num__sleep_hours', 'num__previous_score', 'cat__group_A', 'cat__group_B', 'cat__group_C']
```

### Exemple 2 — recherche d'hyperparamètres et évaluation finale

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

candidates = {
    "logistic": (model, {"classifier__C": [0.01, 0.1, 1, 10]}),
    "random forest": (
        Pipeline([("preprocess", preprocess), ("classifier", RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1))]),
        {"classifier__max_depth": [3, 6, None], "classifier__min_samples_leaf": [1, 5, 20]},
    ),
}
results = {}
for name, (pipeline, grid) in candidates.items():
    search = GridSearchCV(pipeline, grid, cv=5, scoring="roc_auc", n_jobs=-1).fit(X_train, y_train)
    results[name] = search
    print(f"{name:<14} best CV AUC={search.best_score_:.3f}  params={search.best_params_}")

best_name = max(results, key=lambda n: results[n].best_score_)
final_model = results[best_name].best_estimator_
test_auc = roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1])
print(f"selected: {best_name} | test AUC (used once) = {test_auc:.3f}")
```

### Explication du code

- Les noms des hyperparamètres suivent la structure du pipeline : `classifier__C` désigne le paramètre `C` de l'étape `classifier`.
- Chaque combinaison est évaluée par validation croisée sur l'entraînement uniquement ; le choix du modèle se fait sur `best_score_`.
- Le jeu de test n'est utilisé qu'une fois, à la fin, pour le modèle retenu. Le score de test (≈ 0,91) dépasse la moyenne de validation croisée (≈ 0,85) mais reste dans la plage des plis (0,78 à 0,92) : avec seulement 60 étudiants de test, ce score est lui-même incertain. La régression logistique, plus simple, bat ici la forêt aléatoire : la relation est proche de linéaire et les données sont peu nombreuses.

### Sortie attendue

```text output
logistic       best CV AUC=0.845  params={'classifier__C': 1}
random forest  best CV AUC=0.830  params={'classifier__max_depth': 6, 'classifier__min_samples_leaf': 5}
selected: logistic | test AUC (used once) = 0.906
```

### Exemple 3 — sauvegarder, recharger, documenter

```python
import json
import platform
from pathlib import Path

import joblib
import sklearn

model_dir = Path("outputs/module05/models")
model_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(final_model, model_dir / "pass_classifier.joblib")
metadata = {
    "model": best_name,
    "features": numeric + categorical,
    "test_roc_auc": round(test_auc, 4),
    "trained_on": "datasets/generated/students.csv (synthetic data)",
    "python": platform.python_version(),
    "scikit_learn": sklearn.__version__,
}
(model_dir / "pass_classifier.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

reloaded = joblib.load(model_dir / "pass_classifier.joblib")
new_students = pd.DataFrame({
    "study_hours_week": [4.0, 18.0], "attendance_rate": [0.65, 0.97],
    "sleep_hours": [np.nan, 7.5], "previous_score": [7.0, 14.0], "group": ["B", "Z"],
})
print("probabilities of passing:", reloaded.predict_proba(new_students)[:, 1].round(3))
print("same predictions as before saving:", np.allclose(reloaded.predict_proba(X_test), final_model.predict_proba(X_test)))
print(json.loads((model_dir / "pass_classifier.json").read_text(encoding="utf-8"))["scikit_learn"] == sklearn.__version__)
```

### Explication du code

- Le pipeline entier (prétraitement + modèle) est sauvegardé : les nouvelles données peuvent arriver brutes, avec valeurs manquantes.
- Le groupe `"Z"`, inconnu à l'entraînement, ne provoque pas d'erreur grâce à `handle_unknown="ignore"`.
- Les métadonnées JSON documentent le modèle ; la vérification finale confirme que la version de scikit-learn enregistrée est celle utilisée au rechargement.

### Sortie attendue

```text output
probabilities of passing: [0.025 0.974]
same predictions as before saving: True
True
```

## Erreurs fréquentes

- **Prétraiter hors du pipeline** puis faire une validation croisée : fuite.
- **Utiliser le jeu de test pour choisir les hyperparamètres**.
- **Grilles énormes** : des heures de calcul ; commencez grossier, affinez ensuite, ou utilisez `RandomizedSearchCV`.
- **Ignorer l'écart-type des scores** : des différences dans le bruit ne justifient pas un choix.
- **Charger un fichier `.joblib` non fiable** : risque de sécurité.
- **Sauvegarder le modèle sans ses métadonnées** (versions, colonnes, métriques).
- **`OneHotEncoder` sans `handle_unknown="ignore"`** : plantage en production à la première catégorie nouvelle.

## Exercices guidés

1. Remplacez la régression logistique de l'exemple 1 par un SVM (`SVC(probability=True)`) et comparez les AUC de validation croisée.
2. Ajoutez au pipeline une étape `SelectKBest(f_classif, k=3)` après le prétraitement. Quelles variables sont retenues ?
3. Utilisez `RandomizedSearchCV` avec 10 tirages pour la forêt aléatoire (distributions : `max_depth` dans [2, 12], `min_samples_leaf` dans [1, 30]).
4. Rechargez le modèle sauvegardé dans un **nouveau** script Python et prédisez la probabilité de réussite d'un étudiant de votre choix.

Corrigés : [corrections.md](corrections.md), section « Leçon 5.6 ».

## Défis autonomes

- Implémentez vous-même la validation croisée (exercice A05) et vérifiez que vous obtenez les mêmes scores que `cross_val_score` avec `KFold(shuffle=False)`.
- Construisez un pipeline pour `house_prices.csv` (exercice I07) et comparez trois modèles par validation croisée.

## Quiz

1. Pourquoi le prétraitement doit-il faire partie du pipeline ?
2. Que fait `ColumnTransformer` ?
3. Que représente l'écart-type des scores de validation croisée ?
4. Combien d'entraînements pour une grille de 4 × 3 valeurs avec `cv=5` (sans compter le réentraînement final) ?
5. Pourquoi ne jamais charger un `.joblib` de source inconnue ?
6. À quoi sert `handle_unknown="ignore"` ?

Réponses : [corrections.md](corrections.md), section « Leçon 5.6 ».

## Résumé

- Le pipeline (prétraitement + modèle) apprend tout sur l'entraînement et s'applique tel quel aux nouvelles données.
- Validation croisée : moyenne et dispersion ; choix des hyperparamètres par recherche sur l'entraînement uniquement.
- Le test sert une fois, à la fin. Le modèle se sauvegarde avec ses métadonnées, et ne se recharge que depuis une source sûre.

## Pour aller plus loin

- Exercices **I07** et **A05**.
- Projet 6 (`projects/06_end_to_end_ml_project`).
- Module suivant : [Deep Learning](../06_deep_learning/README.md).
