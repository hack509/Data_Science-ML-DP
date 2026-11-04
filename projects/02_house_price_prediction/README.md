# Projet 2 — Prédiction de prix immobiliers

**Difficulté** : ★★☆☆ — **Modules** : 3, 5, 10 — **Durée indicative** : 5 à 8 heures

## Énoncé du problème

Une agence souhaite un outil qui estime le prix de vente d'un logement à partir de ses caractéristiques (surface, pièces, âge, distance au centre, quartier, jardin, classe énergétique), pour préparer ses premières estimations. L'outil doit être **plus précis qu'une estimation naïve**, indiquer **l'incertitude** de sa performance, et **signaler** les logements trop différents de ceux qu'il connaît.

## Objectifs d'apprentissage

- Construire un pipeline de prétraitement complet (`ColumnTransformer`) sans fuite (leçon 5.6).
- Comparer plusieurs familles de modèles par validation croisée et régler leurs hyperparamètres (leçons 5.2, 5.4, 5.6).
- Transformer la cible (logarithme) quand les erreurs sont multiplicatives (leçon 5.2).
- Évaluer une seule fois sur le jeu de test, avec intervalle de confiance et analyse par quartier (leçon 10.1).
- Sauvegarder le modèle avec ses métadonnées et prédire sur un nouveau fichier avec avertissements (leçon 10.2).

## Données

`datasets/generated/house_prices.csv` : 800 logements **synthétiques**.

| Colonne | Type | Description |
|---|---|---|
| `surface_m2` | réel | surface habitable |
| `rooms` | entier | nombre de pièces |
| `age_years` | entier | âge du bâtiment |
| `distance_center_km` | réel | distance au centre-ville |
| `neighborhood` | catégorie | Centre, Residentiel, Peripherie, Rural |
| `has_garden` | 0/1 | présence d'un jardin |
| `energy_class` | catégorie ordonnée | A (meilleure) à G |
| `price_keur` | réel | **cible** : prix en milliers d'euros |

Le générateur (`scripts/generate_datasets.py`) calcule le prix comme un produit de facteurs (prix au m² du quartier × surface × effets de l'âge, de l'énergie et de la distance) avec un bruit multiplicatif : c'est pourquoi modéliser log(prix) fonctionne bien. **Les résultats ne disent rien du marché immobilier réel.**

## Architecture

```text
02_house_price_prediction/
├── run.py                       entraîner (défaut) ou prédire (sous-commande predict)
├── house_prices/modeling.py     prétraitement, candidats, sélection, évaluation, sauvegarde, prédiction
├── tests/test_project.py
└── outputs/                     model.joblib, metadata.json, metrics.json
```

## Étapes

1. **Découpage** 80/20 (`random_state=0`). Le jeu de test est mis de côté jusqu'à la fin.
2. **Prétraitement** : imputation + standardisation (numériques), one-hot avec `handle_unknown="ignore"` (quartier), encodage ordinal (classe énergétique).
3. **Référence** : un `DummyRegressor` (médiane).
4. **Candidats** : Ridge sur log(prix), forêt aléatoire, gradient boosting sur log(prix) (`TransformedTargetRegressor`).
5. **Sélection** par `GridSearchCV` (validation croisée à 5 plis, MAE), en notant moyenne et écart-type.
6. **Évaluation finale** sur le test : MAE avec intervalle bootstrap, RMSE, R², erreur relative par quartier, comparaison à la référence.
7. **Sauvegarde** : pipeline, métadonnées (colonnes, plages, versions), métriques.
8. **Prédiction** sur un nouveau CSV, avec avertissements pour les valeurs hors plage.

## Lancer la solution de référence

```bash
python projects/02_house_price_prediction/run.py                 # about 20 s on CPU
python projects/02_house_price_prediction/run.py --quick         # smaller grids
python projects/02_house_price_prediction/run.py predict mes_maisons.csv predictions.csv
python -m pytest projects/02_house_price_prediction -q
```

Le fichier d'entrée de `predict` doit contenir les 7 colonnes de variables.

## Indicateurs et résultats attendus

Résultats de la configuration complète sur la machine de test :

| Modèle | MAE validation croisée (k€) |
|---|---|
| gradient boosting (log) | ≈ 24,9 ± 3,0 |
| forêt aléatoire | ≈ 26,9 ± 3,4 |
| Ridge (log) | ≈ 30,9 ± 4,6 |

Modèle retenu : gradient boosting ; **MAE de test ≈ 25,8 k€** (IC 95 % ≈ [21 ; 31]), R² ≈ 0,92, contre une MAE de ≈ 112,5 k€ pour la référence. Les écarts entre candidats sont du même ordre que leurs écarts-types : le choix est raisonnable mais pas spectaculaire.

## Défis pour aller plus loin

- Ajoutez des variables dérivées (surface par pièce, interaction surface × quartier) dans le pipeline et mesurez l'effet en validation croisée.
- Comparez la sélection par MAE et par RMSE : le modèle retenu change-t-il ?
- Estimez un **intervalle de prédiction** pour chaque maison (par exemple avec `HistGradientBoostingRegressor(loss="quantile")` pour les quantiles 10 % et 90 %).
- Exposez la prédiction via un petit serveur local (`http.server`) qui reçoit du JSON.

## Solution de référence

`house_prices/modeling.py` : `load_and_split`, `preprocessor`, `candidates`, `select_model`, `evaluate`, `save`, `predict`, `run`.

## Pistes d'amélioration

- Validation croisée imbriquée pour une estimation non biaisée de la procédure de sélection elle-même.
- Analyse des résidus par tranche de prix et par classe énergétique.
- Fiche modèle (leçon 10.2) générée automatiquement à partir de `metrics.json`.
