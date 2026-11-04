# Leçon 5.2 — Régression : régression linéaire et métriques

## Prérequis

- Leçon 5.1 ; leçon 2.2 (moindres carrés) ; leçon 3.2 (encodage one-hot).

## Objectifs d'apprentissage

- Entraîner une régression linéaire avec scikit-learn et interpréter ses coefficients.
- Calculer et interpréter MAE, MSE, RMSE et R².
- Comparer un modèle à une référence (*baseline*).
- Analyser les résidus pour repérer ce que le modèle ne capture pas.
- Connaître la régularisation (Ridge) et la régression par k plus proches voisins.

## Concepts clés

- **Régression linéaire** (*linear regression*) : prédiction par somme pondérée des variables plus une constante.
- **Coefficient** (*coefficient*, `coef_`) et **ordonnée à l'origine** (*intercept*, `intercept_`).
- **Résidu** (*residual*) : y − ŷ, l'erreur sur une observation.
- **MAE** (*Mean Absolute Error*), **MSE** (*Mean Squared Error*), **RMSE** (*Root MSE*), **R²** (*coefficient de détermination*).
- **Référence** (*baseline*) : modèle trivial (prédire la moyenne) que tout vrai modèle doit battre.
- **Régularisation** (*regularization*) : pénalité sur la taille des coefficients pour limiter le surapprentissage (Ridge = pénalité L2).

## Explication détaillée

### Fiche algorithme : régression linéaire

**1. Intuition.** On cherche la « meilleure droite » (ou le meilleur plan, en plusieurs dimensions) qui passe au milieu des points. Chaque variable reçoit un poids : « +1 m² ajoute environ 3 k€, toutes choses égales par ailleurs ».

**2. Idée mathématique.** ŷ = w₁x₁ + ... + wₚxₚ + b. Les poids minimisent la somme des carrés des résidus Σ(yᵢ − ŷᵢ)² ; la solution est celle des moindres carrés (leçon 2.2), calculée exactement par scikit-learn.

**3. Implémentation.** `sklearn.linear_model.LinearRegression` ; `fit(X, y)`, `predict(X)`, attributs `coef_` et `intercept_`.

**4. Données.** `datasets/generated/house_prices.csv` : 800 logements, variables numériques et catégorielles (quartier, classe énergétique), cible `price_keur`.

**5. Entraînement.** Encodage one-hot des catégories, découpage 80/20, `fit` sur l'entraînement uniquement.

**6. Évaluation.** MAE, RMSE et R² sur le jeu de test, comparés à la référence « prédire la moyenne ».

**7. Interprétation.** Chaque coefficient est l'effet d'une unité de la variable, **les autres variables étant fixées**. Les coefficients ne sont comparables entre eux que si les variables sont standardisées. Une association n'est pas une causalité.

**8. Erreurs fréquentes.** Voir la section dédiée plus bas (relations non linéaires, variables colinéaires, extrapolation).

**9-10. Exercices et corrigé.** Section « Exercices guidés », exercice I07, corrigés dans `corrections.md`.

### Les métriques de régression

| Métrique | Formule | Unité | Lecture |
|---|---|---|---|
| MAE | (1/n) Σ abs(yᵢ − ŷᵢ) | celle de y | erreur moyenne « typique », robuste aux grosses erreurs |
| MSE | (1/n) Σ (yᵢ − ŷᵢ)² | unité² | pénalise fortement les grosses erreurs ; c'est la perte optimisée |
| RMSE | √MSE | celle de y | comme la MSE mais lisible ; ≥ MAE |
| R² | 1 − MSE / variance(y) | sans unité | part de la variance expliquée ; 0 = pas mieux que la moyenne, 1 = parfait, peut être négatif |

Un R² se lit toujours **relativement** au problème : 0,6 peut être excellent pour prédire un comportement humain et médiocre pour une loi physique. La MAE est souvent la plus parlante pour un utilisateur : « en moyenne, on se trompe de 30 k€ ».

### Analyser les résidus

Un bon modèle laisse des résidus **sans structure** : centrés sur 0, sans tendance en fonction de la prédiction ou d'une variable. Si les résidus grandissent avec le prix (forme en entonnoir), l'erreur est relative plutôt qu'absolue : modéliser log(prix) peut aider. Si les résidus dessinent une courbe, une relation non linéaire n'est pas capturée.

### Régularisation : Ridge

Avec beaucoup de variables ou des variables corrélées, les coefficients peuvent devenir grands et instables. **Ridge** ajoute une pénalité α·Σwⱼ² à la perte : les coefficients sont « tirés » vers 0. α est un hyperparamètre (choisi par validation croisée). Les variables doivent être standardisées pour que la pénalité les traite équitablement. **Lasso** (pénalité Σ abs(wⱼ)) peut mettre des coefficients exactement à 0 et sert à la sélection de variables.

### Fiche rapide : k plus proches voisins en régression

**Intuition** : pour prédire le prix d'une maison, on regarde les k maisons les plus semblables de l'entraînement et on fait la moyenne de leurs prix. **Mathématiques** : distance euclidienne sur des variables **standardisées** ; k est l'hyperparamètre (petit k = variance élevée, grand k = biais élevé). **Implémentation** : `KNeighborsRegressor`. Avantage : capture des relations non linéaires sans hypothèse ; inconvénients : lent sur de gros jeux de données, sensible aux variables inutiles et à l'échelle. L'algorithme est détaillé en classification (leçon 5.3).

## Intuition mathématique

```text
ŷ = Xw + b                       R² = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²
Ridge : min Σ(yᵢ − ŷᵢ)² + α Σ wⱼ²   (α ≥ 0, α = 0 → régression ordinaire)
```

Le dénominateur de R² est l'erreur de la référence « prédire la moyenne ȳ » : R² mesure de combien le modèle fait mieux que cette référence.

## Applications concrètes

- Estimation de prix (immobilier, voitures d'occasion).
- Prévision de demande à partir de variables calendaires et de prix.
- Estimation de l'effet d'une variable « toutes choses égales par ailleurs » (avec prudence).

## Exemples de code Python

### Exemple 1 — entraîner, évaluer, comparer à la référence

```python
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

houses = pd.read_csv("datasets/generated/house_prices.csv")
X = pd.get_dummies(houses.drop(columns="price_keur"), columns=["neighborhood", "energy_class"], drop_first=True, dtype=float)
y = houses["price_keur"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


def evaluate(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{name:<18} MAE={mae:6.1f}  RMSE={rmse:6.1f}  R2={r2_score(y_test, pred):6.3f}")
    return model


evaluate("baseline (mean)", DummyRegressor(strategy="mean"))
linear = evaluate("linear regression", LinearRegression())
coefficients = pd.Series(linear.coef_, index=X.columns).round(2)
print(coefficients.to_string())
print("intercept:", round(linear.intercept_, 2))
```

### Explication du code

- `get_dummies(..., drop_first=True)` encode quartier et classe énergétique ; la première catégorie (Centre, classe A) sert de référence, ce qui évite une colonne redondante (colinéarité parfaite).
- La référence a un R² ≈ 0 (légèrement négatif sur le test) : c'est le niveau à battre.
- La régression linéaire explique une grande part de la variance. Lecture des coefficients : `surface_m2` ≈ +2,7 k€ par m² ; `neighborhood_Rural` très négatif : à surface égale, une maison rurale vaut beaucoup moins qu'une maison du centre (la référence). Les classes énergétiques E, F, G ont des coefficients négatifs par rapport à A, comme dans le générateur.

### Sortie attendue

```text output
baseline (mean)    MAE= 113.1  RMSE= 147.9  R2=-0.001
linear regression  MAE=  30.4  RMSE=  42.9  R2= 0.916
surface_m2                    2.69
rooms                         4.85
age_years                    -0.75
distance_center_km           -0.29
has_garden                   17.83
neighborhood_Peripherie    -176.04
neighborhood_Residentiel   -100.25
neighborhood_Rural         -244.32
energy_class_B                9.98
energy_class_C               -6.69
energy_class_D              -13.09
energy_class_E              -23.67
energy_class_F              -27.35
energy_class_G              -46.72
intercept: 171.97
```

### Exemple 2 — analyse des résidus et transformation logarithmique

```python
from pathlib import Path

import matplotlib.pyplot as plt

from lab.plotting import CATEGORICAL, apply_style

pred = linear.predict(X_test)
residuals = y_test - pred

log_model = LinearRegression().fit(X_train, np.log(y_train))
log_pred = np.exp(log_model.predict(X_test))
print(f"linear on price:      MAE={mean_absolute_error(y_test, pred):.1f}")
print(f"linear on log(price): MAE={mean_absolute_error(y_test, log_pred):.1f}")
print("residual std for cheap / expensive halves:",
      round(residuals[pred < np.median(pred)].std(), 1), "/", round(residuals[pred >= np.median(pred)].std(), 1))

apply_style()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].scatter(pred, residuals, s=10, alpha=0.6, color=CATEGORICAL[0])
axes[0].axhline(0, color="#2b2b29", linewidth=1)
axes[0].set_xlabel("Prix prédit (k€)")
axes[0].set_ylabel("Résidu (k€)")
axes[0].set_title("Les erreurs grandissent avec le prix")
axes[1].scatter(y_test, log_pred, s=10, alpha=0.6, color=CATEGORICAL[0])
axes[1].plot([0, 1100], [0, 1100], color="#8a8a85", linestyle="--")
axes[1].set_xlabel("Prix réel (k€)")
axes[1].set_ylabel("Prix prédit (k€)")
axes[1].set_title("Modèle sur log(prix) : prédit contre réel")
Path("outputs/module05").mkdir(parents=True, exist_ok=True)
fig.savefig("outputs/module05/regression_residuals.png", bbox_inches="tight")
plt.close(fig)
```

### Explication du code

- Cet exemple réutilise les variables de l'exemple 1 (même script).
- L'écart-type des résidus est plus grand pour les maisons chères : forme en **entonnoir**. Le générateur multiplie les effets (prix ∝ surface × facteurs) et ajoute un bruit multiplicatif : la relation est linéaire en **logarithme**, pas en prix.
- Le modèle entraîné sur log(prix), puis reconverti avec `np.exp`, obtient une MAE plus faible : comprendre la structure des erreurs a amélioré le modèle sans changer d'algorithme.

### Sortie attendue

```text output
linear on price:      MAE=30.4
linear on log(price): MAE=24.4
residual std for cheap / expensive halves: 33.4 / 50.2
```

### Exemple 3 — Ridge et k plus proches voisins

```python
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

for alpha in [0.1, 10, 1000]:
    evaluate(f"ridge alpha={alpha}", make_pipeline(StandardScaler(), Ridge(alpha=alpha)))
for k in [1, 10, 50]:
    evaluate(f"kNN k={k}", make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k)))
```

### Explication du code

- Avec un petit α, Ridge équivaut à la régression ordinaire ; avec α = 1000, les coefficients sont trop contraints et la performance baisse (sous-apprentissage). Ici, avec 800 exemples et peu de variables, la régularisation n'apporte rien : elle devient utile avec beaucoup de variables ou peu de données.
- k-NN : k = 1 surapprend (il recopie le voisin le plus proche, bruit compris) ; un k intermédiaire fait mieux ; un très grand k lisse trop. On retrouve le compromis biais-variance.
- Le `StandardScaler` dans le pipeline est indispensable pour k-NN (distances) et Ridge (pénalité équitable).
- Même avec le meilleur k, le k-NN reste nettement moins bon que la régression linéaire : la relation est ici presque linéaire (en logarithme), et les nombreuses colonnes binaires du one-hot pèsent autant que la surface dans le calcul des distances. Aucun algorithme n'est meilleur partout : il faut comparer.

### Sortie attendue

```text output
ridge alpha=0.1    MAE=  30.4  RMSE=  42.9  R2= 0.916
ridge alpha=10     MAE=  30.9  RMSE=  43.2  R2= 0.914
ridge alpha=1000   MAE=  62.7  RMSE=  87.0  R2= 0.654
kNN k=1            MAE=  53.0  RMSE=  69.1  R2= 0.782
kNN k=10           MAE=  47.0  RMSE=  62.9  R2= 0.819
kNN k=50           MAE=  71.1  RMSE=  93.0  R2= 0.604
```

## Erreurs fréquentes

- **Oublier la référence** : un R² de 0,7 ne veut rien dire sans point de comparaison.
- **Interpréter les coefficients comme des causes**, ou comparer des coefficients de variables non standardisées.
- **Encoder toutes les catégories sans `drop_first`** avec un modèle linéaire non régularisé : colinéarité parfaite (piège des variables indicatrices).
- **Extrapoler** : la droite n'a aucune garantie hors de la plage des données d'entraînement (une maison de 2000 m²).
- **Ignorer les résidus** : la structure des erreurs indique comment améliorer le modèle.
- **Calculer le R² sur l'entraînement** et le présenter comme performance.

## Exercices guidés

1. Calculez à la main (NumPy) MAE, RMSE et R² des prédictions de l'exemple 1 et comparez avec scikit-learn.
2. Entraînez une régression linéaire avec la seule variable `surface_m2`. Quelle est la perte de performance ?
3. Ajoutez la variable `log_distance = log1p(distance_center_km)` aux variables de l'exemple 1. Le modèle s'améliore-t-il ?
4. Sur `regression.csv`, comparez une régression linéaire et un k-NN (k = 10) avec standardisation. Expliquez la différence à l'aide de la formule du générateur (terme x3² et sin(x4)).

Corrigés : [corrections.md](corrections.md), section « Leçon 5.2 ».

## Défis autonomes

- Ajoutez une interaction surface × quartier (`surface_m2 * neighborhood_Rural`, etc.) et mesurez l'effet.
- Tracez la RMSE de validation d'un k-NN en fonction de k (1 à 60) : identifiez les zones de sur- et de sous-apprentissage.

## Quiz

1. Que représente un coefficient de régression linéaire ?
2. Quelle différence entre MAE et RMSE ? Laquelle est la plus sensible aux grosses erreurs ?
3. Que signifie un R² négatif sur le jeu de test ?
4. À quoi sert `drop_first=True` ?
5. Que montre un graphique des résidus en forme d'entonnoir ? Quel remède ?
6. Pourquoi standardiser avant Ridge et k-NN ?

Réponses : [corrections.md](corrections.md), section « Leçon 5.2 ».

## Résumé

- Régression linéaire : somme pondérée, coefficients interprétables « toutes choses égales par ailleurs ».
- MAE (lisible, robuste), RMSE (pénalise les grosses erreurs), R² (part de variance expliquée).
- Toujours comparer à une référence et analyser les résidus.
- Ridge régularise ; k-NN capture le non-linéaire ; les deux exigent des variables standardisées.

## Pour aller plus loin

- Exercice **I07** (`exercises/intermediate/I07_regression_pipeline`).
- Projet 2 (`projects/02_house_price_prediction`).
- Leçon suivante : [Classification](03_classification.md).
