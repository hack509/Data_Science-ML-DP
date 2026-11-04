# Corrigés — Module 10 : Projets, évaluation, déploiement et IA responsable

## Leçon 10.1

### Quiz

1. Pour que le choix reflète le besoin réel (le coût des erreurs) et non les résultats obtenus ; et pour savoir si le modèle apporte quelque chose par rapport à une solution simple.
2. Une métrique qui ne doit pas se dégrader pendant qu'on optimise la métrique principale : temps de réponse, taux de faux positifs, performance minimale dans chaque sous-groupe.
3. On tire avec remise de nombreux échantillons de la taille du jeu de test, on recalcule la métrique sur chacun, et l'on prend les percentiles 2,5 % et 97,5 % des valeurs obtenues.
4. Les deux modèles sont évalués sur les mêmes tirages : la variabilité due à l'échantillon affecte les deux de la même façon et se compense dans la différence.
5. Un score global peut cacher des performances très inégales selon les groupes ; certains utilisateurs peuvent être mal servis.
6. Graines aléatoires fixées, chemins relatifs, configuration centralisée, étapes scriptées (pas de manipulation manuelle), résultats écrits dans des fichiers, tests automatiques, versions des bibliothèques consignées.

### Exercices guidés

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

houses = pd.read_csv("datasets/generated/house_prices.csv")
X, y = houses.drop(columns="price_keur"), houses["price_keur"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
numeric = ["surface_m2", "rooms", "age_years", "distance_center_km", "has_garden"]
prep = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("hood", OneHotEncoder(handle_unknown="ignore"), ["neighborhood"]),
    ("energy", OrdinalEncoder(categories=[list("ABCDEFG")]), ["energy_class"]),
])
linear = Pipeline([("prep", prep), ("reg", LinearRegression())]).fit(X_train, np.log(y_train))
pred = np.exp(linear.predict(X_test))
y_arr = y_test.to_numpy()
rng = np.random.default_rng(0)
r2_values = [r2_score(y_arr[i], pred[i]) for i in rng.integers(0, len(y_arr), size=(1000, len(y_arr)))]
print("R2 =", round(r2_score(y_arr, pred), 3), "| 95% CI =", np.percentile(r2_values, [2.5, 97.5]).round(3))

evaluation = X_test.assign(price=y_arr, abs_error=np.abs(y_arr - pred))
evaluation["relative_error"] = evaluation["abs_error"] / evaluation["price"]
print(evaluation.groupby("energy_class").agg(n=("price", "size"), relative_error=("relative_error", "mean")).round(3).sort_values("relative_error", ascending=False).to_string())

forest = Pipeline([("prep", prep), ("reg", RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1))]).fit(X_train, y_train)
pred_forest = forest.predict(X_test)
for n_boot in (200, 2000):
    rng = np.random.default_rng(1)
    diffs = [mean_absolute_error(y_arr[i], pred_forest[i]) - mean_absolute_error(y_arr[i], pred[i]) for i in rng.integers(0, len(y_arr), size=(n_boot, len(y_arr)))]
    print(f"n_boot={n_boot}: 95% CI of the difference = {np.percentile(diffs, [2.5, 97.5]).round(2)}")
```

```text output
R2 = 0.949 | 95% CI = [0.924 0.965]
               n  relative_error
energy_class                    
F             20           0.122
E             28           0.108
A              8           0.100
B             23           0.097
C             28           0.096
D             39           0.077
G             14           0.055
n_boot=200: 95% CI of the difference = [-3.18  4.81]
n_boot=2000: 95% CI of the difference = [-3.45  5.06]
```

Exercice 2 : comparez l'erreur relative en tenant compte de l'effectif : les classes extrêmes (A, G) comptent peu de maisons de test, leurs moyennes sont donc très incertaines. Exercice 3 : avec plus de tirages, les bornes de l'intervalle sont estimées plus précisément, mais l'intervalle ne rétrécit pas : sa largeur dépend de la taille du jeu de test, pas du nombre de tirages. Exercice 4 : métrique principale : précision de la classe « spam » à rappel au moins égal à une valeur cible (ou F1 si l'on veut un compromis simple) ; garde-fous : taux de faux positifs sur les messages légitimes (aucun message important ne doit être perdu) et performance stable sur les messages au ton commercial légitime.

## Leçon 10.2

### Quiz

1. Le pipeline complet (prétraitement + modèle), ses métadonnées (version, colonnes et plages attendues, catégories connues, versions des bibliothèques, métriques, données d'entraînement) et le code qui le charge et l'utilise.
2. Parce qu'un modèle produit une réponse pour n'importe quelle entrée, même absurde ou hors de son domaine de validité ; seule la validation permet de refuser ou de signaler ces cas.
3. La dérive des données : la distribution des entrées change. La dérive de concept : la relation entre entrées et cible change (à entrées identiques, la bonne réponse n'est plus la même).
4. L'écart entre la distribution de référence et la distribution récente d'une variable, calculé par classes. Usuellement : < 0,1 stable, 0,1 à 0,25 à surveiller, > 0,25 dérive importante.
5. Parce que d'autres variables peuvent servir de substitut (proxy) à la variable sensible, et parce que les données elles-mêmes peuvent refléter des biais historiques ou des mesures moins fiables pour certains groupes.
6. L'usage prévu et les usages à éviter, les données d'entraînement, les performances globales et par groupe, les limites connues, les considérations éthiques et les recommandations, la version et la date.

### Exercices guidés

```python
import numpy as np
import pandas as pd

houses = pd.read_csv("datasets/generated/house_prices.csv")


def missing_values_warnings(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [f"row {i}: missing {c}" for c in columns for i in frame.index[frame[c].isna()]]


sample = pd.DataFrame({"surface_m2": [70, np.nan], "rooms": [3, 2]})
print(missing_values_warnings(sample, ["surface_m2", "rooms"]))


def psi(reference, recent, bins=10):
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.clip(np.histogram(reference, edges)[0] / len(reference), 1e-6, None)
    new = np.clip(np.histogram(recent, edges)[0] / len(recent), 1e-6, None)
    return float(np.sum((new - ref) * np.log(new / ref)))


centre = houses.loc[houses["neighborhood"] == "Centre", "age_years"].to_numpy()
rural = houses.loc[houses["neighborhood"] == "Rural", "age_years"].to_numpy()
print("PSI age_years, centre vs rural:", round(psi(centre, rural), 3))
```

```text output
['row 1: missing surface_m2']
PSI age_years, centre vs rural: 0.098
```

Exercice 2 (script de lot, à enregistrer par exemple dans `outputs/batch_predict.py`) :

```python no-run
import sys
from pathlib import Path

import pandas as pd

# PricePredictor is the class from example 1 of the lesson (copy it or import it from your own module).
predictor = PricePredictor(Path("outputs/module10/price_model"))
frame = pd.read_csv(sys.argv[1])
predictions, warnings = predictor.predict(frame)
predictions.to_csv(sys.argv[2], index=False)
Path(sys.argv[2]).with_suffix(".warnings.txt").write_text("\n".join(warnings), encoding="utf-8")
print(f"{len(predictions)} predictions, {len(warnings)} warnings")
```

Exercice 3 : dans le générateur, l'âge ne dépend pas du quartier : le PSI doit rester faible, et toute valeur modérée s'explique par la taille réduite des échantillons (en particulier pour le rural). Exercice 4 : si la mesure est aussi fiable pour les deux groupes, les écarts de précision et de taux d'erreurs entre groupes deviennent faibles (dus au seul hasard de l'échantillon) ; cela confirme que, dans ce scénario, la disparité venait de la qualité de la mesure et non du nombre d'exemples ou d'une différence réelle entre groupes.
