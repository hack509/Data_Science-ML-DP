# Corrigés — Module 3 : Analyse de données

## Leçon 3.1

### Quiz

1. Pour pouvoir toujours revenir à la source, refaire le nettoyage autrement et garantir la reproductibilité.
2. Elle devient `NaN` (au lieu de lever une erreur).
3. MCAR : la donnée manque au hasard (un fichier corrompu au hasard). MNAR : le fait de manquer dépend de la valeur elle-même (les personnes aux revenus élevés ne déclarent pas leur revenu).
4. Pour garder l'information « cette valeur était manquante », qui peut être utile à l'analyse ou à un modèle, et pour savoir quelles valeurs sont inventées.
5. Sinon les lignes dupliquées comptent plusieurs fois dans les moyennes, sommes et effectifs.
6. Elle n'est pas influencée par les valeurs extrêmes et convient aux distributions asymétriques.

### Exercices guidés

```python
import numpy as np
import pandas as pd

raw = pd.read_csv("datasets/raw/students_raw.csv")
print("rows with at least one missing value:", round(raw.isna().any(axis=1).mean(), 3))

sleep = raw.drop_duplicates()["sleep_hours"]
imputed = sleep.fillna(sleep.median())
print(f"mean {sleep.mean():.3f} -> {imputed.mean():.3f} | std {sleep.std():.3f} -> {imputed.std():.3f}")

clean = raw.drop_duplicates()
print("unique ids after cleaning:", clean["student_id"].is_unique)


def normalize_category(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip().str.upper()
    return cleaned.replace("", pd.NA)


print(normalize_category(pd.Series([" a ", "b", "", None, "C"])).tolist())
```

```text output
rows with at least one missing value: 0.127
mean 7.193 -> 7.198 | std 0.920 -> 0.897
unique ids after cleaning: True
['A', 'B', <NA>, <NA>, 'C']
```

Exercice 2 : la moyenne change à peine, mais l'écart-type **diminue** : on a ajouté des valeurs toutes égales au centre de la distribution. C'est l'effet secondaire classique de l'imputation par une constante.

## Leçon 3.2

### Quiz

1. Une erreur (valeur impossible) doit être corrigée ou mise à `NaN` ; une valeur extrême réelle est une information, à conserver (éventuellement écrêtée ou transformée selon l'objectif).
2. Elle repose sur les quartiles, qui ne sont pas influencés par les valeurs extrêmes, alors que la moyenne et l'écart-type du z-score le sont.
3. Quand les catégories ont un ordre naturel (classes énergétiques, niveaux de satisfaction).
4. Parce que les distances seraient dominées par les variables de grande échelle.
5. Le prix au m² pour prédire le prix ; la variable `passed` pour prédire `final_score`.
6. Découpe x en 4 classes de même effectif (quartiles).

### Exercices guidés

```python
import pandas as pd

students = pd.read_csv("datasets/generated/students.csv")
hours = students["study_hours_week"]
q1, q3 = hours.quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
print(f"upper Tukey bound: {upper:.2f} h -> flagged:", int((hours > upper).sum()), "| max:", hours.max())

students["progress"] = students["final_score"] - students["previous_score"]
print(students.groupby("group")["progress"].mean().round(2).to_dict())

sales = pd.read_csv("datasets/generated/sales.csv", parse_dates=["date"])
sales["month"] = sales["date"].dt.month
sales["day_of_week"] = sales["date"].dt.day_of_week
sales["is_weekend"] = sales["day_of_week"] >= 5
print("weekend share of lines:", round(sales["is_weekend"].mean(), 3))

dummies = pd.get_dummies(sales["product"], prefix="product", dtype=int)
print(dummies.columns.tolist(), "| every row sums to 1:", bool((dummies.sum(axis=1) == 1).all()))
```

```text output
upper Tukey bound: 22.74 h -> flagged: 8 | max: 30.0
{'A': 0.12, 'B': -0.78, 'C': -0.24}
weekend share of lines: 0.223
['product_Casque', 'product_Clavier', 'product_Ecran', 'product_Souris', 'product_Webcam'] | every row sums to 1: True
```

Exercice 1 : les étudiants signalés travaillent entre 23 et 30 heures par semaine. C'est élevé mais plausible : on les conserve.

## Leçon 3.3

### Quiz

1. `agg` renvoie une ligne par groupe ; `transform` renvoie une valeur par ligne d'origine (le résultat du groupe recopié sur chaque ligne).
2. À ajouter une ligne et une colonne de totaux.
3. Agrège une série indexée par des dates en sommes mensuelles (étiquetées au premier jour du mois).
4. Pour juger de la fiabilité d'une moyenne : avec peu d'observations, elle peut varier fortement par hasard.
5. Vue d'ensemble, analyse univariée, bivariée, multivariée, synthèse.
6. Une association peut venir d'une variable de confusion, du hasard ou d'une causalité inverse.

### Exercices guidés

```python
import pandas as pd

sales = pd.read_csv("datasets/generated/sales.csv", parse_dates=["date"])
sales["revenue"] = sales["quantity"] * sales["unit_price"]
print(sales.groupby("category").agg(revenue=("revenue", "sum"), lines=("order_id", "size")).sort_values("revenue", ascending=False).round(0).to_string())

students = pd.read_csv("datasets/generated/students.csv")
students["hours_level"] = (students["study_hours_week"] > students["study_hours_week"].median()).map({True: "au-dessus", False: "en dessous"})
print(students.pivot_table(index="group", columns="hours_level", values="passed", aggfunc="mean").round(2).to_string())

monthly = sales.groupby(["region", sales["date"].dt.month])["revenue"].sum()
best_month = monthly.groupby(level="region").idxmax().map(lambda key: key[1])
print("best month per region:", best_month.to_dict())

customers = pd.read_csv("datasets/generated/customers.csv")
customers["age_band"] = pd.cut(customers["age"], bins=[17, 30, 45, 60, 90])
customers["basket_gap"] = customers["avg_basket_eur"] - customers.groupby("age_band", observed=True)["avg_basket_eur"].transform("median")
print(customers[["customer_id", "age", "avg_basket_eur", "basket_gap"]].head(3).to_string(index=False))
```

```text output
                revenue  lines
category                      
Affichage      208977.0    374
Peripheriques  148036.0   1641
Audio          102381.0    610
Video           71912.0    375
hours_level  au-dessus  en dessous
group                             
A                 0.45        0.33
B                 0.47        0.31
C                 0.60        0.50
best month per region: {'Est': 12, 'Nord': 12, 'Ouest': 11, 'Sud': 12}
customer_id  age  avg_basket_eur  basket_gap
      K0001   57           266.3       143.0
      K0002   58            25.4       -97.9
      K0003   36            49.5       -44.6
```

## Leçon 3.4

### Quiz

1. `concat` empile des tables (mêmes colonnes) ; `merge` associe les lignes de deux tables par une clé.
2. Les lignes de gauche correspondantes sont dupliquées : le nombre de lignes augmente et les agrégats sont faussés.
3. `validate` vérifie la cardinalité attendue et lève une erreur sinon ; `indicator` ajoute une colonne `_merge` qui indique l'origine de chaque ligne.
4. `inner`.
5. Complétude, unicité, validité, cohérence, intégrité référentielle.
6. Pour les exécuter automatiquement et à chaque nouvelle livraison de données, de façon reproductible.

### Exercices guidés

```python
import pandas as pd

students = pd.read_csv("datasets/generated/students.csv")
activities = pd.read_csv("datasets/generated/student_activities.csv")
merged = students.merge(activities, on="student_id", how="left", validate="one_to_one")
print("club 'aucun':", int((merged["club"] == "aucun").sum()), "| no activity row:", int(merged["club"].isna().sum()))

inner = students.merge(activities, on="student_id", how="inner")
print(inner.groupby("group")["volunteer_hours_year"].mean().round(2).to_dict())

print("sleep in [3, 12]:", bool(students["sleep_hours"].between(3, 12).all()))

sales = pd.read_csv("datasets/generated/sales.csv")
doubled = pd.concat([sales, sales], ignore_index=True)
print("order_id unique after concat:", doubled["order_id"].is_unique, "| duplicated ids:", int(doubled["order_id"].duplicated().sum()))
```

```text output
club 'aucun': 61 | no activity row: 50
{'A': 15.51, 'B': 14.59, 'C': 13.39}
sleep in [3, 12]: True
order_id unique after concat: False | duplicated ids: 3000
```
