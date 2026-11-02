# Leçon 1.5 — pandas : DataFrames, CSV et JSON

## Prérequis

- Leçon 1.4 (NumPy).
- Jeux de données générés : `python scripts/generate_datasets.py`.

## Objectifs d'apprentissage

- Comprendre les structures `Series` et `DataFrame`.
- Lire et écrire des fichiers CSV et JSON locaux.
- Sélectionner des lignes et des colonnes avec `[]`, `.loc` et `.iloc`.
- Créer des colonnes calculées, filtrer, trier et agréger avec `groupby`.
- Aplatir des données JSON imbriquées avec `pd.json_normalize`.

## Concepts clés

- **Series** : colonne étiquetée (un tableau NumPy + un index).
- **DataFrame** : table de données ; un ensemble de `Series` partageant le même index.
- **Index** : les étiquettes des lignes (par défaut 0, 1, 2...).
- **dtype** : type de chaque colonne (`int64`, `float64`, `str`, `datetime64`, `category`, `bool`).
- **`.loc` / `.iloc`** : sélection par **étiquettes** / par **positions**.
- **groupby** : découper la table en groupes, appliquer un calcul, combiner les résultats (*split-apply-combine*).
- **JSON** (*JavaScript Object Notation*) : format texte de données imbriquées (dictionnaires et listes).

## Explication détaillée

### Series et DataFrame

Un `DataFrame` ressemble à une feuille de tableur : des lignes (observations) et des colonnes (variables). Chaque colonne est une `Series` avec son propre type. Contrairement à un tableau NumPy 2D, les colonnes peuvent avoir des types différents (texte, nombres, dates).

Premiers réflexes après un chargement :

- `df.shape` : nombre de lignes et de colonnes ;
- `df.head()` / `df.tail()` : premières et dernières lignes ;
- `df.info()` : types et nombre de valeurs non manquantes ;
- `df.describe()` : statistiques des colonnes numériques.

### Lire et écrire

`pd.read_csv("chemin.csv")` lit un CSV ; les options utiles sont `sep=";"` (séparateur), `decimal=","` (décimales à la française), `parse_dates=["date"]`, `dtype={...}`, `na_values=["absent"]`. `df.to_csv("sortie.csv", index=False)` écrit un CSV (sans la colonne d'index, sinon elle réapparaît à la relecture sous le nom `Unnamed: 0`). Pour le JSON : `pd.read_json`, `df.to_json(orient="records", force_ascii=False)` ou le module standard `json` pour des structures imbriquées.

### Sélectionner

- `df["math"]` : une colonne (Series) ; `df[["math", "physics"]]` : plusieurs colonnes (DataFrame).
- `df.loc[ligne, colonne]` : par étiquettes, bornes **incluses** ; accepte des masques booléens : `df.loc[df["math"] > 15, ["student_id", "math"]]`.
- `df.iloc[0:5, 2]` : par positions, borne de fin exclue comme en Python.
- `df.query("math > 15 and group == 'A'")` : syntaxe textuelle pratique pour les filtres.

Pour **modifier** des valeurs, utilisez toujours `df.loc[masque, "colonne"] = valeur`. L'écriture en chaîne `df[masque]["colonne"] = valeur` ne modifie pas `df` (depuis pandas 3, le mode *copy-on-write* la rend sans effet).

### Transformer et agréger

- Colonnes calculées : `df["average"] = df[["math", "physics", "french"]].mean(axis=1)`.
- Tri : `df.sort_values("average", ascending=False)`.
- Agrégation : `df.groupby("group")["math"].mean()` ou, pour plusieurs statistiques nommées : `df.groupby("group").agg(n=("student_id", "count"), mean_math=("math", "mean"))`.
- Comptage : `df["group"].value_counts()`.

### Données JSON imbriquées

Les API et les applications produisent souvent du JSON imbriqué : un client contient une liste de commandes, chaque commande une liste d'articles. `pd.json_normalize(data, record_path=[...], meta=[...])` « déplie » ces listes en une table plate, une ligne par article, en recopiant les informations du parent (`meta`).

## Applications concrètes

- Charger l'export CSV d'un logiciel de caisse et calculer le chiffre d'affaires par région.
- Transformer un export JSON de commandes en table pour l'analyser.
- Préparer une table propre (une ligne = une observation, une colonne = une variable) avant le Machine Learning.

## Exemples de code Python

### Exemple 1 — charger et explorer les notes des étudiants

```python
import pandas as pd

students = pd.read_csv("datasets/generated/students.csv")
print(students.shape)
print(students.head(3).to_string())
print(students.dtypes.astype(str).value_counts().to_dict())
print(students[["math", "physics", "french"]].describe().round(2).to_string())
```

### Explication du code

`to_string()` force l'affichage complet du tableau (sans colonnes masquées). `dtypes.value_counts()` résume combien de colonnes ont chaque type : les identifiants et le groupe sont du texte (`str`), les notes des réels. `describe()` donne effectif, moyenne, écart-type, minimum, quartiles et maximum de chaque colonne numérique.

### Sortie attendue

```text output
(300, 11)
  student_id group  study_hours_week  attendance_rate  sleep_hours  previous_score  math  physics  french  final_score  passed
0      S0001     B               7.1            0.726          6.6            11.5   7.7      9.0     8.5         8.33       0
1      S0002     A               4.7            0.885          7.4            12.4  11.0     12.9    11.2        11.63       1
2      S0003     A              15.3            0.901          7.7             8.4  10.3     10.9    12.9        11.26       1
{'float64': 8, 'str': 2, 'int64': 1}
         math  physics  french
count  300.00   300.00  300.00
mean     9.62     9.59    9.70
std      3.46     2.92    2.74
min      1.50     1.60    2.30
25%      7.40     7.50    7.78
50%      9.30     9.75    9.75
75%     11.92    11.60   11.40
max     20.00    17.50   16.80
```

### Exemple 2 — sélectionner, créer des colonnes, agréger

```python
import pandas as pd

students = pd.read_csv("datasets/generated/students.csv")

top = students.loc[students["math"] >= 16, ["student_id", "group", "math"]]
print(f"{len(top)} students have at least 16 in math")
print(top.sort_values("math", ascending=False).head(3).to_string(index=False))

students["average"] = students[["math", "physics", "french"]].mean(axis=1).round(2)
students["mention"] = pd.cut(
    students["average"], bins=[0, 10, 12, 14, 16, 20],
    labels=["ajourné", "passable", "assez bien", "bien", "très bien"], right=False,
)
print(students["mention"].value_counts().sort_index().to_string())

summary = students.groupby("group").agg(
    n=("student_id", "count"),
    mean_average=("average", "mean"),
    pass_rate=("passed", "mean"),
    mean_study_hours=("study_hours_week", "mean"),
)
print(summary.round(2).to_string())
```

### Explication du code

- Le masque `students["math"] >= 16` sélectionne les lignes ; la liste de colonnes limite l'affichage.
- `pd.cut` découpe une variable continue en intervalles étiquetés ; `right=False` signifie que 12 appartient à « assez bien » ([12, 14[).
- `groupby(...).agg(...)` produit une ligne par groupe avec des colonnes nommées. La moyenne de `passed` (0 ou 1) est un **taux** de réussite.

### Sortie attendue

```text output
12 students have at least 16 in math
student_id group  math
     S0081     C  20.0
     S0103     B  19.4
     S0070     B  19.1
mention
ajourné       168
passable       75
assez bien     42
bien           11
très bien       4
         n  mean_average  pass_rate  mean_study_hours
group                                                
A       88          9.44       0.40             10.66
B      118          9.40       0.39              9.79
C       94         10.11       0.54              9.94
```

### Exemple 3 — JSON imbriqué vers table plate, puis sauvegarde

```python
import json
from pathlib import Path

import pandas as pd

with open("datasets/raw/orders.json", encoding="utf-8") as handle:
    customers = json.load(handle)
print("customers:", len(customers), "| keys:", list(customers[0]))

items = pd.json_normalize(
    customers,
    record_path=["orders", "items"],
    meta=["customer_id", "city", ["orders", "order_id"], ["orders", "date"]],
)
items = items.rename(columns={"orders.order_id": "order_id", "orders.date": "date"})
items["line_total"] = items["quantity"] * items["unit_price"]
print(items.head(4).to_string(index=False))

revenue_by_city = items.groupby("city")["line_total"].sum().sort_values(ascending=False).round(2)
print(revenue_by_city.to_string())

output_dir = Path("outputs/module01")
output_dir.mkdir(parents=True, exist_ok=True)
items.to_csv(output_dir / "order_items.csv", index=False)
revenue_by_city.reset_index().to_json(output_dir / "revenue_by_city.json", orient="records", force_ascii=False, indent=2)
print(pd.read_csv(output_dir / "order_items.csv").shape)
```

### Explication du code

- Le fichier contient une liste de clients ; chaque client a une liste `orders`, chaque commande une liste `items`.
- `record_path=["orders", "items"]` indique le chemin jusqu'à la liste à déplier ; `meta` liste les champs des niveaux supérieurs à recopier sur chaque ligne (`["orders", "order_id"]` désigne un champ de la commande).
- Les clients sans commande ne produisent aucune ligne : c'est normal mais à garder en tête (les compter à part si besoin).
- On relit le CSV écrit pour vérifier l'aller-retour.

### Sortie attendue

```text output
customers: 40 | keys: ['customer_id', 'city', 'member_since', 'newsletter', 'orders']
product  quantity  unit_price customer_id  city order_id       date  line_total
 Casque         3        59.9        C002 Paris    W0001 2024-08-26       179.7
 Webcam         3        69.0        C002 Paris    W0002 2024-03-18       207.0
 Casque         3        59.9        C002 Paris    W0002 2024-03-18       179.7
  Ecran         3       189.0        C002 Paris    W0002 2024-03-18       567.0
city
Bordeaux     4574.1
Lyon         2871.7
Lille        2587.0
Rennes       2502.6
Paris        2083.4
Marseille    1104.8
Nantes        409.1
(124, 8)
```

## Erreurs fréquentes

- **`KeyError: 'Math'`** : les noms de colonnes sont sensibles à la casse et aux espaces. Vérifiez `df.columns`.
- **Écriture en chaîne** `df[df.x > 0]["y"] = 1` : sans effet. Utilisez `df.loc[df.x > 0, "y"] = 1`.
- **Oublier `index=False`** dans `to_csv` : une colonne `Unnamed: 0` apparaît à la relecture.
- **Nombres lus comme du texte** (`dtype` `str`) à cause d'une valeur comme `"absent"` ou `"19,90"` : voir l'exercice B03 et le module 3.
- **Confondre `.loc` (étiquettes, fin incluse) et `.iloc` (positions, fin exclue)**.
- **Boucler sur les lignes** avec `iterrows()` pour un calcul : presque toujours remplaçable par une opération vectorisée sur les colonnes.

## Exercices guidés

1. Chargez `datasets/generated/sales.csv`, ajoutez une colonne `revenue = quantity * unit_price` et affichez le chiffre d'affaires total arrondi.
2. Calculez le chiffre d'affaires par `region` et triez-le par ordre décroissant.
3. Convertissez la colonne `date` en dates (`pd.to_datetime`) et calculez le chiffre d'affaires par mois (`df["date"].dt.month`).
4. À partir de `orders.json`, comptez le nombre de commandes par client, **en incluant les clients sans commande** (*indice : `len(c["orders"])` dans une compréhension*).

Corrigés : [corrections.md](corrections.md), section « Leçon 1.5 ».

## Défis autonomes

- Trouvez, pour chaque groupe d'étudiants, l'étudiant ayant la meilleure moyenne (*piste : `idxmax` dans un `groupby`*).
- Exportez le résumé par groupe de l'exemple 2 en JSON lisible (accents conservés) et relisez-le avec `json.load`.
- Comparez la vitesse de `iterrows()` et d'une opération vectorisée pour calculer `revenue` sur `sales.csv`.

## Quiz

1. Quelle est la différence entre une `Series` et un `DataFrame` ?
2. Que renvoie `df[["a"]]` par rapport à `df["a"]` ?
3. Pourquoi faut-il écrire `index=False` dans `to_csv` ?
4. Quelle différence entre `df.loc[0:2]` et `df.iloc[0:2]` sur un index par défaut ?
5. Que fait `pd.json_normalize` avec `record_path` et `meta` ?
6. Comment calculer le taux de réussite par groupe si `passed` vaut 0 ou 1 ?

Réponses : [corrections.md](corrections.md), section « Leçon 1.5 ».

## Résumé

- `DataFrame` = table de `Series` typées partageant un index.
- `read_csv` / `to_csv(index=False)` / `json.load` / `json_normalize` couvrent la plupart des imports et exports locaux.
- Sélection par étiquettes avec `.loc`, par positions avec `.iloc` ; modification toujours via `.loc`.
- `groupby(...).agg(...)` est l'outil central de l'analyse agrégée.

## Pour aller plus loin

- Exercice **B03** — nettoyage d'un CSV (`exercises/beginner/B03_clean_csv`).
- Exercice **B04** — import et transformation de JSON (`exercises/beginner/B04_json_transform`).
- Exercice **I01** — script de traitement réutilisable.
- Module 3 : analyse et nettoyage de données en profondeur.
