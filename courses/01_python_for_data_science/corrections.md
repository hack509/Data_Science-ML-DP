# Corrigés — Module 1 : Python pour la Data Science

Ce fichier contient les réponses aux quiz et les corrigés des exercices guidés. Essayez toujours de répondre **avant** de lire. Le code de ce fichier est exécuté par les tests automatiques (`tests/test_corrections_code.py`).

## Leçon 1.1

### Quiz

1. `type(3 / 2)` est `float` (1.5) : `/` est toujours une division réelle. `type(3 // 2)` est `int` (1) : `//` est la division entière.
2. Un **ensemble** (`set`) : il élimine les doublons et le test `x in s` est quasi instantané.
3. Les réels sont stockés en binaire et 0,1 n'a pas de représentation binaire exacte ; la somme vaut `0.30000000000000004`.
4. Avec `x = 0`, `if x:` est faux (0 est une valeur « fausse ») alors que `if x is not None:` est vrai. Pour une note, seul le second test est correct.
5. Un tuple (de valeurs hachables) peut être une clé ; une liste non, car elle est modifiable (non hachable).
6. `0` : la valeur par défaut, car la clé `"b"` est absente.

### Exercices guidés

```python
# 1. Conversion with filtering of invalid values
raw = ["12", "15.5", "abs", "9"]
scores = [float(v) for v in raw if v.replace(".", "", 1).isdigit()]
print(scores)

# 2. Mean with a loop
values = [12, 15.5, 9, 18]
total = 0
for v in values:
    total += v
print(total / len(values))

# 3. Unique sorted ids
ids = ["S01", "S02", "S01", "S03", "S02"]
print(sorted(set(ids)))

# 4. Letter frequencies
counts = {}
for letter in "statistiques":
    counts[letter] = counts.get(letter, 0) + 1
most_common = max(counts, key=counts.get)
print(counts, "->", most_common)
```

```text output
[12.0, 15.5, 9.0]
13.625
['S01', 'S02', 'S03']
{'s': 3, 't': 3, 'a': 1, 'i': 2, 'q': 1, 'u': 1, 'e': 1} -> s
```

Remarque sur l'exercice 4 : « s » et « t » apparaissent chacun 3 fois ; `max` renvoie la première clé rencontrée parmi les ex aequo. Pour obtenir toutes les lettres les plus fréquentes : `[k for k, v in counts.items() if v == max(counts.values())]`.

## Leçon 1.2

### Quiz

1. `None`.
2. La liste par défaut est créée **une seule fois**, à la définition de la fonction, et partagée entre tous les appels : elle accumule les valeurs d'un appel à l'autre.
3. `r = [w.lower() for w in mots]`.
4. À exécuter du code uniquement quand le fichier est lancé comme script, et pas quand il est importé comme module.
5. Le fichier est fermé automatiquement à la sortie du bloc, même si une exception survient.
6. `ValueError` pour `float("abc")` ; `KeyError` pour `{}["x"]`.

### Exercices guidés

```python
from pathlib import Path


def clean_price(text: str) -> float:
    """Convert '19,90 EUR', ' 24.9 ' or '189' to a float."""
    cleaned = text.strip().removesuffix("EUR").strip().replace(",", ".")
    return float(cleaned)


print([clean_price(t) for t in ["19,90 EUR", " 24.9 ", "189"]])

products = ["Souris", "Clavier"]
prices = [24.9, 39.9]
print({product: price for product, price in zip(products, prices)})


def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        print("Warning: division by zero")
        return None


print(safe_divide(10, 4), safe_divide(1, 0))

notes_path = Path("outputs/module01/notes.txt")
notes_path.parent.mkdir(parents=True, exist_ok=True)
notes_path.write_text("premiere ligne\n\ntroisieme ligne\n", encoding="utf-8")
with notes_path.open(encoding="utf-8") as handle:
    non_empty = sum(1 for line in handle if line.strip())
print("non-empty lines:", non_empty)
```

```text output
[19.9, 24.9, 189.0]
{'Souris': 24.9, 'Clavier': 39.9}
Warning: division by zero
2.5 None
non-empty lines: 2
```

## Leçon 1.3

### Quiz

1. La classe est le plan (le type) ; une instance est un objet concret construit à partir de ce plan, avec ses propres valeurs d'attributs.
2. `self` désigne l'objet sur lequel la méthode est appelée ; il donne accès à ses attributs.
3. `__init__`, `__repr__` et `__eq__` (et les comparaisons d'ordre avec `order=True`).
4. Pour offrir une **interface commune** : on peut remplacer un modèle par un autre sans modifier le reste du code (polymorphisme).
5. Seulement l'installation des paquets (`pip install`). Créer et activer l'environnement se fait hors ligne.
6. `sys.prefix != sys.base_prefix`.

### Exercices guidés

```python
from collections import Counter
from dataclasses import dataclass, field


class Gradebook:
    def __init__(self, student_id):
        self.student_id = student_id
        self.scores = {}

    def add_score(self, subject, score):
        if not 0 <= score <= 20:
            raise ValueError(f"invalid score {score}")
        self.scores[subject] = score

    def best_subject(self):
        return max(self.scores, key=self.scores.get)


book = Gradebook("S01")
book.add_score("math", 14)
book.add_score("french", 16)
print(book.best_subject())


@dataclass
class Student:
    student_id: str
    group: str
    scores: dict = field(default_factory=dict)

    @property
    def average(self) -> float:
        return sum(self.scores.values()) / len(self.scores)


print(Student("S02", "A", {"math": 12, "physics": 15}).average)


class MostFrequentClassifier:
    def fit(self, X, y):
        self.most_frequent_ = Counter(y).most_common(1)[0][0]
        return self

    def predict(self, X):
        return [self.most_frequent_] * len(X)


print(MostFrequentClassifier().fit([[1], [2], [3]], ["ham", "spam", "ham"]).predict([[4], [5]]))
```

```text output
french
13.5
['ham', 'ham']
```

## Leçon 1.4

### Quiz

1. `(4,)` : une moyenne par colonne.
2. Les données sont stockées de façon contiguë avec un type unique, et les boucles sont exécutées en code compilé (C), sans l'interprétation Python élément par élément.
3. Un tableau 1D (copie) contenant les éléments strictement positifs.
4. Une **vue** : elle partage la mémoire avec `a`.
5. `(5, 3) + (3,)` : oui (la ligne est répétée). `(5, 3) + (5,)` : non, les dimensions de droite (3 et 5) diffèrent ; il faut `(5, 1)`, obtenu avec `b[:, None]`.
6. À rendre les tirages reproductibles : même graine, mêmes nombres.

### Exercices guidés

```python
import numpy as np

celsius = np.linspace(12, 30, 10)
print((celsius * 9 / 5 + 32).round(1))

rng = np.random.default_rng(seed=0)
grades = np.round(rng.normal(loc=11, scale=3, size=(6, 3)), 1).clip(0, 20)
print("below 10 per subject:", (grades < 10).sum(axis=0))
centered = grades - grades.mean(axis=1, keepdims=True)
print("row means after centering:", centered.mean(axis=1).round(6))

rng = np.random.default_rng(seed=1)
dice = rng.integers(1, 7, size=(10_000, 2))
print("P(sum == 7) ~", (dice.sum(axis=1) == 7).mean())
```

```text output
[53.6 57.2 60.8 64.4 68.  71.6 75.2 78.8 82.4 86. ]
below 10 per subject: [3 3 2]
row means after centering: [ 0. -0.  0.  0.  0. -0.]
P(sum == 7) ~ 0.1652
```

## Leçon 1.5

### Quiz

1. Une `Series` est une colonne unique (valeurs + index) ; un `DataFrame` est une table de plusieurs `Series` partageant le même index.
2. `df[["a"]]` renvoie un `DataFrame` d'une colonne ; `df["a"]` renvoie une `Series`.
3. Sinon l'index est écrit comme une colonne supplémentaire, qui réapparaît sous le nom `Unnamed: 0` à la relecture.
4. `df.loc[0:2]` renvoie les étiquettes 0, 1 **et 2** (3 lignes) ; `df.iloc[0:2]` les positions 0 et 1 (2 lignes).
5. `record_path` désigne la liste imbriquée à déplier en lignes ; `meta` liste les champs des niveaux parents à recopier sur chaque ligne.
6. `df.groupby("group")["passed"].mean()` : la moyenne d'une variable 0/1 est la proportion de 1.

### Exercices guidés

```python
import json

import pandas as pd

sales = pd.read_csv("datasets/generated/sales.csv")
sales["revenue"] = sales["quantity"] * sales["unit_price"]
print("total revenue:", round(sales["revenue"].sum(), 2))
print(sales.groupby("region")["revenue"].sum().sort_values(ascending=False).round(2).to_string())

sales["date"] = pd.to_datetime(sales["date"])
monthly = sales.groupby(sales["date"].dt.month)["revenue"].sum().round(0)
print(monthly.to_string())

with open("datasets/raw/orders.json", encoding="utf-8") as handle:
    customers = json.load(handle)
orders_per_customer = pd.Series({c["customer_id"]: len(c["orders"]) for c in customers})
print("customers without orders:", int((orders_per_customer == 0).sum()))
print(orders_per_customer.head().to_string())
```

```text output
total revenue: 531305.9
region
Nord     153939.00
Sud      134403.01
Ouest    128111.64
Est      114852.25
date
1     39385.0
2     34913.0
3     43530.0
4     35890.0
5     31857.0
6     31357.0
7     42019.0
8     37404.0
9     37825.0
10    44168.0
11    72967.0
12    79989.0
customers without orders: 14
C001    0
C002    2
C003    1
C004    3
C005    0
```
