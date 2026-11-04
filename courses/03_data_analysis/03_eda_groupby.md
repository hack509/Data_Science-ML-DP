# Leçon 3.3 — Analyse exploratoire, agrégations et statistiques descriptives

## Prérequis

- Leçons 3.1 et 3.2, leçon 2.3.

## Objectifs d'apprentissage

- Mener une analyse exploratoire (EDA) structurée, guidée par des questions.
- Maîtriser `groupby`, `agg`, `transform`, `pivot_table` et `crosstab`.
- Analyser une série temporelle simple (tendance, saisonnalité) avec `resample`.
- Formuler des constats chiffrés prudents, sans confondre corrélation et causalité.

## Concepts clés

- **Analyse exploratoire des données** (*Exploratory Data Analysis*, EDA) : explorer les données pour comprendre leur structure, générer des hypothèses et détecter des problèmes, avant toute modélisation.
- **Analyse univariée / bivariée / multivariée** : une variable seule / deux variables / plusieurs à la fois.
- **Split-apply-combine** : découper en groupes, appliquer un calcul, combiner les résultats.
- **Table pivot** (*pivot table*) : tableau croisé d'agrégats (lignes × colonnes).
- **Tableau de contingence** (*contingency table*, `crosstab`) : effectifs croisés de deux variables catégorielles.
- **Rééchantillonnage temporel** (*resampling*) : agréger une série temporelle par jour, semaine, mois.

## Explication détaillée

### Une EDA guidée par des questions

Une EDA efficace n'est pas une accumulation de graphiques : c'est une enquête. On part de questions métier (« quels produits rapportent le plus ? », « les ventes sont-elles saisonnières ? », « qu'est-ce qui distingue les étudiants qui réussissent ? »), on les traduit en calculs, et chaque réponse suscite de nouvelles questions.

Plan type :

1. **Vue d'ensemble** : taille, période couverte, variables, qualité (leçon 3.1).
2. **Univarié** : distribution de chaque variable importante (centre, dispersion, forme, catégories rares).
3. **Bivarié** : relation de chaque variable avec la variable d'intérêt (taux de réussite par groupe, prix selon le quartier).
4. **Multivarié** : interactions (l'effet des heures d'étude est-il le même dans tous les groupes ?).
5. **Synthèse** : 3 à 5 constats chiffrés, leurs limites, les questions ouvertes.

### Les outils d'agrégation de pandas

- `df.groupby("col")["x"].mean()` : une statistique par groupe.
- `df.groupby(["a", "b"]).agg(n=("x", "size"), moyenne=("x", "mean"))` : plusieurs statistiques nommées, plusieurs clés.
- `transform` : renvoie un résultat **de la même taille** que le DataFrame (par exemple, la moyenne du groupe sur chaque ligne), utile pour comparer chaque observation à son groupe.
- `pivot_table(index=..., columns=..., values=..., aggfunc=...)` : tableau croisé.
- `pd.crosstab(a, b, normalize="index")` : proportions croisées.

### Séries temporelles

Avec une colonne de dates convertie (`pd.to_datetime`) et placée en index, `resample("MS").sum()` agrège par mois (*Month Start*), `resample("W")` par semaine. On repère la **tendance** (évolution de long terme), la **saisonnalité** (motif qui se répète : week-ends, fin d'année) et les **anomalies**. Une moyenne mobile (`rolling(7).mean()`) lisse le bruit journalier.

### Rester prudent

Un écart entre groupes peut être dû au hasard (leçon 2.4), à une variable de confusion, ou à un effectif trop faible. Indiquez toujours les **effectifs** à côté des moyennes, et n'écrivez pas « X cause Y » à partir d'une simple agrégation.

## Applications concrètes

- Tableau de bord des ventes par région, produit et mois.
- Analyse des facteurs associés à la réussite scolaire.
- Préparation d'un projet de ML : comprendre la cible, repérer les variables prometteuses et les problèmes potentiels (déséquilibre de classes, fuite).

## Exemples de code Python

### Exemple 1 — analyse univariée et bivariée des étudiants

```python
import pandas as pd

students = pd.read_csv("datasets/generated/students.csv")
print("pass rate:", round(students["passed"].mean(), 3), "| n =", len(students))

students["study_level"] = pd.qcut(students["study_hours_week"], q=4, labels=["faible", "moyen", "élevé", "très élevé"])
by_level = students.groupby("study_level", observed=True).agg(
    n=("student_id", "size"),
    hours_min=("study_hours_week", "min"),
    hours_max=("study_hours_week", "max"),
    pass_rate=("passed", "mean"),
    mean_final=("final_score", "mean"),
)
print(by_level.round(2).to_string())

students["gap_to_group_mean"] = students["final_score"] - students.groupby("group")["final_score"].transform("mean")
print(students[["student_id", "group", "final_score", "gap_to_group_mean"]].head(4).round(2).to_string(index=False))
```

### Explication du code

- Le taux de réussite global sert de référence.
- `qcut` crée 4 niveaux d'heures d'étude de même effectif ; `observed=True` n'affiche que les catégories présentes.
- Le tableau montre une relation **non régulière** : le taux de réussite reste autour de 38 % pour les trois premiers quarts, puis bondit à 62 % pour le quart qui étudie le plus (13 h et plus). Une seule moyenne globale ou une corrélation aurait caché cette forme.
- C'est une **association** : dans de vraies données, il faudrait se demander si d'autres facteurs (motivation, niveau initial) expliquent à la fois les heures et la réussite.
- `transform("mean")` place la moyenne du groupe sur chaque ligne, ce qui permet de calculer l'écart de chaque étudiant à son groupe.

### Sortie attendue

```text output
pass rate: 0.44 | n = 300
              n  hours_min  hours_max  pass_rate  mean_final
study_level                                                 
faible       75        1.5        6.1       0.37        8.73
moyen        76        6.2        9.3       0.39        9.61
élevé        76        9.4       12.8       0.38        9.16
très élevé   73       13.0       30.0       0.62       11.09
student_id group  final_score  gap_to_group_mean
     S0001     B         8.33              -1.08
     S0002     A        11.63               2.21
     S0003     A        11.26               1.84
     S0004     B        10.85               1.44
```

### Exemple 2 — tables pivots et tableaux de contingence sur les ventes

```python
import pandas as pd

sales = pd.read_csv("datasets/generated/sales.csv", parse_dates=["date"])
sales["revenue"] = sales["quantity"] * sales["unit_price"]

pivot = sales.pivot_table(index="product", columns="region", values="revenue", aggfunc="sum", margins=True, margins_name="Total")
print(pivot.round(0).to_string())

share = pd.crosstab(sales["region"], sales["product"], normalize="index")
print((share * 100).round(1).to_string())

discounted = sales["unit_price"] < sales.groupby("product")["unit_price"].transform("max")
print("share of discounted lines:", round(discounted.mean(), 3))
```

### Explication du code

- `pivot_table` croise produits (lignes) et régions (colonnes) ; `margins=True` ajoute les totaux.
- `crosstab(..., normalize="index")` donne, pour chaque région, la répartition des lignes de vente entre produits (chaque ligne somme à 100 %). Les répartitions sont très proches d'une région à l'autre : les clients achètent les mêmes produits partout.
- Une ligne est « remisée » si son prix est inférieur au prix maximal (prix catalogue) du produit : on compare chaque ligne à une statistique de son groupe, grâce à `transform`.

### Sortie attendue

```text output
region        Est      Nord     Ouest       Sud     Total
product                                                  
Casque    22798.0   30854.0   20809.0   27919.0  102381.0
Clavier   17376.0   29933.0   19363.0   20584.0   87257.0
Ecran     48724.0   53997.0   55850.0   50406.0  208977.0
Souris    11339.0   17771.0   15460.0   16207.0   60778.0
Webcam    14614.0   21383.0   16629.0   19286.0   71912.0
Total    114852.0  153939.0  128112.0  134403.0  531306.0
product  Casque  Clavier  Ecran  Souris  Webcam
region                                         
Est        21.3     25.2   13.9    26.8    12.8
Nord       19.7     28.0   11.2    28.6    12.5
Ouest      18.0     24.9   13.6    31.2    12.2
Sud        22.6     23.4   11.7    29.8    12.5
share of discounted lines: 0.388
```

### Exemple 3 — série temporelle : tendance et saisonnalité

```python
import pandas as pd

sales = pd.read_csv("datasets/generated/sales.csv", parse_dates=["date"])
sales["revenue"] = sales["quantity"] * sales["unit_price"]
daily = sales.set_index("date")["revenue"].resample("D").sum()

monthly = daily.resample("MS").sum()
print(monthly.round(0).to_string())
print("Nov-Dec share of yearly revenue:", round(monthly.iloc[10:].sum() / monthly.sum(), 3))

weekday_mean = daily.groupby(daily.index.day_of_week).mean()
print("mean daily revenue Mon..Sun:", weekday_mean.round(0).tolist())
smooth = daily.rolling(window=7, center=True).mean()
print("7-day moving average on 2024-12-15:", round(smooth.loc["2024-12-15"], 1))
```

### Explication du code

- `resample("D")` crée une série journalière (les jours sans vente valent 0) ; `resample("MS")` agrège par mois.
- Novembre et décembre pèsent beaucoup plus que 2/12 ≈ 17 % du chiffre d'affaires : saisonnalité de fin d'année (elle a été introduite volontairement dans le générateur).
- La moyenne par jour de la semaine (0 = lundi) révèle un creux le week-end.
- La moyenne mobile sur 7 jours lisse les variations journalières.

### Sortie attendue

```text output
date
2024-01-01    39385.0
2024-02-01    34913.0
2024-03-01    43530.0
2024-04-01    35890.0
2024-05-01    31857.0
2024-06-01    31357.0
2024-07-01    42019.0
2024-08-01    37404.0
2024-09-01    37825.0
2024-10-01    44168.0
2024-11-01    72967.0
2024-12-01    79989.0
Freq: MS
Nov-Dec share of yearly revenue: 0.288
mean daily revenue Mon..Sun: [1510.0, 1543.0, 1715.0, 1404.0, 1711.0, 1034.0, 1241.0]
7-day moving average on 2024-12-15: 2431.0
```

## Erreurs fréquentes

- **Montrer des moyennes sans les effectifs** : une moyenne sur 3 observations n'a pas la même valeur qu'une moyenne sur 300.
- **Oublier `observed=True`** avec des catégories : des groupes vides apparaissent.
- **Agréger des dates non converties** (texte) : tri alphabétique et `resample` impossible.
- **Conclure à une causalité** à partir d'un `groupby`.
- **Explorer sans question** : on se perd dans des dizaines de tableaux sans conclusion.

## Exercices guidés

1. Calculez le chiffre d'affaires et le nombre de lignes par catégorie de produit (`category`), triés par chiffre d'affaires.
2. Construisez la table pivot du taux de réussite par `group` (lignes) et par niveau d'heures d'étude (colonnes, en 2 classes : au-dessus / en dessous de la médiane).
3. Quel est le mois de plus fort chiffre d'affaires pour chaque région ? (*indice : `groupby(["region", month])` puis `idxmax`*)
4. Pour chaque client de `customers.csv`, calculez l'écart de son panier moyen à la médiane de sa tranche d'âge (`pd.cut` sur l'âge).

Corrigés : [corrections.md](corrections.md), section « Leçon 3.3 ».

## Défis autonomes

- Rédigez une synthèse de 5 constats chiffrés sur `sales.csv`, chacun avec sa limite.
- Le prix moyen payé pour un même produit varie-t-il selon le mois ? Que révèle cette analyse sur la politique de remises du générateur ?

## Quiz

1. Quelle différence entre `agg` et `transform` dans un `groupby` ?
2. À quoi sert `margins=True` dans une table pivot ?
3. Que fait `resample("MS").sum()` ?
4. Pourquoi faut-il afficher les effectifs des groupes ?
5. Citez les cinq étapes d'une EDA structurée.
6. Pourquoi un `groupby` ne suffit-il pas à prouver un effet causal ?

Réponses : [corrections.md](corrections.md), section « Leçon 3.3 ».

## Résumé

- Une EDA est une enquête guidée par des questions, de l'univarié au multivarié, conclue par des constats chiffrés.
- `groupby` + `agg` résume, `transform` compare chaque ligne à son groupe, `pivot_table` et `crosstab` croisent.
- `resample` et `rolling` révèlent tendance, saisonnalité et anomalies.
- Effectifs, incertitude et prudence causale accompagnent chaque conclusion.

## Pour aller plus loin

- Exercice **I03** — fusion et agrégations.
- Module 4 : représenter graphiquement ces analyses.
- Leçon suivante : [Fusion de tables et contrôles de qualité](04_fusion_qualite.md).
