# Leçon 4.2 — Distributions, boîtes à moustaches, corrélations et cartes de chaleur

## Prérequis

- Leçon 4.1, leçon 2.3 (quantiles, corrélation).

## Objectifs d'apprentissage

- Lire et tracer une boîte à moustaches (*box plot*) et comparer des groupes.
- Tracer des distributions lissées (densité, *KDE*) et des diagrammes en violon avec Seaborn.
- Construire et lire une matrice de corrélation sous forme de carte de chaleur (*heatmap*).
- Choisir une palette de couleurs adaptée : séquentielle (intensité) ou divergente (signe).

## Concepts clés

- **Boîte à moustaches** : boîte de Q1 à Q3, trait à la médiane, moustaches jusqu'aux valeurs les plus extrêmes à moins de 1,5·IQR, points au-delà (valeurs suspectes).
- **Estimation de densité par noyau** (*Kernel Density Estimation*, KDE) : version lissée d'un histogramme.
- **Diagramme en violon** (*violin plot*) : densité dessinée symétriquement, par groupe.
- **Fonction de répartition empirique** (*ECDF*) : proportion des valeurs inférieures ou égales à x ; aucune classe à choisir.
- **Carte de chaleur** (*heatmap*) : tableau dont chaque cellule est colorée selon sa valeur.
- **Palette séquentielle** : une teinte, du clair au foncé, pour une intensité. **Palette divergente** : deux teintes opposées et un milieu neutre, pour des valeurs signées.

## Explication détaillée

### Boîtes à moustaches

La boîte à moustaches résume une distribution en cinq nombres et montre les valeurs suspectes. Son grand avantage : **comparer de nombreux groupes côte à côte** (notes par groupe, prix par quartier). Sa limite : elle cache la forme (deux pics ne se voient pas). On la complète par les points eux-mêmes (*strip plot*) quand l'effectif est faible, ou par un violon.

### Distributions lissées et ECDF

Un histogramme dépend du choix des classes ; une KDE dépend d'une **largeur de bande** (*bandwidth*) : trop petite, la courbe est hérissée ; trop grande, les détails disparaissent. Pour comparer **plusieurs distributions**, superposer des KDE ou des ECDF est souvent plus lisible que superposer des histogrammes. L'ECDF se lit directement : « 60 % des maisons du centre coûtent moins de 400 k€ ».

### Seaborn

Seaborn est construit sur Matplotlib et travaille directement avec des DataFrames : `sns.boxplot(data=df, x="group", y="final_score")`. Il calcule lui-même les agrégations (moyennes, intervalles de confiance, densités). Chaque fonction Seaborn accepte `ax=` : on reste dans l'interface orientée objet. On peut toujours finir la mise en forme avec Matplotlib.

### Matrices de corrélation et cartes de chaleur

`df.corr()` produit une matrice symétrique de corrélations. En carte de chaleur, on repère d'un coup d'œil les variables liées à la cible et les variables redondantes entre elles. Les corrélations vont de −1 à 1 avec un **zéro qui a un sens** : il faut une palette **divergente** (bleu pour négatif, rouge pour positif, gris neutre pour 0) et une échelle **centrée** (`vmin=-1, vmax=1`). Une palette arc-en-ciel ou une palette séquentielle trompe : un 0,0 et un −0,8 ne doivent pas avoir l'air « proches » parce qu'ils sont de couleurs voisines. Affichez les valeurs dans les cellules (`annot=True`) et masquez le triangle redondant.

Pour une intensité sans signe (nombre de ventes par jour et par heure), on utilise au contraire une palette **séquentielle** d'une seule teinte.

## Applications concrètes

- Comparer la distribution des salaires par service avec des boîtes à moustaches.
- Sélectionner des variables : celles corrélées à la cible, et éliminer les doublons très corrélés entre eux.
- Calendrier des ventes : carte de chaleur jour de la semaine × mois.

## Exemples de code Python

### Exemple 1 — boîtes à moustaches et violons

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lab.plotting import CATEGORICAL, apply_style, save_figure

apply_style()
houses = pd.read_csv("datasets/generated/house_prices.csv")
order = ["Centre", "Residentiel", "Peripherie", "Rural"]
houses["price_per_m2"] = houses["price_keur"] * 1000 / houses["surface_m2"]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
sns.boxplot(data=houses, x="neighborhood", y="price_per_m2", order=order, color=CATEGORICAL[0],
            width=0.5, fliersize=3, ax=axes[0])
axes[0].set_title("Boîtes à moustaches")
sns.violinplot(data=houses, x="neighborhood", y="price_per_m2", order=order, color=CATEGORICAL[0],
               inner="quartile", cut=0, ax=axes[1])
axes[1].set_title("Violons (quartiles en pointillés)")
for ax in axes:
    ax.set_xlabel("")
axes[0].set_ylabel("Prix au m² (€)")
fig.suptitle("Le prix au m² diminue du centre vers la campagne", fontweight="bold")
save_figure(fig, "06_price_per_m2_box_violin.png")
summary = houses.groupby("neighborhood")["price_per_m2"].describe()[["25%", "50%", "75%"]]
print(summary.reindex(order).round(0).to_string())
```

### Explication du code

- Une seule couleur pour toutes les boîtes : les quartiers sont identifiés par l'axe x, la couleur n'ajouterait rien.
- `inner="quartile"` trace médiane et quartiles dans les violons ; `cut=0` empêche la densité de déborder au-delà des valeurs observées.
- Le tableau des quartiles confirme ce que montre le graphique : chaque quartier a un prix médian au m² nettement différent, et les boîtes ne se chevauchent même pas (le premier quartile du centre dépasse le troisième quartile du résidentiel). Seules les moustaches et quelques points se recouvrent.

### Sortie attendue

```text output
                 25%     50%     75%
neighborhood                        
Centre        4061.0  4333.0  4647.0
Residentiel   3092.0  3351.0  3610.0
Peripherie    2322.0  2529.0  2726.0
Rural         1632.0  1806.0  1960.0
```

### Exemple 2 — KDE et ECDF pour comparer des distributions

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lab.plotting import CATEGORICAL, apply_style, save_figure

apply_style()
students = pd.read_csv("datasets/generated/students.csv")
students["résultat"] = students["passed"].map({1: "admis", 0: "non admis"})
palette = {"admis": CATEGORICAL[0], "non admis": CATEGORICAL[1]}

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
sns.kdeplot(data=students, x="study_hours_week", hue="résultat", palette=palette, fill=True,
            alpha=0.25, common_norm=False, ax=axes[0])
axes[0].set_title("Densité des heures d'étude")
sns.ecdfplot(data=students, x="study_hours_week", hue="résultat", palette=palette, ax=axes[1])
axes[1].axvline(12, color="#8a8a85", linestyle="--", linewidth=1)
axes[1].set_title("Fonction de répartition empirique")
for ax in axes:
    ax.set_xlabel("Heures d'étude par semaine")
save_figure(fig, "07_study_hours_kde_ecdf.png")

share_below_12 = students.groupby("résultat")["study_hours_week"].apply(lambda s: (s <= 12).mean())
print("share studying <= 12 h/week:", share_below_12.round(2).to_dict())
```

### Explication du code

- `hue=` sépare les distributions selon le résultat ; `common_norm=False` normalise chaque courbe séparément (sinon le groupe le plus nombreux paraîtrait « plus grand »).
- L'ECDF permet une lecture chiffrée : à l'abscisse 12 h, on lit la proportion de chaque groupe qui étudie au plus 12 heures. Le calcul imprimé confirme la lecture.
- Couleurs fixes par catégorie (`palette=` en dictionnaire) : « admis » garde la même couleur dans toutes les figures du cours.

### Sortie attendue

```text output
share studying <= 12 h/week: {'admis': 0.61, 'non admis': 0.78}
```

### Exemple 3 — matrice de corrélation et carte de chaleur

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lab.plotting import apply_style, diverging_cmap, save_figure, sequential_cmap

apply_style()
students = pd.read_csv("datasets/generated/students.csv")
cols = ["study_hours_week", "attendance_rate", "sleep_hours", "previous_score", "math", "physics", "french", "final_score"]
corr = students[cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(corr, mask=mask, cmap=diverging_cmap(), vmin=-1, vmax=1, center=0, annot=True, fmt=".2f",
            annot_kws={"size": 7}, linewidths=1, linecolor="white", cbar_kws={"label": "corrélation de Pearson"}, ax=ax)
ax.set_title("Les notes sont liées entre elles ; le sommeil l'est peu")
ax.grid(False)
save_figure(fig, "08_correlation_heatmap.png")

sales = pd.read_csv("datasets/generated/sales.csv", parse_dates=["date"])
calendar = pd.crosstab(sales["date"].dt.day_of_week, sales["date"].dt.month)
calendar.index = ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"]
fig, ax = plt.subplots(figsize=(8, 3))
sns.heatmap(calendar, cmap=sequential_cmap(), linewidths=1, linecolor="white", cbar_kws={"label": "lignes de vente"}, ax=ax)
ax.set_xlabel("Mois")
ax.set_ylabel("")
ax.set_title("Plus de ventes en fin d'année, moins le week-end")
ax.grid(False)
save_figure(fig, "09_sales_calendar_heatmap.png")
print(corr["final_score"].drop("final_score").sort_values(ascending=False).round(2).to_string())
```

### Explication du code

- `np.triu(..., k=1)` masque le triangle supérieur (redondant, la matrice étant symétrique).
- La palette divergente (bleu – gris – rouge) avec `center=0` et l'échelle fixée de −1 à 1 rend le zéro neutre et le signe immédiat.
- Les notes de matières sont fortement corrélées à la note finale (normal : elle en est calculée ; ce serait une **fuite** si l'on voulait prédire la note finale à partir d'elles).
- La seconde carte compte les lignes de vente par jour de la semaine et par mois : intensité sans signe, donc palette **séquentielle** à une teinte.

### Sortie attendue

```text output
math                0.93
physics             0.86
french              0.80
previous_score      0.59
study_hours_week    0.34
attendance_rate     0.26
sleep_hours         0.07
```

## Erreurs fréquentes

- **Palette arc-en-ciel (`jet`) pour une carte de chaleur** : non perceptuellement uniforme, illisible pour les daltoniens.
- **Échelle de corrélation non centrée** : un 0 apparaît coloré comme un effet.
- **Superposer des histogrammes de groupes d'effectifs différents** sans normaliser.
- **Boîte à moustaches sur 5 valeurs** : montrez plutôt les points.
- **Conclure à partir d'une carte de corrélation** qu'une variable est inutile : elle peut avoir un effet non linéaire ou conditionnel.
- **Annoter avec trop de décimales** : deux suffisent.

## Exercices guidés

1. Tracez les boîtes à moustaches de `final_score` par `group` avec les points individuels superposés (`sns.stripplot`).
2. Tracez la KDE de `annual_income_keur` de `customers.csv`. Combien de « bosses » voyez-vous ? Qu'est-ce que cela suggère ?
3. Construisez la carte de chaleur des corrélations de `house_prices.csv` (variables numériques seulement).
4. Comparez avec une ECDF la distribution de `avg_basket_eur` des clients de moins et de plus de 50 ans.

Corrigés : [corrections.md](corrections.md), section « Leçon 4.2 ».

## Défis autonomes

- Tracez `sns.pairplot` sur 4 variables de `customers.csv` : quels regroupements naturels apparaissent ? (On les retrouvera avec k-means au module 5.)
- Faites varier `bw_adjust` (0,3 ; 1 ; 3) dans `sns.kdeplot` et décrivez l'effet.

## Quiz

1. Que représentent la boîte, le trait central et les moustaches d'un box plot ?
2. Quel avantage a l'ECDF sur l'histogramme ?
3. Pourquoi une palette divergente pour des corrélations ?
4. Que fait `common_norm=False` dans `sns.kdeplot` ?
5. Pourquoi masquer un triangle de la matrice de corrélation ?
6. Quand préférer un violon à une boîte à moustaches ?

Réponses : [corrections.md](corrections.md), section « Leçon 4.2 ».

## Résumé

- Boîtes à moustaches pour comparer de nombreux groupes ; violons, KDE et ECDF pour la forme.
- Seaborn agrège à partir de DataFrames ; on garde le contrôle avec `ax=`.
- Carte de chaleur : séquentielle pour une intensité, divergente et centrée pour des valeurs signées.

## Pour aller plus loin

- Leçon suivante : [Choisir un graphique et communiquer un résultat](03_choisir_communiquer.md).
