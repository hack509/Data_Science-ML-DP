# Corrigés — Module 4 : Visualisation de données

Les figures produites par ce code sont enregistrées dans `outputs/module04/corrections/`.

## Leçon 4.1

### Quiz

1. La `Figure` est la feuille entière ; un `Axes` est une zone de tracé (avec ses axes x et y). Une figure peut contenir plusieurs `Axes`.
2. La longueur d'une barre encode sa valeur : couper l'axe déforme les rapports. Une courbe encode la valeur par la **position** ; on peut zoomer sur la plage utile si on l'indique clairement.
3. Des barres horizontales, triées par valeur (sauf ordre naturel des catégories).
4. Une série de petits graphiques identiques (mêmes axes), un par catégorie ; utile quand une seule figure aurait trop de séries.
5. À rendre les points semi-transparents pour voir les zones denses.
6. Pour libérer la mémoire, surtout dans une boucle ou un script qui produit beaucoup de figures.

### Exercices guidés

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.plotting import CATEGORICAL, apply_style, save_figure

apply_style()
folder = "outputs/module04/corrections"
sales = pd.read_csv("datasets/generated/sales.csv", parse_dates=["date"])

weekly = sales.set_index("date").resample("W")["order_id"].count()
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(weekly.index, weekly.values)
ax.set_ylim(bottom=0)
ax.set_title("Le nombre de ventes hebdomadaires double en fin d'année")
ax.set_ylabel("Lignes de vente par semaine")
save_figure(fig, "e1_weekly_lines.png", folder)

by_region = sales.groupby("region")["quantity"].sum().sort_values()
fig, ax = plt.subplots(figsize=(5, 2.5))
bars = ax.barh(by_region.index, by_region.values, color=CATEGORICAL[0])
ax.bar_label(bars, padding=3)
ax.set_title("Le Nord vend le plus d'unités")
save_figure(fig, "e2_quantity_by_region.png", folder)

customers = pd.read_csv("datasets/generated/customers.csv")
fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
axes[0].hist(customers["annual_income_keur"], bins=30, color=CATEGORICAL[0], edgecolor="white")
axes[0].set_title("Revenu (k€)")
axes[1].hist(np.log(customers["annual_income_keur"]), bins=30, color=CATEGORICAL[0], edgecolor="white")
axes[1].set_title("log(revenu)")
save_figure(fig, "e3_income_hist.png", folder)
print("skew income:", round(customers["annual_income_keur"].skew(), 2), "| skew log:", round(np.log(customers["annual_income_keur"]).skew(), 2))

students = pd.read_csv("datasets/generated/students.csv")
fig, ax = plt.subplots(figsize=(5, 3.5))
for color, (value, label) in zip(CATEGORICAL, [(1, "admis"), (0, "non admis")]):
    subset = students[students["passed"] == value]
    ax.scatter(subset["study_hours_week"], subset["final_score"], s=14, alpha=0.7, color=color, label=label)
ax.axhline(10, color="#8a8a85", linestyle="--", linewidth=1)
ax.set_xlabel("Heures d'étude par semaine")
ax.set_ylabel("Note finale")
ax.legend()
save_figure(fig, "e4_hours_vs_score.png", folder)
print("weekly lines, first and last weeks:", weekly.iloc[0], weekly.iloc[-2])
```

```text output
skew income: 1.36 | skew log: 0.54
weekly lines, first and last weeks: 42 95
```

Exercice 3 : la distribution du revenu est asymétrique vers la droite (quelques revenus élevés, segment « cadres premium ») ; son logarithme est plus symétrique.

## Leçon 4.2

### Quiz

1. La boîte va du premier au troisième quartile, le trait central est la médiane, les moustaches s'étendent jusqu'aux valeurs les plus extrêmes situées à moins de 1,5 IQR de la boîte ; les points au-delà sont des valeurs suspectes.
2. Aucun choix de classes ; lecture directe des proportions et des quantiles ; comparaison facile de plusieurs distributions.
3. Les corrélations ont un signe et un zéro neutre : il faut deux teintes opposées et un milieu neutre.
4. Chaque courbe est normalisée séparément : on compare les formes, pas les effectifs.
5. La matrice est symétrique : le triangle masqué répète l'autre.
6. Quand la forme compte (plusieurs pics, asymétrie) et que les effectifs sont suffisants.

### Exercices guidés

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lab.plotting import CATEGORICAL, apply_style, diverging_cmap, save_figure

apply_style()
folder = "outputs/module04/corrections"
students = pd.read_csv("datasets/generated/students.csv")
fig, ax = plt.subplots(figsize=(5, 3.5))
sns.boxplot(data=students, x="group", y="final_score", order=["A", "B", "C"], color="#cde2fb", showfliers=False, ax=ax)
sns.stripplot(data=students, x="group", y="final_score", order=["A", "B", "C"], color=CATEGORICAL[0], size=3, alpha=0.6, ax=ax)
save_figure(fig, "e5_box_strip.png", folder)

customers = pd.read_csv("datasets/generated/customers.csv")
fig, ax = plt.subplots(figsize=(5, 3))
sns.kdeplot(customers["annual_income_keur"], fill=True, color=CATEGORICAL[0], ax=ax)
save_figure(fig, "e6_income_kde.png", folder)

houses = pd.read_csv("datasets/generated/house_prices.csv")
corr = houses.select_dtypes("number").corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool), k=1), cmap=diverging_cmap(), vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
save_figure(fig, "e7_house_corr.png", folder)
print(corr["price_keur"].drop("price_keur").round(2).sort_values(ascending=False).to_dict())

customers["âge"] = np.where(customers["age"] < 50, "< 50 ans", ">= 50 ans")
fig, ax = plt.subplots(figsize=(5, 3))
sns.ecdfplot(data=customers, x="avg_basket_eur", hue="âge", palette=CATEGORICAL[:2], ax=ax)
save_figure(fig, "e8_basket_ecdf.png", folder)
print(customers.groupby("âge")["avg_basket_eur"].median().round(1).to_dict())
```

```text output
{'surface_m2': 0.77, 'rooms': 0.69, 'age_years': -0.16, 'has_garden': -0.17, 'distance_center_km': -0.44}
{'< 50 ans': 66.3, '>= 50 ans': 56.6}
```

Exercice 2 : la densité du revenu montre plusieurs bosses (un mélange de populations) : c'est l'indice de segments de clientèle distincts, que l'on retrouvera avec k-means au module 5.

## Leçon 4.3

### Quiz

1. Quand le message tient en une valeur (un taux, un total) : une tuile avec le nombre, une phrase et éventuellement une comparaison.
2. L'alignement des deux échelles est arbitraire : on peut fabriquer une corrélation visuelle. Alternatives : deux graphiques, petits multiples, ou indices base 100 sur un seul axe.
3. Un titre qui énonce ce que le lecteur doit retenir (« Le Nord réalise 29 % des ventes ») au lieu de décrire le graphique.
4. Palette testée pour le daltonisme, information doublée par des étiquettes, des formes ou des motifs, légende, contraste suffisant, tableau des valeurs.
5. Des barres horizontales triées (éventuellement les 10 premières + « autres »).
6. Un taux calculé sur 5 personnes et un taux sur 500 n'ont pas la même fiabilité.

### Exercices guidés

```python
import matplotlib.pyplot as plt
import pandas as pd

from lab.plotting import CATEGORICAL, NEUTRAL, TEXT, apply_style, save_figure

apply_style()
folder = "outputs/module04/corrections"
sales = pd.read_csv("datasets/generated/sales.csv", parse_dates=["date"])
sales["revenue"] = sales["quantity"] * sales["unit_price"]

by_region = (sales.groupby("region")["revenue"].sum() / 1000).sort_values()
share_nord = by_region["Nord"] / by_region.sum()
fig, ax = plt.subplots(figsize=(5, 2.5))
bars = ax.barh(by_region.index, by_region.values, color=CATEGORICAL[0])
ax.bar_label(bars, fmt="%.0f k€", padding=3)
ax.set_title(f"Le Nord réalise {share_nord:.0%} du chiffre d'affaires")
save_figure(fig, "e9_region_headline.png", folder)

students = pd.read_csv("datasets/generated/students.csv")
fig, ax = plt.subplots(figsize=(3, 1.5))
ax.axis("off")
ax.text(0, 0.6, f"{students['passed'].mean():.0%}", fontsize=30, fontweight="bold", color=TEXT, transform=ax.transAxes)
ax.text(0, 0.2, f"des {len(students)} étudiants sont admis", color=TEXT, transform=ax.transAxes)
save_figure(fig, "e10_pass_rate_tile.png", folder)

monthly = sales.set_index("date").groupby("region")["revenue"].resample("MS").sum().unstack(level=0)
fig, ax = plt.subplots(figsize=(7, 3))
for region in monthly.columns:
    ax.plot(monthly.index, monthly[region], color=CATEGORICAL[0] if region == "Nord" else NEUTRAL,
            linewidth=2.4 if region == "Nord" else 1.1, label=region)
ax.set_title("Le Nord est en tête un mois sur deux, dont novembre et décembre")
ax.legend()
save_figure(fig, "e11_highlight_nord.png", folder)
print(f"Nord share: {share_nord:.1%} | months where Nord is first: {(monthly.idxmax(axis=1) == 'Nord').sum()} / 12")
```

```text output
Nord share: 29.0% | months where Nord is first: 6 / 12
```

Exercice 4 : (1) axe y tronqué à 35 % qui multiplie visuellement les écarts → axe de 0 à 100 % ; (2) trois couleurs sans signification → une seule couleur ; (3) ni valeurs ni effectif → étiquettes de valeurs et « n = 300 » ; on pourrait ajouter (4) l'incertitude (intervalles de confiance), puisque les écarts ne sont pas significatifs.
