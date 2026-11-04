# Leçon 4.1 — Graphiques fondamentaux avec Matplotlib

## Prérequis

- Leçon 1.5 (pandas) et leçon 3.3 (agrégations).

## Objectifs d'apprentissage

- Comprendre l'architecture de Matplotlib : figure, axes, éléments graphiques.
- Tracer des courbes (*line charts*), des diagrammes en barres (*bar charts*), des histogrammes et des nuages de points (*scatter plots*).
- Personnaliser titres, axes, légendes et annotations.
- Enregistrer les figures localement et les retrouver.

## Concepts clés

- **Figure** : la « feuille » qui contient un ou plusieurs graphiques.
- **Axes** : une zone de tracé avec ses axes x et y (attention : *axes* désigne le graphique entier, pas seulement les axes).
- **Interface orientée objet** : `fig, ax = plt.subplots()` puis `ax.plot(...)`, `ax.set_title(...)`. Plus explicite que l'interface `plt.plot(...)`.
- **Marque** (*mark*) : l'élément qui porte la donnée (ligne, barre, point).
- **Encodage** (*encoding*) : la correspondance entre une variable et une propriété visuelle (position, longueur, couleur, taille).
- **Backend** : le moteur de rendu ; `Agg` produit des fichiers sans fenêtre, utile sans écran.

## Explication détaillée

### Figure et axes

`fig, ax = plt.subplots(figsize=(6, 4))` crée une figure de 6 × 4 pouces avec un seul graphique. `plt.subplots(2, 2)` en crée quatre (tableau `axes` de forme (2, 2)). On dessine toujours **sur un `ax`** : `ax.plot`, `ax.bar`, `ax.hist`, `ax.scatter`. On termine par `fig.savefig("chemin.png")` puis `plt.close(fig)` pour libérer la mémoire. Dans Jupyter, la figure s'affiche aussi sous la cellule ; dans un script, `plt.show()` ouvre une fenêtre.

Ce dépôt fournit `lab.plotting.apply_style()` : un style commun (marques fines, grille discrète, palette de couleurs vérifiée pour les daltonismes) et `save_figure(fig, nom)` qui enregistre dans `outputs/module04/`.

### Quel graphique pour quelle question ?

| Question | Graphique | Encodage principal |
|---|---|---|
| Comment une valeur évolue-t-elle dans le temps ? | courbe | position sur l'axe y, x = temps |
| Quelle catégorie est la plus grande ? | barres (souvent horizontales et triées) | longueur |
| Comment les valeurs sont-elles réparties ? | histogramme | hauteur des classes |
| Deux variables numériques sont-elles liées ? | nuage de points | position x et y |

L'œil compare très bien des **positions** et des **longueurs** alignées sur une même base ; il compare mal des angles (camemberts), des aires ou des couleurs. C'est pourquoi la barre et le point dominent.

### Règles de base

- Les barres **commencent à zéro** : la longueur encode la valeur ; tronquer l'axe exagère les écarts. (Une courbe peut ne pas commencer à zéro si l'on s'intéresse aux variations.)
- **Triez** les barres de catégories nominales par valeur : le classement se lit immédiatement.
- **Une seule couleur** suffit quand toutes les barres représentent la même mesure ; la couleur sert à distinguer des **séries** différentes.
- Titres et axes avec **unités** (« Chiffre d'affaires (k€) »).
- Histogramme : le nombre de classes (*bins*) change la lecture ; essayez-en plusieurs.
- Nuage de points dense : réduisez la taille des points et ajoutez de la transparence (`alpha`) pour voir les zones de forte densité.

## Applications concrètes

- Suivi mensuel du chiffre d'affaires (courbe).
- Classement des produits ou des régions (barres).
- Distribution des notes d'un examen (histogramme).
- Relation surface / prix d'un logement (nuage de points).

## Exemples de code Python

### Exemple 1 — courbe : évolution mensuelle des ventes

```python
import matplotlib.pyplot as plt
import pandas as pd

from lab.plotting import apply_style, label_line_ends, save_figure

apply_style()
sales = pd.read_csv("datasets/generated/sales.csv", parse_dates=["date"])
sales["revenue"] = sales["quantity"] * sales["unit_price"]
monthly = sales.set_index("date").groupby("region")["revenue"].resample("MS").sum().unstack(level=0) / 1000

fig, ax = plt.subplots(figsize=(7, 3.6))
for region in monthly.columns:
    ax.plot(monthly.index, monthly[region], marker="o", markersize=4, label=region)
ax.set_ylim(bottom=0)
label_line_ends(ax, monthly.index[-1], monthly.iloc[-1].to_dict())
ax.set_title("Le chiffre d'affaires bondit en novembre-décembre dans toutes les régions")
ax.set_ylabel("Chiffre d'affaires (k€)")
ax.legend(ncols=4, loc="upper left")
path = save_figure(fig, "01_monthly_revenue_line.png")
print("saved:", path.as_posix())
print(monthly.round(1).iloc[[0, 10, 11]].to_string())
```

### Explication du code

- `groupby("region")[...].resample("MS").sum()` agrège par région et par mois ; `unstack` place chaque région dans une colonne.
- Une boucle trace une courbe par région ; la couleur suit la palette dans un ordre fixe.
- Les étiquettes directes en bout de courbe (`label_line_ends`, une petite fonction de `lab/plotting.py` basée sur `ax.annotate`, qui écarte les étiquettes trop proches) évitent les allers-retours vers la légende ; la légende reste présente pour la lecture sans couleur.
- Le **titre énonce le constat**, au lieu de décrire le graphique (« CA par mois »).

### Sortie attendue

```text output
saved: outputs/module04/01_monthly_revenue_line.png
region       Est  Nord  Ouest   Sud
date                               
2024-01-01   7.1  11.5   12.1   8.7
2024-11-01  15.6  22.0   18.1  17.4
2024-12-01  18.8  24.7   17.0  19.5
```

### Exemple 2 — barres horizontales triées et histogramme

```python
import matplotlib.pyplot as plt
import pandas as pd

from lab.plotting import CATEGORICAL, apply_style, save_figure

apply_style()
sales = pd.read_csv("datasets/generated/sales.csv")
sales["revenue"] = sales["quantity"] * sales["unit_price"]
by_product = (sales.groupby("product")["revenue"].sum() / 1000).sort_values()

fig, ax = plt.subplots(figsize=(6, 3))
bars = ax.barh(by_product.index, by_product.values, color=CATEGORICAL[0], height=0.6)
ax.bar_label(bars, fmt="%.0f k€", padding=3, fontsize=8)
ax.set_title("L'écran rapporte deux fois plus que le casque")
ax.set_xlabel("Chiffre d'affaires 2024 (k€)")
ax.grid(axis="y", visible=False)
save_figure(fig, "02_revenue_by_product_bar.png")

students = pd.read_csv("datasets/generated/students.csv")
fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
for ax, bins in zip(axes, [5, 15, 40]):
    ax.hist(students["final_score"], bins=bins, color=CATEGORICAL[0], edgecolor="white", linewidth=0.8)
    ax.axvline(10, color="#2b2b29", linewidth=1, linestyle="--")
    ax.set_title(f"{bins} classes")
    ax.set_xlabel("Note finale /20")
axes[0].set_ylabel("Nombre d'étudiants")
save_figure(fig, "03_final_score_hist.png")
print(by_product.round(1).to_dict())
print("students below 10:", int((students["final_score"] < 10).sum()), "/", len(students))
```

### Explication du code

- `sort_values()` puis `barh` : la barre la plus longue est en haut, le classement se lit sans effort. Une seule couleur, car il n'y a qu'une mesure.
- `bar_label` ajoute les valeurs au bout des barres (étiquetage direct).
- Les trois histogrammes montrent les **mêmes données** : avec 5 classes, la forme est grossière ; avec 40, le bruit domine ; 15 classes donnent une lecture équilibrée. La ligne pointillée marque le seuil de 10/20.

### Sortie attendue

```text output
{'Souris': 60.8, 'Webcam': 71.9, 'Clavier': 87.3, 'Casque': 102.4, 'Ecran': 209.0}
students below 10: 168 / 300
```

### Exemple 3 — nuage de points et petits multiples

```python
import matplotlib.pyplot as plt
import pandas as pd

from lab.plotting import CATEGORICAL, apply_style, save_figure

apply_style()
houses = pd.read_csv("datasets/generated/house_prices.csv")
order = ["Centre", "Residentiel", "Peripherie", "Rural"]

fig, ax = plt.subplots(figsize=(6, 4))
for color, hood in zip(CATEGORICAL, order):
    subset = houses[houses["neighborhood"] == hood]
    ax.scatter(subset["surface_m2"], subset["price_keur"], s=12, alpha=0.6, color=color, label=hood)
ax.set_xlabel("Surface (m²)")
ax.set_ylabel("Prix (k€)")
ax.set_title("À surface égale, le centre est le plus cher")
ax.legend(title="Quartier", markerscale=1.5)
save_figure(fig, "04_surface_price_scatter.png")

fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=True, sharey=True)
for ax, color, hood in zip(axes, CATEGORICAL, order):
    ax.scatter(houses["surface_m2"], houses["price_keur"], s=5, color="#d6d6d1")
    subset = houses[houses["neighborhood"] == hood]
    ax.scatter(subset["surface_m2"], subset["price_keur"], s=8, color=color)
    ax.set_title(hood)
    ax.set_xlabel("Surface (m²)")
axes[0].set_ylabel("Prix (k€)")
save_figure(fig, "05_surface_price_small_multiples.png")
price_per_m2 = (houses["price_keur"] * 1000 / houses["surface_m2"]).groupby(houses["neighborhood"]).median()
print(price_per_m2.reindex(order).round(0).to_dict())
```

### Explication du code

- Le premier nuage superpose les 4 quartiers avec 4 couleurs et une légende : lisible mais chargé.
- Les **petits multiples** (*small multiples*) donnent un graphique par quartier, avec les mêmes axes (`sharex`, `sharey`) : chaque quartier ressort sur le fond gris de l'ensemble des maisons. Avec beaucoup de séries, c'est souvent plus clair qu'un seul graphique multicolore.
- Le prix médian au m² par quartier confirme numériquement ce que montrent les pentes des nuages.

### Sortie attendue

```text output
{'Centre': 4333.0, 'Residentiel': 3351.0, 'Peripherie': 2529.0, 'Rural': 1806.0}
```

## Erreurs fréquentes

- **Barres qui ne commencent pas à zéro** : les écarts paraissent énormes.
- **Une couleur différente par barre** pour une seule mesure : la couleur ne porte aucune information et distrait.
- **Pas d'unités ni de titre** : le lecteur doit deviner.
- **Oublier `plt.close(fig)`** dans une boucle : la mémoire se remplit (avertissement « More than 20 figures »).
- **Mélanger interface `plt.` et `ax.`** : on modifie le mauvais graphique.
- **Histogramme avec un seul nombre de classes** sans vérifier la robustesse de la forme.
- **Points opaques superposés** : la densité est invisible ; utilisez `alpha` et une taille réduite.

## Exercices guidés

1. Tracez la courbe du nombre de lignes de vente par semaine (`resample("W")`) avec un titre qui énonce un constat.
2. Tracez en barres horizontales triées la quantité totale vendue par région, avec les valeurs au bout des barres.
3. Tracez l'histogramme de `annual_income_keur` de `customers.csv` puis celui de son logarithme. Laquelle des deux formes est la plus symétrique ?
4. Tracez le nuage `study_hours_week` / `final_score` en coloriant les points selon `passed` (deux couleurs, une légende).

Corrigés : [corrections.md](corrections.md), section « Leçon 4.1 ».

## Défis autonomes

- Reproduisez l'exemple 1 sous forme de petits multiples (un graphique par région) : quelle version préférez-vous et pourquoi ?
- Créez une figure 2 × 2 qui résume `customers.csv` (âge, revenu, score de dépense, panier moyen) et enregistrez-la en PNG et en SVG (`fig.savefig("x.svg")`).

## Quiz

1. Quelle différence entre une `Figure` et un `Axes` ?
2. Pourquoi les barres doivent-elles commencer à zéro alors que ce n'est pas obligatoire pour une courbe ?
3. Quel graphique choisir pour comparer 12 catégories ? Comment les ordonner ?
4. Qu'est-ce qu'un petit multiple et quand l'utiliser ?
5. À quoi sert `alpha` dans un nuage de points ?
6. Pourquoi appeler `plt.close(fig)` ?

Réponses : [corrections.md](corrections.md), section « Leçon 4.1 ».

## Résumé

- `fig, ax = plt.subplots()` puis on dessine sur `ax` ; on enregistre avec `savefig` et on ferme.
- Courbe = évolution ; barres = comparaison ; histogramme = distribution ; nuage = relation.
- Barres à zéro, triées, une couleur par série ; unités et titre-constat.
- Les petits multiples remplacent avantageusement les graphiques surchargés.

## Pour aller plus loin

- Ouvrez les fichiers de `outputs/module04/` dans VS Code et comparez-les à leurs explications.
- Leçon suivante : [Distributions, boîtes à moustaches, corrélations et cartes de chaleur](02_distributions_correlations.md).
