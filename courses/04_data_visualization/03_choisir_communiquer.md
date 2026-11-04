# Leçon 4.3 — Choisir un graphique et communiquer un résultat

## Prérequis

- Leçons 4.1 et 4.2.

## Objectifs d'apprentissage

- Choisir une forme de graphique à partir de la question posée, ou décider qu'un nombre suffit.
- Reconnaître et corriger les graphiques trompeurs (axes tronqués, double axe, couleurs mal choisies).
- Mettre en valeur un message : titre-constat, annotation, mise en avant d'une série.
- Rendre un graphique accessible (daltonisme, impression, lecture sans couleur).

## Concepts clés

- **Forme** (*chart form*) : le type de graphique choisi pour une question.
- **Titre-constat** (*headline title*) : un titre qui énonce ce que montre le graphique.
- **Mise en avant** (*highlighting*) : colorer une série, griser les autres.
- **Indicateur clé** (*KPI, stat tile*) : un seul nombre mis en valeur, parfois préférable à un graphique.
- **Double axe** (*dual axis*) : deux échelles y sur un même graphique ; presque toujours trompeur.
- **Accessibilité** : le graphique reste lisible sans la couleur (daltonisme, impression en noir et blanc).

## Explication détaillée

### Partir de la question

| La question porte sur... | Forme recommandée | À éviter |
|---|---|---|
| un seul chiffre clé | un nombre mis en valeur (+ comparaison) | une barre unique, un camembert à 2 parts |
| une évolution dans le temps | courbe | barres pour de longues séries |
| un classement / une comparaison | barres horizontales triées | camembert, radar |
| une distribution | histogramme, boîte, ECDF | moyenne seule |
| une relation entre deux variables | nuage de points | deux courbes sur deux axes |
| une composition (parts d'un tout) | barre empilée à 100 % (≤ 6 parts) ou barres | camembert à 10 parts |
| de nombreuses séries | petits multiples, mise en avant | 12 couleurs sur un graphique |

### Les graphiques trompeurs

- **Axe des barres tronqué** : une différence de 2 % paraît énorme.
- **Double axe y** : on peut faire « coïncider » n'importe quelles courbes en ajustant les deux échelles ; le lecteur voit une corrélation inventée. Faites deux graphiques, ou ramenez les deux séries à une base 100 commune.
- **Aires et volumes** : doubler le rayon d'un disque quadruple sa surface.
- **Palette arc-en-ciel pour une intensité** : des frontières de couleur apparaissent là où les données sont continues.
- **Cumuler des choux et des carottes** : additionner des unités différentes.
- **Choisir la période** qui arrange le message (*cherry picking*).

### Mettre en valeur un message

1. **Titre-constat** : « Les ventes de fin d'année représentent 29 % du chiffre d'affaires » plutôt que « Ventes par mois ».
2. **Mise en avant** : la série qui porte le message en couleur, les autres en gris.
3. **Annotation** : une flèche ou un texte sur le point important (« pic de décembre »).
4. **Simplicité** : supprimer ce qui n'aide pas (bordures, quadrillage marqué, 3D, légendes inutiles).
5. **Contexte** : une ligne de référence (moyenne, objectif, seuil de réussite).

### Accessibilité

Environ 8 % des hommes et 0,5 % des femmes ont une forme de daltonisme. Règles simples : ne jamais faire porter l'information **par la couleur seule** (légende + étiquettes directes, formes de marqueurs différentes, motifs) ; utiliser une palette testée (celle de `lab/plotting.py` a été vérifiée avec un simulateur de daltonisme) ; texte suffisamment contrasté ; une version tableau des chiffres pour qui en a besoin.

### Communiquer honnêtement

Un graphique est un argument. Indiquez la source et la période, les effectifs, l'incertitude quand elle compte (intervalles de confiance, barres d'erreur) et les limites. Avec des données synthétiques comme celles de ce cours, précisez-le : les constats illustrent une méthode, ils ne décrivent pas le monde réel.

## Applications concrètes

- Une diapositive de comité de direction : un titre-constat, un graphique, une annotation.
- Un rapport automatique hebdomadaire où chaque figure répond à une question fixe.
- La présentation des résultats d'un modèle de ML : métriques avec incertitude, comparaison à une référence.

## Exemples de code Python

### Exemple 1 — un nombre plutôt qu'un graphique, et la mise en avant

```python
import matplotlib.pyplot as plt
import pandas as pd

from lab.plotting import CATEGORICAL, NEUTRAL, TEXT, apply_style, label_line_ends, save_figure

apply_style()
sales = pd.read_csv("datasets/generated/sales.csv", parse_dates=["date"])
sales["revenue"] = sales["quantity"] * sales["unit_price"]
share_q4 = sales.loc[sales["date"].dt.month >= 11, "revenue"].sum() / sales["revenue"].sum()

fig, ax = plt.subplots(figsize=(3.2, 1.6))
ax.axis("off")
ax.text(0, 0.62, f"{share_q4:.0%}", fontsize=34, fontweight="bold", color=TEXT, transform=ax.transAxes)
ax.text(0, 0.18, "du chiffre d'affaires 2024\nréalisé en novembre-décembre", fontsize=9, color=TEXT, transform=ax.transAxes)
save_figure(fig, "10_stat_tile.png")

monthly = sales.set_index("date").groupby("product")["revenue"].resample("MS").sum().unstack(level=0) / 1000
fig, ax = plt.subplots(figsize=(7, 3.4))
for product in monthly.columns:
    highlighted = product == "Ecran"
    ax.plot(monthly.index, monthly[product], color=CATEGORICAL[0] if highlighted else NEUTRAL,
            linewidth=2.4 if highlighted else 1.2, alpha=1 if highlighted else 0.6, label=product)
ax.set_ylim(bottom=0)
label_line_ends(ax, monthly.index[-1], monthly.iloc[-1].to_dict())
peak = monthly["Ecran"].idxmax()
ax.annotate("pic de fin d'année", xy=(peak, monthly.loc[peak, "Ecran"]), xytext=(-120, -10),
            textcoords="offset points", arrowprops={"arrowstyle": "->", "color": TEXT}, fontsize=8)
ax.set_ylabel("Chiffre d'affaires (k€)")
ax.set_title("L'écran, premier produit, double ses ventes en fin d'année")
save_figure(fig, "11_highlight_one_series.png")
print(f"Nov-Dec share: {share_q4:.1%}")
print("Ecran Dec / Jan ratio:", round(monthly["Ecran"].iloc[-1] / monthly["Ecran"].iloc[0], 2))
```

### Explication du code

- Quand le message est **un seul chiffre**, une « tuile » avec le nombre et sa phrase d'explication est plus efficace qu'un graphique.
- Dans la deuxième figure, un seul produit est en couleur, les autres en gris fin : l'œil va directement à l'écran. Les noms en bout de ligne remplacent la légende.
- L'annotation avec une flèche désigne le point important.

### Sortie attendue

```text output
Nov-Dec share: 28.8%
Ecran Dec / Jan ratio: 2.2
```

### Exemple 2 — corriger un graphique trompeur

```python
import matplotlib.pyplot as plt
import pandas as pd

from lab.plotting import CATEGORICAL, apply_style, save_figure

apply_style()
students = pd.read_csv("datasets/generated/students.csv")
rate = students.groupby("group")["passed"].mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].bar(rate.index, rate.values, color=CATEGORICAL[:3])
axes[0].set_ylim(35, 56)
axes[0].set_title("Trompeur : axe tronqué, couleurs inutiles")
axes[1].bar(rate.index, rate.values, color=CATEGORICAL[0], width=0.6)
axes[1].bar_label(axes[1].containers[0], fmt="%.0f %%", padding=3)
axes[1].set_ylim(0, 100)
axes[1].set_title("Honnête : échelle complète, n = %d" % len(students))
for ax in axes:
    ax.set_ylabel("Taux de réussite (%)")
    ax.set_xlabel("Groupe")
save_figure(fig, "12_truncated_axis.png")
print(rate.round(1).to_dict())
print("ratio of bar heights, truncated axis: %.1f" % ((rate["C"] - 35) / (rate["B"] - 35)))
print("true ratio: %.2f" % (rate["C"] / rate["B"]))
```

### Explication du code

À gauche, l'axe commence à 35 % : la barre C paraît plusieurs fois plus haute que la barre B, alors que le vrai rapport est d'environ 1,4. Les trois couleurs n'apportent rien (le groupe est déjà écrit sous chaque barre). À droite : échelle de 0 à 100 %, une seule couleur, valeurs affichées et effectif indiqué. Rappel de la leçon 2.4 : ces écarts entre groupes ne sont pas significatifs, et les groupes n'ont aucun effet dans le générateur — un graphique honnête n'empêche pas une sur-interprétation, il faut aussi parler d'incertitude.

### Sortie attendue

```text output
{'A': 39.8, 'B': 39.0, 'C': 54.3}
ratio of bar heights, truncated axis: 4.8
true ratio: 1.39
```

### Exemple 3 — remplacer un double axe par une base 100

```python
import matplotlib.pyplot as plt
import pandas as pd

from lab.plotting import apply_style, label_line_ends, save_figure

apply_style()
sales = pd.read_csv("datasets/generated/sales.csv", parse_dates=["date"])
sales["revenue"] = sales["quantity"] * sales["unit_price"]
monthly = sales.set_index("date").resample("MS").agg({"revenue": "sum", "order_id": "count"})
monthly["avg_line_value"] = monthly["revenue"] / monthly["order_id"]

indexed = 100 * monthly[["revenue", "order_id", "avg_line_value"]] / monthly[["revenue", "order_id", "avg_line_value"]].iloc[0]
indexed.columns = ["chiffre d'affaires", "nombre de lignes", "valeur moyenne d'une ligne"]
fig, ax = plt.subplots(figsize=(7, 3.4))
for column in indexed.columns:
    ax.plot(indexed.index, indexed[column], label=column)
ax.axhline(100, color="#8a8a85", linewidth=1, linestyle="--")
label_line_ends(ax, indexed.index[-1], indexed.iloc[-1].to_dict())
ax.set_ylabel("Indice (janvier = 100)")
ax.set_title("La hausse de fin d'année vient du volume, pas du panier")
ax.legend(loc="upper left")
save_figure(fig, "13_indexed_instead_of_dual_axis.png")
print(indexed.iloc[[0, 10, 11]].round(0).to_string())
```

### Explication du code

Le chiffre d'affaires (des dizaines de milliers d'euros) et le nombre de lignes (quelques centaines) n'ont pas la même unité. Plutôt que deux axes y, on divise chaque série par sa valeur de janvier et on multiplie par 100 : toutes les courbes partent de 100 et se comparent sur **un seul axe**. On voit que la hausse vient du nombre de ventes, la valeur moyenne d'une ligne restant à peu près stable.

### Sortie attendue

```text output
            chiffre d'affaires  nombre de lignes  valeur moyenne d'une ligne
date                                                                        
2024-01-01               100.0             100.0                       100.0
2024-11-01               185.0             178.0                       104.0
2024-12-01               203.0             199.0                       102.0
```

## Erreurs fréquentes

- **Titre descriptif** (« Ventes par mois ») au lieu d'un constat.
- **Graphique pour un seul nombre**.
- **Double axe y**.
- **Couleurs décoratives** sans rôle (une couleur par barre d'une même mesure).
- **Information portée par la couleur seule** : inaccessible à une partie des lecteurs.
- **Oublier effectifs, source, période et incertitude**.
- **Surcharge** : 3D, ombres, quadrillage épais, trop de séries.

## Exercices guidés

1. Transformez le graphique « chiffre d'affaires par région » en barres triées avec un titre-constat et les valeurs affichées.
2. Créez une tuile qui affiche le taux de réussite global des étudiants avec l'effectif.
3. Sur la courbe mensuelle par région, mettez en avant la région Nord et grisez les autres.
4. Listez trois défauts du graphique de gauche de l'exemple 2 et proposez une correction pour chacun.

Corrigés : [corrections.md](corrections.md), section « Leçon 4.3 ».

## Défis autonomes

- Prenez un graphique d'un article de presse ou d'un rapport que vous avez sous la main (hors ligne) et réécrivez son titre en titre-constat. Le graphique soutient-il vraiment le constat ?
- Réalisez une page de synthèse (figure 2 × 2) sur `customers.csv` où chaque sous-graphique a un titre-constat.

## Quiz

1. Quand préférer un nombre seul à un graphique ?
2. Pourquoi le double axe y est-il trompeur ? Quelle alternative ?
3. Qu'est-ce qu'un titre-constat ?
4. Comment rendre un graphique lisible par une personne daltonienne ?
5. Quelle forme pour comparer 15 catégories ?
6. Pourquoi indiquer l'effectif sur un graphique de taux ?

Réponses : [corrections.md](corrections.md), section « Leçon 4.3 ».

## Résumé

- La question choisit la forme ; parfois la meilleure forme est un nombre.
- Évitez axes tronqués pour les barres, doubles axes, couleurs décoratives et arc-en-ciel.
- Titre-constat, mise en avant, annotation et ligne de référence portent le message.
- Accessibilité : jamais la couleur seule ; palette testée ; tableau disponible.

## Pour aller plus loin

- Exercice **I06** (`exercises/intermediate/I06_visualization_report`).
- Module 5 : visualiser les résultats des modèles (matrices de confusion, courbes ROC, courbes d'apprentissage).
