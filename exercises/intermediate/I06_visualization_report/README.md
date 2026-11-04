# I06 — Rapport visuel automatisé

- **Identifiant** : I06
- **Niveau** : intermédiaire
- **Module** : 4 (jalon du module)
- **Durée indicative** : 2 h

## Objectifs d'apprentissage

- Produire automatiquement plusieurs figures à partir d'un fichier de données.
- Écrire des titres-constats calculés à partir des données (et non écrits à la main).
- Gérer proprement les figures (enregistrement, fermeture, dossier de sortie).

## Énoncé

Complétez `exercise.py` :

1. `prepare(sales)` : copie du DataFrame avec `date` converti en date, `revenue = quantity × unit_price` et `month` (1 à 12).
2. `monthly_revenue(df)` : Series du chiffre d'affaires par mois (index 1 à 12).
3. `build_report(sales, output_dir)` : crée le dossier si nécessaire et enregistre **quatre** figures PNG :

| Clé | Fichier | Graphique | Le titre doit contenir |
|---|---|---|---|
| `monthly_revenue` | `monthly_revenue.png` | courbe du CA mensuel | la part de novembre + décembre en % |
| `revenue_by_product` | `revenue_by_product.png` | barres horizontales triées | le produit n° 1 et sa part en % |
| `basket_distribution` | `basket_distribution.png` | histogramme des montants de lignes | la médiane en € |
| `price_vs_quantity` | `price_vs_quantity.png` | nuage prix unitaire / quantité | la corrélation r |

   La fonction renvoie `{clé: {"path": chemin, "title": titre}}` et ne laisse **aucune figure ouverte**.

## Données d'entrée

`datasets/generated/sales.csv`.

## Sortie attendue

Quatre fichiers PNG et des titres comme :

```text
Novembre et décembre font 29% du chiffre d'affaires annuel
Ecran : 39% du chiffre d'affaires
```

## Contraintes

- Les nombres des titres sont **calculés** (f-strings), jamais écrits en dur.
- Une couleur par série au maximum ; barres commençant à zéro ; axes avec unités.
- Utilisez le backend `Agg` (pas de fenêtre) et fermez chaque figure.

## Indices

- **Indice 1** : `from lab.plotting import apply_style, CATEGORICAL` donne le style du cours.
- **Indice 2** : `f"{part:.0%}"` formate 0.288 en `29%`.
- **Indice 3** : `plt.get_fignums()` liste les figures encore ouvertes.

## Vérification

```bash
python -m pytest exercises/intermediate/I06_visualization_report -q
```

Ouvrez ensuite les PNG produits dans `outputs/I06/` (lancez `python exercises/intermediate/I06_visualization_report/exercise.py`) et relisez-les avec la grille de la leçon 4.3.

Solution : `solutions/intermediate/I06_visualization_report/solution.py`.
