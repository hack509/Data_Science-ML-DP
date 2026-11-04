# B02 — Analyse de ventes

- **Identifiant** : B02
- **Niveau** : débutant
- **Module** : 1 (leçons 1.1 et 1.2)
- **Durée indicative** : 45 min

## Objectifs d'apprentissage

- Lire un fichier CSV avec le module standard `csv` et convertir les types.
- Agréger des valeurs dans un dictionnaire (comptage, sommes).
- Trier selon un critère personnalisé.

## Énoncé

Complétez `exercise.py` :

1. `load_sales(path)` : lit le CSV et renvoie une liste de dictionnaires, avec `quantity` converti en `int` et `unit_price` en `float`.
2. `revenue_by(records, key)` : chiffre d'affaires (`quantity × unit_price`) par valeur de la colonne `key` (par exemple `"region"`), arrondi à 2 décimales **à la fin**, trié par chiffre d'affaires décroissant.
3. `top_products(records, n=3)` : liste des `n` couples `(produit, quantité totale vendue)` triés par quantité décroissante, puis par nom de produit en cas d'égalité.
4. `monthly_revenue(records)` : `{ "AAAA-MM": chiffre d'affaires }` trié par mois croissant.

## Données d'entrée

`datasets/generated/sales.csv` : colonnes `order_id, date, region, product, category, quantity, unit_price`.

## Sortie attendue

```text
revenue_by(sales, "region")
{'Nord': 153939.0, 'Sud': 134403.01, 'Ouest': 128111.64, 'Est': 114852.25}
top_products(sales, 3)
[('Souris', 2591), ('Clavier', 2323), ('Casque', 1819)]
```

## Contraintes

- Python standard uniquement (pas de pandas) : le but est de comprendre ce que pandas fera ensuite pour vous.
- Les dictionnaires renvoyés doivent respecter l'ordre demandé (les dictionnaires Python conservent l'ordre d'insertion).

## Indices

- **Indice 1** : `collections.defaultdict(float)` évite de tester si la clé existe.
- **Indice 2** : Pour trier par quantité décroissante puis par nom : `key=lambda kv: (-kv[1], kv[0])`.
- **Indice 3** : Le mois est `date[:7]` pour une date au format `AAAA-MM-JJ`.

## Vérification

```bash
python -m pytest exercises/beginner/B02_sales_records -q
```

Solution : `solutions/beginner/B02_sales_records/solution.py`.
