# B03 — Nettoyage d'un fichier CSV

- **Identifiant** : B03
- **Niveau** : débutant
- **Module** : 1 (leçon 1.5), prépare le module 3
- **Durée indicative** : 1 h

## Objectifs d'apprentissage

- Repérer les problèmes de qualité d'un export réel : formats incohérents, valeurs manquantes, doublons, valeurs impossibles.
- Corriger ces problèmes avec pandas sans modifier les données d'origine.
- Produire un petit rapport de nettoyage chiffré.

## Énoncé

Le fichier `datasets/raw/sales_raw.csv` est un export « sale » de 410 lignes de ventes. Complétez `exercise.py` :

1. `parse_price(value)` : convertit `"19,90 EUR"`, `"39.9"` ou `39.9` en `float` ; une valeur manquante donne `NaN`.
2. `clean_sales(raw)` : renvoie un **nouveau** DataFrame dans lequel :
   - les noms de produits sont nettoyés (`"  CLAVIER "` et `"clavier"` deviennent `"Clavier"`) ;
   - `unit_price` est numérique, `quantity` est entier (`int`) et `date` est de type date ;
   - les commandes en double (même `order_id`) sont supprimées (on garde la première) ;
   - les lignes sans date, sans quantité ou sans prix sont supprimées, ainsi que les quantités ≤ 0 ;
   - une colonne `revenue = quantity × unit_price` arrondie à 2 décimales est ajoutée ;
   - les lignes sont triées par `order_id` et l'index est renuméroté à partir de 0.
3. `cleaning_report(raw, clean)` : dictionnaire `rows_before`, `rows_after`, `duplicates_removed` (nombre d'`order_id` en double dans `raw`), `total_revenue` (arrondi à 2 décimales).

## Données d'entrée

`datasets/raw/sales_raw.csv` — colonnes `order_id, date, region, product, category, quantity, unit_price`. Ouvrez-le d'abord avec `pd.read_csv` et inspectez `head(20)`, `info()` et `isna().sum()`.

## Sortie attendue

```text
{'rows_before': 410, 'rows_after': 379, 'duplicates_removed': 10, 'total_revenue': 64939.08}
```

Les produits restants sont exactement `Casque, Clavier, Ecran, Souris, Webcam`.

## Contraintes

- `clean_sales` ne doit pas modifier le DataFrame reçu (travaillez sur `raw.copy()`).
- Pas de boucle `for` sur les lignes : utilisez les opérations vectorisées de pandas (`.str`, `map`, `to_numeric`...).

## Indices

- **Indice 1** : `df["product"].str.strip().str.capitalize()`.
- **Indice 2** : `pd.to_numeric(serie, errors="coerce")` transforme les valeurs invalides en `NaN`.
- **Indice 3** : `str.removesuffix("EUR")` puis `replace(",", ".")`.
- **Indice 4** : supprimez les doublons **avant** de filtrer, pour compter correctement.

## Vérification

```bash
python -m pytest exercises/beginner/B03_clean_csv -q
```

Solution : `solutions/beginner/B03_clean_csv/solution.py`.
