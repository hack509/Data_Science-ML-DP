# B04 — Import et transformation de données JSON

- **Identifiant** : B04
- **Niveau** : débutant
- **Module** : 1 (leçon 1.5)
- **Durée indicative** : 1 h

## Objectifs d'apprentissage

- Lire un fichier JSON imbriqué avec le module `json`.
- Aplatir une structure à trois niveaux (clients → commandes → articles) en table.
- Construire un résumé par client en n'oubliant pas les clients sans commande.

## Énoncé

Complétez `exercise.py` :

1. `load_customers(path)` : charge le fichier JSON (liste de clients).
2. `flatten_orders(customers)` : DataFrame avec **une ligne par article commandé** et exactement les colonnes `customer_id, city, order_id, date, product, quantity, unit_price, line_total` (dans cet ordre), où `line_total = quantity × unit_price` arrondi à 2 décimales.
3. `customer_summary(customers)` : DataFrame indexé par `customer_id` (index nommé `customer_id`, dans l'ordre du fichier), avec les colonnes :
   - `n_orders` : nombre de commandes (0 si aucune) ;
   - `total_spent` : somme des `line_total` (0.0 si aucune commande), arrondie à 2 décimales ;
   - `favorite_product` : produit acheté en plus grande quantité (en cas d'égalité, le premier par ordre alphabétique) ; valeur manquante (`NaN`/`None`) si aucune commande.

## Données d'entrée

`datasets/raw/orders.json` : 40 clients. Structure d'un client :

```json
{"customer_id": "C002", "city": "Paris", "member_since": 2019, "newsletter": false,
 "orders": [{"order_id": "W0001", "date": "2024-08-26",
             "items": [{"product": "Casque", "quantity": 3, "unit_price": 59.9}]}]}
```

## Sortie attendue

- `flatten_orders` : 124 lignes.
- `customer_summary` : 40 lignes, dont 14 clients avec `n_orders == 0`.
- Plus gros client : `C024` avec 1331.1 EUR, produit favori `Clavier`.

## Contraintes

- Aucun client ne doit disparaître du résumé.
- Les fonctions ne doivent rien afficher.

## Indices

- **Indice 1** : trois boucles imbriquées suffisent pour `flatten_orders` (ou `pd.json_normalize`, vu en leçon 1.5).
- **Indice 2** : `groupby(...).sum().reindex(liste_des_ids, fill_value=0.0)` ajoute les clients absents.
- **Indice 3** : pour le favori, triez par client, quantité décroissante puis produit, et gardez la première ligne par client (`drop_duplicates`).

## Vérification

```bash
python -m pytest exercises/beginner/B04_json_transform -q
```

Solution : `solutions/beginner/B04_json_transform/solution.py`.
