# I01 — Script de traitement de données réutilisable

- **Identifiant** : I01
- **Niveau** : intermédiaire
- **Module** : 1 (jalon pratique du module)
- **Durée indicative** : 2 h

## Objectifs d'apprentissage

- Structurer un script en fonctions courtes et testables : charger → nettoyer → résumer → sauvegarder.
- Écrire une interface en ligne de commande avec `argparse`.
- Gérer proprement les erreurs (fichier absent, colonnes manquantes) avec un code de retour.
- Produire des sorties CSV et JSON réutilisables par d'autres outils.

## Énoncé

Complétez `exercise.py` :

1. `load(path)` : lit le CSV. Lève `FileNotFoundError` si le fichier n'existe pas et `ValueError` si une colonne de `REQUIRED_COLUMNS` manque.
2. `clean(df)` : mêmes règles que l'exercice B03 (vous pouvez réutiliser votre code), avec la colonne `revenue`.
3. `summarize(df, by)` : une ligne par valeur de la colonne `by`, colonnes `[by, n_orders, quantity, revenue, share]`, triées par `revenue` décroissant ; `share` = part du chiffre d'affaires total (arrondie à 4 décimales). Lève `ValueError` si `by` n'est pas une colonne.
4. `save(clean_df, summary, output_dir)` : crée le dossier si besoin, écrit `clean_sales.csv` et `summary.json` (liste d'objets, accents conservés) et renvoie `{"clean": chemin, "summary": chemin}`.
5. `main(argv)` : options `--input` (obligatoire), `--output-dir` (obligatoire), `--by` (défaut `region`), `--verbose`. Renvoie `0` en cas de succès, `1` en cas d'erreur (avec un message clair, **sans** trace d'erreur Python).

## Données d'entrée

`datasets/raw/sales_raw.csv` (voir B03).

## Sortie attendue

```bash
python exercises/intermediate/I01_data_processing_script/exercise.py --input datasets/raw/sales_raw.csv --output-dir outputs/I01 --by region
```

```text
379 clean rows -> outputs/I01/clean_sales.csv
summary by region -> outputs/I01/summary.json
```

Premier élément de `summary.json` : `{"region": "Nord", "n_orders": 122, "quantity": 358, "revenue": 20446.06, "share": 0.3148}`.

## Contraintes

- `main` doit accepter une liste d'arguments (`main(["--input", ...])`) pour être testable.
- Aucune variable globale modifiée ; aucune dépendance autre que pandas et la bibliothèque standard.
- Le script doit fonctionner quel que soit le dossier courant (chemins reçus en arguments).

## Indices

- **Indice 1** : `parser.parse_args(argv)` accepte une liste ; avec `None`, il lit `sys.argv`.
- **Indice 2** : capturez `FileNotFoundError` et `ValueError` dans `main` uniquement, pas dans les fonctions de traitement.
- **Indice 3** : `summary.to_dict(orient="records")` puis `json.dump(..., ensure_ascii=False, indent=2)`.
- **Indice 4** : le module `logging` permet d'afficher des détails seulement avec `--verbose`.

## Vérification

```bash
python -m pytest exercises/intermediate/I01_data_processing_script -q
```

Solution : `solutions/intermediate/I01_data_processing_script/solution.py`.
