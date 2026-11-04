# Jeux de données

Tous les jeux de données sont **synthétiques** et créés localement, sans téléchargement, par :

```bash
python scripts/generate_datasets.py          # regenerate everything (about 15 s)
python scripts/generate_datasets.py --list   # list the datasets and their status
```

La graine aléatoire est fixe (`SEED = 42`) : le script produit toujours les mêmes fichiers avec les mêmes versions de bibliothèques. Les résultats obtenus sur ces données illustrent des méthodes ; **ils ne prouvent rien sur le monde réel**.

| Dossier | Contenu |
|---|---|
| `raw/` | données « sales », telles qu'un export réel les fournirait ; on ne les modifie jamais à la main |
| `generated/` | données propres |
| `processed/` | emplacement libre pour vos fichiers nettoyés (contenu ignoré par Git) |

## `generated/students.csv` — notes et habitudes de travail (300 lignes)

Sert aux modules 1 à 5 et aux projets 1 et 6.

| Colonne | Type | Description |
|---|---|---|
| `student_id` | texte | identifiant `S0001`... |
| `group` | texte | groupe A, B ou C, tiré **au hasard** (aucun effet sur les notes) |
| `study_hours_week` | réel | heures d'étude par semaine |
| `attendance_rate` | réel | assiduité entre 0 et 1 |
| `sleep_hours` | réel | heures de sommeil par nuit |
| `previous_score` | réel | note du trimestre précédent /20 |
| `math`, `physics`, `french` | réel | notes /20 |
| `final_score` | réel | 0,4 maths + 0,3 physique + 0,3 français |
| `passed` | 0/1 | 1 si `final_score` ≥ 10 |

Génération : un niveau caché (loi normale) et les habitudes de travail déterminent les notes, avec du bruit.

## `raw/students_raw.csv` — version sale (308 lignes)

Mêmes étudiants sans `final_score` ni `passed`, avec : 8 lignes dupliquées, valeurs manquantes (heures d'étude, sommeil, physique), groupes mal écrits (`" a "`, `"b"`), assiduité parfois en pourcentage (`"90.4%"`), `"absent"` en maths, 3 heures d'étude impossibles.

## `generated/student_activities.csv` — activités (260 lignes)

`student_id`, `club` (sport, musique, theatre, robotique, aucun), `volunteer_hours_year`. 250 étudiants existants et 10 identifiants inconnus (`S0901`...), pour les leçons sur les jointures.

## `generated/sales.csv` — ventes 2024 (3 000 lignes)

`order_id`, `date`, `region` (Nord, Sud, Est, Ouest), `product`, `category`, `quantity`, `unit_price`. Saisonnalité introduite : plus de ventes en novembre et décembre, moins le week-end ; remises aléatoires de 10 ou 20 %.

## `raw/sales_raw.csv` — export sale (410 lignes)

400 lignes de `sales.csv` avec noms de produits mal écrits, prix du type `"19,90 EUR"`, quantités manquantes ou négatives, dates manquantes et 10 doublons.

## `raw/orders.json` — commandes imbriquées (40 clients)

Liste de clients `{customer_id, city, member_since, newsletter, orders: [{order_id, date, items: [{product, quantity, unit_price}]}]}`. Certains clients n'ont aucune commande.

## `generated/house_prices.csv` — logements (800 lignes)

`surface_m2`, `rooms`, `age_years`, `distance_center_km`, `neighborhood`, `has_garden`, `energy_class` (A à G), `price_keur` (cible). Le prix est un produit de facteurs avec un bruit multiplicatif : la relation est presque linéaire en logarithme.

## `generated/customers.csv` (500 lignes) et `customers_true_segments.csv`

`customer_id`, `age`, `annual_income_keur`, `spending_score`, `visits_per_month`, `online_ratio`, `avg_basket_eur`. Quatre segments cachés (`jeunes_connectes`, `familles_economes`, `cadres_premium`, `seniors_fideles`), dans le fichier séparé `customers_true_segments.csv`, réservé à la vérification.

## `generated/sms_spam.csv` — SMS (1 200 lignes)

`message_id`, `text`, `label` (`ham`, `spam`). Modèles de phrases en français, avec cas difficiles, 15 % de messages sans accents, 15 % avec une faute de frappe et **2 % d'étiquettes inversées** (erreurs d'annotation simulées). Nombreux doublons : 742 textes distincts.

## `generated/reviews_fr.csv` — avis (800 lignes)

`review_id`, `text`, `sentiment` (`positif`, `négatif`), `rating` (1, 2, 4, 5). Comprend des tournures où l'ordre des mots inverse le sens (« n'est pas décevant ») et **3 % d'étiquettes inversées**.

## `generated/binary_classification.csv`, `multiclass_classification.csv`, `regression.csv`

Jeux génériques : `x1`... et `target`, générés par `sklearn.datasets.make_classification` (600 lignes à 2 classes, 900 lignes à 3 classes). `regression.csv` (500 lignes) : y = 3·x1 − 2·x2 + 0,5·x3² + 2·sin(x4) + bruit.

## `generated/shapes/` et `generated/shapes.npz` — images (300)

100 cercles, 100 carrés, 100 triangles en PNG 28 × 28 niveaux de gris, dessinés avec Pillow (taille, position, intensité, orientation aléatoires, bruit). `shapes/index.csv` : `path`, `label`, `label_id`. `shapes.npz` : tableaux `images`, `labels`, `class_names`.

## `generated/knowledge_base.json` — base documentaire (14 fiches)

`doc_id`, `title`, `text` : courtes fiches de cours en français pour le mini-RAG de la leçon 8.4.

## Fichier manquant ?

Les fonctions de `lab/data.py` ne téléchargent jamais rien : un fichier absent provoque une erreur `MissingDatasetError` qui indique la commande de régénération.
