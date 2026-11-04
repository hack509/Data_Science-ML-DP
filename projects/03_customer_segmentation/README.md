# Projet 3 — Segmentation de clientèle

**Difficulté** : ★★☆☆ — **Modules** : 3, 4, 5 — **Durée indicative** : 4 à 6 heures

## Énoncé du problème

Une enseigne veut adapter ses offres à des profils de clients, mais ne dispose d'aucune étiquette : il faut **découvrir** des groupes de clients au comportement semblable à partir de leurs données d'achat, choisir un nombre de segments justifié, décrire chaque segment dans le langage de l'équipe marketing et vérifier que la segmentation est stable.

## Objectifs d'apprentissage

- Préparer des données pour un algorithme à base de distances (standardisation, leçon 3.2).
- Choisir k avec la méthode du coude et le score de silhouette, et mesurer la stabilité (leçon 5.5).
- Décrire et nommer des segments à partir de leurs profils moyens.
- Visualiser des données en 6 dimensions avec l'ACP (leçon 5.5).
- Rédiger un rapport qui distingue ce que l'algorithme montre et ce qu'il faut valider.

## Données

`datasets/generated/customers.csv` : 500 clients **synthétiques**.

| Colonne | Description |
|---|---|
| `customer_id` | identifiant |
| `age` | âge |
| `annual_income_keur` | revenu annuel (k€) |
| `spending_score` | score de dépense 1-100 |
| `visits_per_month` | visites par mois |
| `online_ratio` | part des achats en ligne (0-1) |
| `avg_basket_eur` | panier moyen (€) |

Le fichier séparé `customers_true_segments.csv` contient les 4 segments utilisés par le générateur. Il **n'est utilisé que pour contrôler la méthode** à la fin : avec de vraies données, une telle vérité n'existe pas.

## Architecture

```text
03_customer_segmentation/
├── run.py
├── segmentation/clustering.py   choix de k, stabilité, profils, noms, figures, rapport
├── tests/test_project.py
└── outputs/                     report.md, customers_with_segments.csv, figures
```

## Étapes

1. **Explorer** : distributions et corrélations des 6 variables (module 4).
2. **Standardiser** toutes les variables.
3. **Choisir k** de 2 à 8 : inertie (coude) et silhouette. Retenir k et justifier.
4. **Stabilité** : relancer k-means avec 5 graines (une initialisation chacune) et calculer l'ARI moyen entre partitions.
5. **Profils** : moyenne de chaque variable par segment, effectif et part.
6. **Nommer** chaque segment par des règles lisibles (âge, revenu, part du numérique, score de dépense) comparées au client moyen.
7. **Visualiser** : courbes de choix de k, projection ACP colorée par segment.
8. **Rapport** et fichier des clients avec leur segment ; contrôle final avec les segments du générateur.

## Lancer la solution de référence

```bash
python projects/03_customer_segmentation/run.py
python -m pytest projects/03_customer_segmentation -q
```

## Indicateurs et résultats attendus

| Indicateur | Valeur |
|---|---|
| k retenu (silhouette maximale) | 4 (silhouette ≈ 0,517, de justesse devant k = 3 : 0,511) |
| Stabilité (ARI moyen entre graines) | ≈ 0,88 |
| ARI avec les segments du générateur | ≈ 0,985 |
| Segments | familles économes (≈ 29 %), clients premium (≈ 21 %), jeunes connectés (≈ 32 %), seniors fidèles (≈ 19 %) |

La stabilité n'est pas parfaite parce que chaque relance n'utilise qu'une initialisation : certaines tombent dans un minimum local. C'est la raison d'être de `n_init=10` pour le modèle final.

## Défis pour aller plus loin

- Comparez avec `AgglomerativeClustering` et `GaussianMixture` (scikit-learn) : obtient-on les mêmes segments ?
- Ajoutez une variable bruitée sans structure et observez l'effet sur la silhouette et l'ARI.
- Produisez une « fiche segment » par groupe (graphique radar ou barres des écarts à la moyenne, en écart-types).
- Affectez de nouveaux clients à un segment avec `model.predict` après la même standardisation.

## Solution de référence

`segmentation/clustering.py` : `choose_k`, `stability`, `name_segment`, `profile_segments`, `make_figures`, `write_report`, `run`.

## Pistes d'amélioration

- Choisir k aussi selon l'usage métier (nombre de campagnes réalisables).
- Suivre la stabilité des segments dans le temps (les clients changent de segment).
- Remplacer les règles de nommage par un atelier avec l'équipe marketing, à partir des profils.
