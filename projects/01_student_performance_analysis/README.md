# Projet 1 — Analyse de la performance des étudiants

**Difficulté** : ★☆☆☆ — **Modules** : 1, 2, 3, 4 — **Durée indicative** : 4 à 6 heures

## Énoncé du problème

Une équipe pédagogique reçoit chaque trimestre un export brut des notes et des habitudes de travail de ses étudiants. Le fichier contient des doublons, des formats incohérents, des valeurs manquantes et quelques erreurs de saisie. L'équipe veut un **rapport fiable et reproductible** : qualité des données, résultats par groupe et par activité, facteurs associés à la réussite, avec des graphiques clairs et des conclusions prudentes.

## Objectifs d'apprentissage

- Nettoyer un fichier réel en documentant chaque décision (modules 1 et 3).
- Fusionner deux sources de données sans perte ni duplication (leçon 3.4).
- Produire des statistiques descriptives et un test d'hypothèse correctement interprété (module 2).
- Produire des figures avec titres-constats et un rapport Markdown automatique (module 4).

## Données

| Fichier | Contenu | Problèmes connus |
|---|---|---|
| `datasets/raw/students_raw.csv` | 308 lignes : identifiant, groupe, heures d'étude, assiduité, sommeil, note précédente, notes de maths, physique, français | 8 doublons, groupes mal écrits (« a », « b »), assiduité parfois en %, « absent » en maths, 3 heures d'étude impossibles, valeurs manquantes |
| `datasets/generated/student_activities.csv` | 260 lignes : identifiant, activité (club), heures de bénévolat | 10 identifiants inconnus ; 50 étudiants sans ligne |

Les données sont **synthétiques**, générées localement par `python scripts/generate_datasets.py` (graine fixe) : les groupes n'ont aucun effet sur les notes, les heures d'étude, l'assiduité et le niveau antérieur en ont un. Les conclusions illustrent la méthode et ne décrivent pas de vrais étudiants.

## Architecture

```text
01_student_performance_analysis/
├── README.md                       cet énoncé
├── run.py                          point d'entrée (python .../run.py)
├── student_analysis/analysis.py    solution de référence : nettoyage, fusion, tableaux, test, figures, rapport
├── tests/test_project.py           tests pytest
└── outputs/                        résultats générés (ignorés par Git)
```

## Étapes

1. **Diagnostic** : chargez le fichier brut, listez les problèmes (`info`, `isna().sum()`, `duplicated()`, `value_counts()`, `describe()`).
2. **Nettoyage** : supprimez les doublons, normalisez les groupes, convertissez l'assiduité, gardez une colonne `math_absent`, remplacez les heures impossibles par `NaN`, imputez (heures d'étude par la médiane du groupe, sommeil et physique par la médiane globale). Ne pas inventer de note de maths.
3. **Variables dérivées** : note finale = 0,4 maths + 0,3 physique + 0,3 français ; `passed` = note finale ≥ 10.
4. **Fusion** avec les activités (jointure gauche, `validate="one_to_one"`, club `"inconnu"` si absent) ; comptez les identifiants orphelins.
5. **Analyse** : tableaux par groupe, par quartile d'heures d'étude, par activité (avec effectifs) ; corrélations avec la note finale ; test de Welch entre les groupes C et A.
6. **Figures** : distribution des notes, heures d'étude contre note, notes par groupe, carte de corrélations.
7. **Rapport** : `report.md` avec qualité des données, constats chiffrés, tableaux, liste des figures et limites.
8. **Tests** : vérifiez les chiffres clés par des tests pytest.

Essayez de réaliser ces étapes dans votre propre fichier (par exemple `projects/01_student_performance_analysis/mon_analyse.py`) avant de lire la solution.

## Lancer la solution de référence

```bash
python projects/01_student_performance_analysis/run.py
python -m pytest projects/01_student_performance_analysis -q
```

## Indicateurs et résultats attendus

| Indicateur | Valeur attendue |
|---|---|
| Lignes : brutes → nettoyées | 308 → 300 (8 doublons) |
| Valeurs impossibles corrigées | 3 |
| Étudiants sans note finale (absents en maths) | 6 |
| Taux de réussite (étudiants notés) | ≈ 44 % |
| Variable la plus corrélée à la note finale | `previous_score` (r ≈ 0,59) |
| Groupes C vs A | écart ≈ 0,75 point, p ≈ 0,053 : non significatif |

Fichiers produits dans `outputs/` : `students_clean.csv`, `report.md` et 4 figures PNG.

## Défis pour aller plus loin

- Ajoutez au rapport un intervalle de confiance bootstrap du taux de réussite de chaque groupe.
- Le taux de réussite par activité varie de 35 % à 57 % : ces écarts sont-ils significatifs ? Testez et commentez (pensez aux comparaisons multiples).
- Transformez les règles de nettoyage en contrôles de qualité automatiques (leçon 3.4) exécutés avant l'analyse.
- Exportez le rapport en PDF avec une adaptation de `scripts/export_lessons_to_pdf.py`.

## Solution de référence

`student_analysis/analysis.py` : fonctions `clean_students`, `add_activities`, `summaries`, `group_test`, `make_figures`, `write_report`, `run`.

## Pistes d'amélioration

- Imputer les heures d'étude avec un modèle (régression sur les autres variables) plutôt qu'une médiane.
- Ajouter une analyse multivariée (régression linéaire de la note finale) pour estimer les effets « toutes choses égales par ailleurs ».
- Paramétrer le rapport (seuil de réussite, coefficients) par un fichier de configuration.
