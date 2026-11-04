# Leçon 3.4 — Fusion de tables et contrôles de qualité

## Prérequis

- Leçons 3.1 à 3.3.

## Objectifs d'apprentissage

- Combiner des tables avec `concat` et `merge` en choisissant le bon type de jointure.
- Diagnostiquer une jointure : lignes perdues, lignes dupliquées, clés orphelines.
- Utiliser `validate=` et `indicator=` pour sécuriser les fusions.
- Écrire des contrôles de qualité automatiques et un rapport de qualité.

## Concepts clés

- **Concaténation** (*concatenation*) : empiler des tables qui ont les mêmes colonnes (par exemple les exports de plusieurs mois).
- **Jointure** (*join / merge*) : associer les lignes de deux tables via une **clé** commune.
- **Jointure interne / gauche / externe** (*inner / left / outer*) : garder les clés communes / toutes celles de la table de gauche / toutes.
- **Cardinalité** : un-à-un (1:1), plusieurs-à-un (m:1), plusieurs-à-plusieurs (m:m).
- **Clé orpheline** : clé présente dans une table mais pas dans l'autre.
- **Contrôle de qualité** (*data quality check*) : règle vérifiable automatiquement (unicité, plage, non-nullité, cohérence).

## Explication détaillée

### Concaténer

`pd.concat([df_janvier, df_fevrier], ignore_index=True)` empile des tables. Vérifiez que les colonnes sont identiques (même noms, mêmes types), sinon pandas crée des colonnes remplies de `NaN`. Ajoutez une colonne indiquant la source (`df.assign(source="janvier")`) pour garder la traçabilité.

### Joindre

`pd.merge(gauche, droite, on="cle", how=...)` :

| `how` | Garde | Usage typique |
|---|---|---|
| `inner` | clés présentes des deux côtés | analyse limitée aux cas complets |
| `left` | toutes les lignes de gauche | enrichir une table principale sans perdre de lignes |
| `right` | toutes les lignes de droite | rarement utile (inverser les tables) |
| `outer` | toutes les clés | audit : repérer les orphelins des deux côtés |

Les pièges des jointures sont silencieux :

- **Perte de lignes** : une jointure interne supprime sans prévenir les étudiants sans activité.
- **Multiplication de lignes** : si la clé n'est pas unique dans la table de droite, chaque ligne de gauche est dupliquée autant de fois. Une somme calculée ensuite est fausse.
- **Clés incompatibles** : `"S0001"` contre `"s0001 "`, ou entier contre texte.

Deux options de `merge` protègent contre ces erreurs : `validate="one_to_one"` (ou `"many_to_one"`) lève une erreur si la cardinalité n'est pas celle attendue ; `indicator=True` ajoute une colonne `_merge` qui indique pour chaque ligne si elle vient des deux tables (`both`), seulement de gauche (`left_only`) ou de droite (`right_only`). **Comptez toujours les lignes avant et après une jointure.**

### Contrôles de qualité automatiques

Plutôt que de vérifier « à l'œil », on écrit des règles exécutables :

- **Complétude** : pas de valeur manquante dans les colonnes obligatoires.
- **Unicité** : l'identifiant est unique.
- **Validité** : valeurs dans une plage (`between`), dans un ensemble autorisé (`isin`), au bon format (expression régulière).
- **Cohérence** : relations entre colonnes (la note finale est la moyenne pondérée des notes ; une date de fin est après la date de début).
- **Intégrité référentielle** : toute clé étrangère existe dans la table de référence.

Ces contrôles peuvent être exécutés à chaque arrivée de nouvelles données et intégrés dans des tests `pytest`, exactement comme du code.

## Applications concrètes

- Enrichir une table de ventes avec les informations clients.
- Réunir les exports mensuels d'une application.
- Valider automatiquement un fichier reçu d'un partenaire avant de l'utiliser.

## Exemples de code Python

### Exemple 1 — concaténation et jointures

```python
import pandas as pd

students = pd.read_csv("datasets/generated/students.csv")
activities = pd.read_csv("datasets/generated/student_activities.csv")
print("students:", len(students), "| activities:", len(activities))
print("unique ids in activities:", activities["student_id"].is_unique)

for how in ["inner", "left", "outer"]:
    merged = pd.merge(students, activities, on="student_id", how=how)
    print(f"{how:<6} -> {len(merged)} rows")

audit = pd.merge(students[["student_id"]], activities, on="student_id", how="outer", indicator=True)
print(audit["_merge"].value_counts().to_dict())
print("orphan activity ids:", audit.loc[audit["_merge"] == "right_only", "student_id"].tolist()[:5], "...")

first_half = students.iloc[:150].assign(source="export_1")
second_half = students.iloc[150:].assign(source="export_2")
combined = pd.concat([first_half, second_half], ignore_index=True)
print("concat:", combined.shape, combined["source"].value_counts().to_dict())
```

### Explication du code

- 300 étudiants, 260 lignes d'activités dont 10 avec des identifiants inconnus.
- La jointure interne ne garde que les 250 étudiants qui ont une activité ; la jointure gauche garde les 300 étudiants (activité manquante pour 50) ; la jointure externe ajoute en plus les 10 activités orphelines.
- `indicator=True` permet de compter précisément chaque situation : c'est l'audit à faire avant de choisir le type de jointure.
- `concat` empile deux exports et la colonne `source` garde la trace de leur origine.

### Sortie attendue

```text output
students: 300 | activities: 260
unique ids in activities: True
inner  -> 250 rows
left   -> 300 rows
outer  -> 310 rows
{'both': 250, 'left_only': 50, 'right_only': 10}
orphan activity ids: ['S0901', 'S0902', 'S0903', 'S0904', 'S0905'] ...
concat: (300, 12) {'export_1': 150, 'export_2': 150}
```

### Exemple 2 — le piège de la duplication et `validate`

```python
import pandas as pd
from pandas.errors import MergeError

students = pd.read_csv("datasets/generated/students.csv")
activities = pd.read_csv("datasets/generated/student_activities.csv")

# Simulate an activity table where one student appears twice.
broken = pd.concat([activities, activities[activities["student_id"] == "S0002"]], ignore_index=True)
merged = students.merge(broken, on="student_id", how="left")
print("rows after merge:", len(merged), "(expected 300)")
print("sum of final scores: correct =", round(students["final_score"].sum(), 2), "| after merge =", round(merged["final_score"].sum(), 2))

try:
    students.merge(broken, on="student_id", how="left", validate="one_to_one")
except MergeError as exc:
    print("MergeError:", exc)

safe = students.merge(activities, on="student_id", how="left", validate="one_to_one")
safe["club"] = safe["club"].fillna("inconnu")
print(safe.groupby("club").agg(n=("student_id", "size"), pass_rate=("passed", "mean")).round(2).to_string())
```

### Explication du code

- Une seule clé dupliquée dans la table de droite suffit à créer une ligne de trop et à fausser une somme : l'erreur est **silencieuse**.
- `validate="one_to_one"` transforme cette erreur silencieuse en exception explicite.
- Les étudiants sans ligne d'activité reçoivent la valeur `"inconnu"` : ce n'est pas la même chose que `"aucun"` (l'étudiant a déclaré ne pas avoir d'activité).

### Sortie attendue

```text output
rows after merge: 301 (expected 300)
sum of final scores: correct = 2890.19 | after merge = 2901.82
MergeError: Merge keys are not unique in right dataset; not a one-to-one merge
Duplicates in right:
 student_id
     S0002 ...
            n  pass_rate
club                    
aucun      61       0.57
inconnu    50       0.38
musique    41       0.49
robotique  49       0.35
sport      74       0.41
theatre    25       0.44
```

### Exemple 3 — un rapport de qualité automatique

```python
import pandas as pd


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Run data-quality rules and return one row per rule."""
    checks = {
        "student_id is unique": df["student_id"].is_unique,
        "student_id format S0000": df["student_id"].str.fullmatch(r"S\d{4}").all(),
        "no missing value": not df.isna().any().any(),
        "group in {A, B, C}": df["group"].isin(["A", "B", "C"]).all(),
        "scores in [0, 20]": df[["math", "physics", "french"]].stack().between(0, 20).all(),
        "attendance in [0, 1]": df["attendance_rate"].between(0, 1).all(),
        "final = 0.4m + 0.3p + 0.3f": (
            (0.4 * df["math"] + 0.3 * df["physics"] + 0.3 * df["french"] - df["final_score"]).abs() < 0.01
        ).all(),
        "passed consistent with final": ((df["final_score"] >= 10).astype(int) == df["passed"]).all(),
    }
    return pd.DataFrame({"rule": list(checks), "passed": [bool(v) for v in checks.values()]})


students = pd.read_csv("datasets/generated/students.csv")
print(quality_report(students).to_string(index=False))

corrupted = students.copy()
corrupted.loc[0, "math"] = 25
corrupted.loc[1, "group"] = "D"
report = quality_report(corrupted)
print("failed rules on corrupted data:", report.loc[~report["passed"], "rule"].tolist())
```

### Explication du code

Chaque règle est une expression booléenne lisible. Sur les données propres, toutes passent ; après avoir introduit deux erreurs, le rapport désigne exactement les règles violées. La règle de cohérence vérifie que la note finale est bien la moyenne pondérée : la note de 25 casse aussi cette règle. Ce type de fonction se transforme facilement en test `pytest`.

### Sortie attendue

```text output
                        rule  passed
        student_id is unique    True
     student_id format S0000    True
            no missing value    True
          group in {A, B, C}    True
           scores in [0, 20]    True
        attendance in [0, 1]    True
  final = 0.4m + 0.3p + 0.3f    True
passed consistent with final    True
failed rules on corrupted data: ['group in {A, B, C}', 'scores in [0, 20]', 'final = 0.4m + 0.3p + 0.3f']
```

## Erreurs fréquentes

- **Ne pas compter les lignes avant/après une jointure**.
- **Clé non unique dans la table de référence** : duplication silencieuse.
- **Jointure interne par défaut** (c'est le défaut de `merge` !) : perte silencieuse de lignes.
- **Clés de types différents** (`int` contre `str`) : aucune correspondance, ou erreur.
- **Confondre « valeur manquante après jointure » et « valeur aucun »**.
- **Contrôles de qualité manuels** : non répétables et vite oubliés.

## Exercices guidés

1. Combien d'étudiants ont un club `"aucun"` ? Combien n'ont aucune ligne dans la table des activités ?
2. Calculez la moyenne des heures de bénévolat par groupe d'étudiants, en ne gardant que les étudiants présents dans les deux tables.
3. Ajoutez au rapport de qualité une règle « `sleep_hours` entre 3 et 12 » et testez-la.
4. Concaténez `sales.csv` avec lui-même et écrivez la règle qui détecte le problème.

Corrigés : [corrections.md](corrections.md), section « Leçon 3.4 ».

## Défis autonomes

- Transformez `quality_report` en fichier de tests `pytest` (une fonction de test par règle) qui s'exécute sur `students.csv`.
- Fusionnez `customers.csv` avec `customers_true_segments.csv` et vérifiez l'intégrité référentielle dans les deux sens.

## Quiz

1. Quelle différence entre `concat` et `merge` ?
2. Que se passe-t-il si la clé n'est pas unique dans la table de droite d'une jointure gauche ?
3. À quoi servent `validate=` et `indicator=` ?
4. Quel est le type de jointure par défaut de `pd.merge` ?
5. Citez quatre familles de contrôles de qualité.
6. Pourquoi écrire les contrôles de qualité sous forme de code ?

Réponses : [corrections.md](corrections.md), section « Leçon 3.4 ».

## Résumé

- `concat` empile, `merge` associe par clé ; choisissez `how` consciemment.
- Comptez les lignes, utilisez `validate=` et `indicator=` : les erreurs de jointure sont silencieuses.
- Les contrôles de qualité (complétude, unicité, validité, cohérence, intégrité) s'écrivent en code et s'automatisent.

## Pour aller plus loin

- Exercice **I03** (`exercises/intermediate/I03_groupby_merge`).
- Projet 1 : analyse de la performance des étudiants (`projects/01_student_performance_analysis`).
- Module suivant : [Visualisation de données](../04_data_visualization/README.md).
