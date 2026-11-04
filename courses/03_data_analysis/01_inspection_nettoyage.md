# Leçon 3.1 — Inspection et nettoyage des données

## Prérequis

- Leçon 1.5 (pandas), exercice B03 conseillé.
- Jeux de données générés (`python scripts/generate_datasets.py`).

## Objectifs d'apprentissage

- Mener un diagnostic systématique d'un fichier inconnu.
- Détecter et corriger les problèmes de types, de formats et de catégories incohérentes.
- Comprendre les mécanismes de données manquantes et choisir une stratégie de traitement.
- Identifier et supprimer les doublons de façon contrôlée.
- Documenter chaque transformation pour la rendre reproductible.

## Concepts clés

- **Données brutes** (*raw data*) : données telles qu'elles ont été collectées ; on ne les modifie **jamais** sur place.
- **Valeur manquante** (*missing value*, `NaN`, `None`, `NA`) : absence d'information.
- **MCAR / MAR / MNAR** : manquant complètement au hasard / au hasard conditionnellement à d'autres variables / non au hasard (lié à la valeur elle-même).
- **Imputation** : remplacement d'une valeur manquante par une estimation (médiane, mode, modèle).
- **Doublon** (*duplicate*) : enregistrement répété, exact ou partiel (même identifiant).
- **Conversion de type** (*type casting*) : transformer une colonne texte en nombre, date ou catégorie.
- **Pipeline de nettoyage** : suite ordonnée et reproductible de transformations.

## Explication détaillée

### Règle d'or : ne jamais modifier les données brutes

Gardez `datasets/raw/` intact et écrivez le résultat dans `datasets/processed/` ou `outputs/`. Tout le nettoyage doit être fait par **du code**, pas à la main dans un tableur : ainsi, si de nouvelles données arrivent ou si vous découvrez une erreur, il suffit de relancer le script.

### Le diagnostic en six questions

1. **Forme** : combien de lignes et de colonnes ? (`shape`)
2. **Types** : chaque colonne a-t-elle le type attendu ? (`dtypes`, `info()`) Une colonne de notes de type texte cache des valeurs non numériques.
3. **Valeurs manquantes** : combien par colonne ? (`isna().sum()`) Sont-elles concentrées sur certaines lignes ?
4. **Doublons** : lignes identiques (`duplicated()`), identifiants répétés (`duplicated(subset="id")`).
5. **Valeurs uniques des catégories** : `value_counts()` révèle les fautes de frappe, les casses et espaces différents (`"A"`, `" a "`).
6. **Plages de valeurs** : `describe()` révèle les valeurs impossibles (120 heures d'étude par semaine, un âge négatif).

### Corriger les types et formats

- Texte → nombre : `pd.to_numeric(serie, errors="coerce")` transforme les valeurs non convertibles en `NaN`. **Comptez** les `NaN` avant et après pour savoir combien de valeurs ont été perdues, et examinez-les.
- Formats particuliers : `"87.0%"` doit devenir `0.87` ; `"19,90 EUR"` doit devenir `19.9`. On utilise les méthodes `.str` (`strip`, `lower`, `replace`, `removesuffix`).
- Catégories : normaliser la casse et les espaces (`.str.strip().str.upper()`), puis vérifier que l'ensemble des valeurs est celui attendu.
- Dates : `pd.to_datetime(..., errors="coerce")`.

### Valeurs manquantes : comprendre avant d'agir

La bonne stratégie dépend de **pourquoi** la valeur manque :

- **MCAR** (un capteur tombe en panne au hasard) : supprimer les lignes est sans biais, mais réduit la taille de l'échantillon.
- **MAR** (les étudiants d'un groupe ont moins souvent rempli le questionnaire) : l'imputation qui tient compte des autres variables (médiane par groupe) limite le biais.
- **MNAR** (les personnes qui dorment peu ne déclarent pas leurs heures de sommeil) : aucune méthode simple ne corrige le biais ; il faut le signaler. Ajouter une colonne indicatrice `sleep_hours_missing` permet au modèle d'exploiter cette information.

Stratégies courantes :

| Stratégie | Quand | Risque |
|---|---|---|
| supprimer les lignes (`dropna`) | peu de lignes concernées, MCAR | perte d'information, biais si non MCAR |
| supprimer la colonne | colonne majoritairement vide | perte d'une variable |
| imputer par la médiane | variable numérique asymétrique | réduit artificiellement la variance |
| imputer par le mode | variable catégorielle | sur-représente la catégorie dominante |
| imputer par groupe | la valeur dépend d'une catégorie | nécessite un groupe pertinent |
| indicateur + imputation | le fait d'être manquant est informatif | une colonne de plus |

**Point crucial pour le ML** : les valeurs d'imputation (médianes...) doivent être calculées sur les données d'**entraînement** seulement, puis appliquées au test. Sinon, on crée une fuite de données (module 5). Dans cette leçon d'analyse, on travaille sur l'ensemble du fichier.

### Doublons

Un doublon exact (toutes les colonnes identiques) provient généralement d'un export répété : on le supprime. Un doublon partiel (même identifiant, valeurs différentes) est plus délicat : il faut une règle métier (garder la plus récente, la plus complète...). Comptez toujours avant de supprimer.

## Applications concrètes

- Fusion d'exports de plusieurs logiciels aux conventions différentes (décimales, dates, casse).
- Questionnaires avec réponses manquantes ou incohérentes.
- Préparation des données avant tout modèle de ML : la plupart des algorithmes de scikit-learn refusent les `NaN`.

## Exemples de code Python

### Exemple 1 — diagnostic d'un fichier brut

```python
import pandas as pd

raw = pd.read_csv("datasets/raw/students_raw.csv")
print("shape:", raw.shape)
print(raw.dtypes.astype(str).to_string())
print("\nmissing values per column:")
print(raw.isna().sum()[raw.isna().sum() > 0].to_string())
print("\nexact duplicated rows:", raw.duplicated().sum(), "| duplicated ids:", raw["student_id"].duplicated().sum())
print("\ngroup values:", raw["group"].value_counts().to_dict())
non_numeric_math = raw.loc[pd.to_numeric(raw["math"], errors="coerce").isna(), "math"]
print("non numeric math values:", non_numeric_math.unique().tolist())
print("attendance examples:", raw["attendance_rate"].head(8).tolist())
print(raw["study_hours_week"].describe().round(2).to_string())
```

### Explication du code

Le diagnostic révèle tous les problèmes en quelques lignes : les colonnes `attendance_rate` et `math` sont de type texte (`str`) au lieu de numériques ; trois colonnes ont des valeurs manquantes ; 8 lignes sont des doublons exacts ; le groupe apparaît sous des formes `"a"`, `" b "`... ; certaines notes valent `"absent"` ; l'assiduité est parfois écrite en pourcentage ; enfin `describe()` montre une valeur minimale négative et une maximale de 120 heures, impossibles.

### Sortie attendue

```text output
shape: (308, 9)
student_id              str
group                   str
study_hours_week    float64
attendance_rate         str
sleep_hours         float64
previous_score      float64
math                    str
physics             float64
french              float64

missing values per column:
study_hours_week    18
sleep_hours         15
physics              9

exact duplicated rows: 8 | duplicated ids: 8

group values: {'B': 113, 'C': 87, 'A': 83, 'c': 7, ' c ': 4, 'b': 4, ' a ': 4, ' b ': 4, 'a': 2}
non numeric math values: ['absent']
attendance examples: ['0.973', '0.735', '0.908', '0.892', '0.655', '90.4%', '0.884', '0.896']
count    290.00
mean      10.53
std        9.66
min       -3.00
25%        6.10
50%        9.30
75%       12.70
max      120.00
```

### Exemple 2 — corriger types, formats et catégories

```python
import numpy as np
import pandas as pd

raw = pd.read_csv("datasets/raw/students_raw.csv")
df = raw.drop_duplicates().copy()

df["group"] = df["group"].str.strip().str.upper()


def parse_rate(value):
    """'87.5%' -> 0.875 ; '0.91' -> 0.91 ; NaN stays NaN."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.endswith("%"):
        return float(text.removesuffix("%")) / 100
    return float(text)


df["attendance_rate"] = df["attendance_rate"].map(parse_rate)
df["math_absent"] = df["math"].eq("absent")
df["math"] = pd.to_numeric(df["math"], errors="coerce")

impossible = ~df["study_hours_week"].between(0, 80)
print("impossible study hours:", df.loc[impossible & df["study_hours_week"].notna(), "study_hours_week"].tolist())
df.loc[impossible, "study_hours_week"] = np.nan

print("rows:", len(raw), "->", len(df))
print("groups:", sorted(df["group"].unique()))
print("attendance range:", df["attendance_rate"].min(), "-", df["attendance_rate"].max())
print("dtypes:", df[["attendance_rate", "math", "study_hours_week"]].dtypes.astype(str).to_dict())
print(df.isna().sum()[df.isna().sum() > 0].to_string())
```

### Explication du code

- `drop_duplicates()` retire les 8 doublons exacts **avant** tout calcul.
- La normalisation `strip().upper()` ramène les groupes à `A`, `B`, `C`.
- `parse_rate` gère les deux formats de l'assiduité. On applique la fonction à chaque valeur avec `map`.
- On **garde la trace** des absences dans une colonne booléenne `math_absent` avant de convertir `"absent"` en `NaN` : l'information n'est pas perdue.
- `between(0, 80)` définit une plage plausible ; les valeurs impossibles deviennent manquantes (on ne sait pas quelle était la vraie valeur). Remarquez que `~between` est aussi vrai pour les `NaN` déjà présents, d'où le filtre `notna()` pour l'affichage.

### Sortie attendue

```text output
impossible study hours: [95.0, -3.0, 120.0]
rows: 308 -> 300
groups: ['A', 'B', 'C']
attendance range: 0.408 - 0.999
dtypes: {'attendance_rate': 'float64', 'math': 'float64', 'study_hours_week': 'float64'}
study_hours_week    21
sleep_hours         15
math                 6
physics              9
```

### Exemple 3 — imputer en gardant la trace

```python
for column in ["study_hours_week", "sleep_hours", "physics", "math"]:
    df[f"{column}_was_missing"] = df[column].isna()

df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].median())
df["study_hours_week"] = df["study_hours_week"].fillna(
    df.groupby("group")["study_hours_week"].transform("median")
)
df["physics"] = df["physics"].fillna(df["physics"].median())
print("remaining missing values:", int(df[["sleep_hours", "study_hours_week", "physics"]].isna().sum().sum()))
print("math still missing (absent students, kept as NaN):", int(df["math"].isna().sum()))
print(df.filter(like="_was_missing").sum().to_string())
```

### Explication du code

- Les colonnes `*_was_missing` mémorisent quelles valeurs ont été imputées : indispensable pour l'analyse et souvent utile à un modèle.
- `groupby("group")[...].transform("median")` renvoie, pour chaque ligne, la médiane de **son** groupe : une imputation conditionnelle (stratégie MAR).
- On choisit de **ne pas** imputer la note de mathématiques des absents : inventer une note serait trompeur. La décision dépend de l'usage (exclure ces étudiants d'une moyenne de classe, par exemple).

### Sortie attendue

```text output
remaining missing values: 0
math still missing (absent students, kept as NaN): 6
study_hours_week_was_missing    21
sleep_hours_was_missing         15
physics_was_missing              9
math_was_missing                 6
```

## Erreurs fréquentes

- **Nettoyer à la main** dans un tableur : non reproductible.
- **`to_numeric(errors="coerce")` sans compter les `NaN` créés** : des données valides au format inattendu disparaissent silencieusement.
- **Imputer avant de séparer entraînement et test** en ML : fuite de données.
- **Imputer la moyenne d'une variable asymétrique** : préférez la médiane.
- **Supprimer toutes les lignes incomplètes** (`dropna()` sans argument) : on peut perdre une grande partie des données.
- **Supprimer les doublons après avoir calculé des statistiques** : les résultats sont faussés.

## Exercices guidés

1. Dans `students_raw.csv`, quelle proportion de lignes a au moins une valeur manquante ? (*indice : `isna().any(axis=1).mean()`*)
2. Comparez la moyenne de `sleep_hours` avant et après imputation par la médiane, puis l'écart-type. Lequel change le plus ? Pourquoi ?
3. Vérifiez que, après nettoyage, chaque `student_id` est unique.
4. Écrivez une fonction `normalize_category(series)` qui supprime les espaces, met en majuscules et remplace les chaînes vides par `NaN`.

Corrigés : [corrections.md](corrections.md), section « Leçon 3.1 ».

## Défis autonomes

- Le taux de valeurs manquantes de `study_hours_week` diffère-t-il selon le groupe ? Que pourriez-vous en conclure sur le mécanisme (MCAR/MAR) ?
- Écrivez une fonction `clean_students(raw) -> (clean, report)` qui réalise tout le nettoyage et renvoie un rapport chiffré de chaque étape (c'est l'exercice I02).

## Quiz

1. Pourquoi ne faut-il jamais modifier les données brutes ?
2. Que fait `pd.to_numeric(s, errors="coerce")` sur la valeur `"absent"` ?
3. Différence entre MCAR et MNAR ? Donnez un exemple de chaque.
4. Pourquoi ajouter une colonne indicatrice avant d'imputer ?
5. Pourquoi supprimer les doublons avant de calculer des statistiques ?
6. Pourquoi la médiane est-elle souvent préférée à la moyenne pour imputer ?

Réponses : [corrections.md](corrections.md), section « Leçon 3.1 ».

## Résumé

- Diagnostic systématique : forme, types, manquants, doublons, catégories, plages.
- Corriger formats et types avec les méthodes `.str`, `map` et `to_numeric(errors="coerce")`, en comptant ce qui est perdu.
- Choisir le traitement des manquants selon leur mécanisme ; garder une trace des imputations.
- Tout le nettoyage est fait par du code, à partir de données brutes intactes.

## Pour aller plus loin

- Exercice **I02** (`exercises/intermediate/I02_missing_values_outliers`).
- Leçon suivante : [Valeurs aberrantes, transformations et création de variables](02_outliers_transformation_features.md).
