# Leçon 3.2 — Valeurs aberrantes, transformations et création de variables

## Prérequis

- Leçon 3.1 (nettoyage), leçon 2.3 (quantiles, écart-type).

## Objectifs d'apprentissage

- Détecter des valeurs aberrantes avec la règle de l'IQR et les z-scores.
- Décider quoi faire d'une valeur aberrante : corriger, supprimer, borner ou conserver.
- Transformer des variables : standardisation, normalisation min-max, logarithme, discrétisation.
- Encoder des variables catégorielles (one-hot).
- Créer des variables dérivées pertinentes (*feature engineering*).

## Concepts clés

- **Valeur aberrante** (*outlier*) : observation très éloignée de la masse des données. Elle peut être une **erreur** (saisie, unité) ou une **valeur rare mais réelle**.
- **Règle de Tukey (IQR)** : suspecte si en dehors de [Q1 − 1,5·IQR ; Q3 + 1,5·IQR].
- **z-score** : (x − moyenne) / écart-type ; |z| > 3 est souvent jugé suspect pour des données en cloche.
- **Écrêtage** (*clipping*, *winsorizing*) : remplacer les valeurs extrêmes par une borne.
- **Standardisation** (*standardization*) : moyenne 0, écart-type 1. **Normalisation min-max** : ramener entre 0 et 1.
- **Encodage one-hot** (*one-hot encoding*) : une colonne binaire par catégorie.
- **Ingénierie des variables** (*feature engineering*) : créer des variables plus informatives à partir des variables existantes.

## Explication détaillée

### Détecter les valeurs aberrantes

La **règle de l'IQR** est robuste : elle s'appuie sur les quartiles, qui ne sont pas influencés par les valeurs extrêmes. Le **z-score** suppose une distribution à peu près normale et est lui-même influencé par les extrêmes (ils gonflent l'écart-type et peuvent ainsi se « masquer »). Pour une variable très asymétrique (prix, revenus), on peut appliquer la règle de l'IQR sur le **logarithme** de la variable.

### Décider

Détecter n'est pas supprimer. Posez-vous trois questions :

1. **Est-ce physiquement possible ?** Un appartement de 3 m² ou un âge de 230 ans : erreur → corriger si possible (unité, virgule déplacée), sinon mettre à `NaN`.
2. **Est-ce rare mais réel ?** Une maison de luxe à 1 100 k€ : c'est une information. La supprimer rendrait le modèle incapable de traiter ces cas.
3. **Quel est l'objectif ?** Pour une moyenne descriptive, un écrêtage limite l'influence ; pour un modèle d'arbres (module 5), les valeurs extrêmes posent peu de problèmes ; pour une régression linéaire, elles peuvent tirer la droite.

Documentez toujours combien de valeurs ont été modifiées et pourquoi.

### Transformer

- **Standardisation** z = (x − μ)/σ : indispensable pour les méthodes basées sur des distances (k-NN, k-means, SVM) et la descente de gradient.
- **Min-max** (x − min)/(max − min) : utile pour des valeurs bornées (pixels d'une image entre 0 et 1). Sensible aux extrêmes.
- **Logarithme** `np.log1p(x)` : réduit l'asymétrie à droite ; les écarts relatifs deviennent des écarts absolus.
- **Discrétisation** (`pd.cut`, `pd.qcut`) : transformer une variable continue en classes (tranches d'âge). Perte d'information, mais parfois plus lisible.

### Encoder les catégories

Les modèles travaillent sur des nombres. Coder `Centre=0, Résidentiel=1, Périphérie=2, Rural=3` impose un ordre et des distances artificielles. Le **one-hot** crée une colonne binaire par catégorie (`pd.get_dummies`). Pour une variable **ordinale** (classe énergétique A à G), un codage ordinal (A=1... G=7) est au contraire pertinent car l'ordre a un sens.

### Créer des variables

Le *feature engineering* injecte de la connaissance du domaine :

- **Ratios** : prix au m², moyenne des notes, taux de remise ;
- **Différences** : progression = note finale − note précédente ;
- **Dates** : mois, jour de la semaine, week-end, saisonnalité ;
- **Interactions** : surface × quartier (le prix au m² dépend du quartier) ;
- **Agrégats** : nombre de commandes par client, panier moyen.

Une bonne variable est souvent plus utile qu'un modèle plus complexe. Mais attention : une variable construite à partir de la **cible** (par exemple le prix au m² quand on veut prédire le prix) est une fuite de données.

## Intuition mathématique

```text
IQR = Q3 − Q1          bornes de Tukey : [Q1 − 1,5·IQR ; Q3 + 1,5·IQR]
z = (x − μ) / σ        pour une loi normale, P(|z| > 3) ≈ 0,27 %
min-max : x' = (x − min) / (max − min) ∈ [0, 1]
log : log(a·b) = log a + log b  → un écart de +10 % devient un écart constant
```

## Applications concrètes

- Détection de fraude : les transactions aberrantes sont précisément celles qui intéressent.
- Immobilier : prix au m² par quartier comme variable clé.
- Ventes : variables calendaires (mois, week-end, fêtes) pour prévoir la demande.

## Exemples de code Python

### Exemple 1 — IQR et z-scores sur les prix immobiliers

```python
import numpy as np
import pandas as pd

houses = pd.read_csv("datasets/generated/house_prices.csv")
price = houses["price_keur"]

q1, q3 = price.quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
iqr_outliers = houses[(price < low) | (price > high)]
print(f"Tukey bounds: [{low:.1f}, {high:.1f}] -> {len(iqr_outliers)} outliers")

z = (price - price.mean()) / price.std()
print("|z| > 3:", int((z.abs() > 3).sum()))

log_price = np.log(price)
lq1, lq3 = log_price.quantile([0.25, 0.75])
lim = 1.5 * (lq3 - lq1)
print("outliers on log scale:", int(((log_price < lq1 - lim) | (log_price > lq3 + lim)).sum()))
print(iqr_outliers.sort_values("price_keur", ascending=False).head(3)[["surface_m2", "neighborhood", "price_keur"]].to_string(index=False))
print("skew raw:", round(price.skew(), 2), "| skew log:", round(log_price.skew(), 2))
```

### Explication du code

Sur l'échelle brute, la règle de l'IQR signale 26 maisons chères. Ce ne sont pas des erreurs : les plus chères sont de très grandes surfaces (jusqu'à 320 m²) ou des logements du centre-ville, au prix au m² élevé. Sur l'échelle logarithmique, la distribution est presque symétrique (asymétrie proche de 0) et beaucoup moins de points sont signalés : la plupart des « aberrations » apparentes venaient seulement de l'asymétrie de la distribution.

### Sortie attendue

```text output
Tukey bounds: [-69.5, 607.7] -> 26 outliers
|z| > 3: 8
outliers on log scale: 3
 surface_m2 neighborhood  price_keur
      320.0  Residentiel      1106.4
      239.0       Centre       949.8
      183.3       Centre       932.5
skew raw: 1.4 | skew log: -0.06
```

### Exemple 2 — écrêtage, standardisation, min-max, discrétisation

```python
import pandas as pd

houses = pd.read_csv("datasets/generated/house_prices.csv")
surface = houses["surface_m2"]

clipped = surface.clip(lower=surface.quantile(0.01), upper=surface.quantile(0.99))
print("max before/after clipping:", surface.max(), "->", round(clipped.max(), 1))

standardized = (surface - surface.mean()) / surface.std(ddof=0)
min_max = (surface - surface.min()) / (surface.max() - surface.min())
print("standardized mean/std:", round(standardized.mean(), 6), round(standardized.std(ddof=0), 6))
print("min-max range:", min_max.min(), "-", min_max.max())

houses["size_class"] = pd.cut(surface, bins=[0, 50, 90, 150, 1000], labels=["petit", "moyen", "grand", "très grand"])
houses["surface_quartile"] = pd.qcut(surface, q=4, labels=["Q1", "Q2", "Q3", "Q4"])
print(houses["size_class"].value_counts().sort_index().to_string())
print(houses["surface_quartile"].value_counts().sort_index().to_dict())
```

### Explication du code

`clip` borne les surfaces au 1er et au 99e percentile. La standardisation donne une moyenne 0 et un écart-type 1 ; la normalisation min-max ramène entre 0 et 1. `pd.cut` utilise des bornes choisies (classes de tailles inégales en effectif), `pd.qcut` des bornes calculées pour obtenir des classes de même effectif (200 maisons par quartile).

### Sortie attendue

```text output
max before/after clipping: 320.0 -> 203.9
standardized mean/std: -0.0 1.0
min-max range: 0.0 - 1.0
size_class
petit          69
moyen         385
grand         285
très grand     61
{'Q1': 200, 'Q2': 201, 'Q3': 199, 'Q4': 200}
```

### Exemple 3 — encodage et création de variables

```python
import numpy as np
import pandas as pd

houses = pd.read_csv("datasets/generated/house_prices.csv")
energy_order = {letter: rank for rank, letter in enumerate("ABCDEFG", start=1)}
houses["energy_rank"] = houses["energy_class"].map(energy_order)
houses["log_distance"] = np.log1p(houses["distance_center_km"])
houses["is_new"] = (houses["age_years"] <= 5).astype(int)
houses["surface_per_room"] = (houses["surface_m2"] / houses["rooms"]).round(1)

encoded = pd.get_dummies(houses, columns=["neighborhood"], prefix="hood", dtype=int)
print([c for c in encoded.columns if c.startswith("hood_")])
print(encoded.filter(like="hood_").sum().to_dict())

features = ["surface_m2", "energy_rank", "log_distance", "is_new", "surface_per_room", "price_keur"]
print(houses[features].corr()["price_keur"].round(3).sort_values(ascending=False).to_string())
```

### Explication du code

- `energy_rank` est un codage **ordinal** : A=1 (meilleur) à G=7.
- `log1p` (log(1 + x)) gère les distances proches de 0.
- `get_dummies` remplace la colonne `neighborhood` par 4 colonnes binaires ; chaque maison a un 1 dans exactement une d'entre elles.
- La corrélation avec le prix montre que la surface domine (0,77) et que la distance au centre est associée à des prix plus bas (−0,50).
- La classe énergétique a une corrélation très faible (−0,07) alors que le générateur lui donne un vrai effet (jusqu'à ±14 % du prix). Cet effet est masqué par les variations de surface et de quartier, beaucoup plus importantes : une corrélation simple ne mesure pas l'effet « toutes choses égales par ailleurs ». Un modèle multivarié (module 5) saura l'isoler.

### Sortie attendue

```text output
['hood_Centre', 'hood_Peripherie', 'hood_Residentiel', 'hood_Rural']
{'hood_Centre': 174, 'hood_Peripherie': 234, 'hood_Residentiel': 281, 'hood_Rural': 111}
price_keur          1.000
surface_m2          0.767
is_new              0.026
surface_per_room   -0.064
energy_rank        -0.070
log_distance       -0.502
```

## Erreurs fréquentes

- **Supprimer automatiquement toutes les valeurs hors IQR** : vous perdez des cas réels et importants.
- **Calculer les bornes (quantiles, moyenne, écart-type) sur tout le jeu de données avant la séparation train/test** en ML : fuite de données.
- **Encoder des catégories nominales par des entiers** : ordre artificiel.
- **Créer une variable à partir de la cible** (prix au m² pour prédire le prix).
- **Logarithme de valeurs nulles ou négatives** : utilisez `log1p` ou décalez les valeurs.
- **One-hot sur une variable à des milliers de catégories** (identifiants) : explosion du nombre de colonnes.

## Exercices guidés

1. Appliquez la règle de l'IQR à `study_hours_week` de `students.csv`. Combien d'étudiants sont signalés ? Ces valeurs sont-elles plausibles ?
2. Créez dans `students.csv` une variable `progress = final_score - previous_score` et calculez sa moyenne par groupe.
3. Dans `sales.csv`, créez les variables `month`, `day_of_week` et `is_weekend` à partir de la date.
4. Encodez `product` en one-hot dans `sales.csv` et vérifiez que la somme de chaque ligne vaut 1.

Corrigés : [corrections.md](corrections.md), section « Leçon 3.2 ».

## Défis autonomes

- Écrivez une fonction `flag_outliers(df, columns, method="iqr")` qui ajoute une colonne booléenne par variable et renvoie un résumé du nombre de valeurs signalées.
- Comparez la corrélation entre `surface_m2` et `price_keur` avant et après transformation logarithmique des deux variables. Interprétez.

## Quiz

1. Quelle différence entre une valeur aberrante due à une erreur et une valeur extrême réelle ? Que faire dans chaque cas ?
2. Pourquoi la règle de l'IQR est-elle plus robuste que le z-score ?
3. Quand préférer un codage ordinal au one-hot ?
4. Pourquoi standardiser les variables avant un k-NN ?
5. Donnez un exemple de variable qui créerait une fuite de données.
6. Que fait `pd.qcut(x, 4)` ?

Réponses : [corrections.md](corrections.md), section « Leçon 3.2 ».

## Résumé

- Détecter (IQR, z-score, échelle log) puis décider (corriger, borner, garder) en documentant.
- Standardiser pour les méthodes à distances et à gradient ; log pour réduire l'asymétrie.
- One-hot pour les catégories nominales, codage ordinal pour les catégories ordonnées.
- Le feature engineering injecte la connaissance du domaine, sans jamais utiliser la cible.

## Pour aller plus loin

- Exercice **I02** — valeurs manquantes et aberrantes.
- Module 5, leçon 5.6 : intégrer ces transformations dans un `Pipeline` scikit-learn pour éviter les fuites.
- Leçon suivante : [Analyse exploratoire, agrégations et statistiques descriptives](03_eda_groupby.md).
