# Leçon 10.2 — Déploiement local, surveillance et IA responsable

## Prérequis

- Leçon 10.1 ; leçon 5.6 (pipelines, persistance) ; leçon 2.4 (tests statistiques).

## Objectifs d'apprentissage

- Transformer un modèle entraîné en service de prédiction local fiable : chargement, validation des entrées, prédiction par lots.
- Détecter une dérive des données (*data drift*) avec des tests statistiques simples.
- Mesurer des écarts de performance entre groupes et comprendre les principales notions d'équité.
- Rédiger une fiche modèle (*model card*) qui documente usage prévu, données, performances et limites.
- Identifier les risques : vie privée, sécurité, mauvais usage, données synthétiques.

## Concepts clés

- **Déploiement** (*deployment*) : rendre un modèle utilisable par d'autres (script, service, application).
- **Validation du schéma** (*schema validation*) : vérifier colonnes, types et plages des entrées avant de prédire.
- **Prédiction par lots** (*batch scoring*) : prédire sur un fichier entier, par exemple chaque nuit.
- **Dérive des données** (*data drift*) : les nouvelles entrées ne ressemblent plus aux données d'entraînement. **Dérive de concept** (*concept drift*) : la relation entre entrées et cible change.
- **Indice de stabilité de population** (*Population Stability Index*, PSI) : mesure de l'écart entre deux distributions.
- **Équité** (*fairness*) : absence de disparités injustifiées entre groupes ; plusieurs définitions, parfois incompatibles.
- **Fiche modèle** (*model card*) : document standardisé qui accompagne un modèle.

## Explication détaillée

### Un service de prédiction local

Un modèle en production n'est pas un notebook. Le minimum :

1. **Un artefact versionné** : le pipeline complet (prétraitement + modèle) et ses métadonnées (versions, colonnes attendues, métriques, date).
2. **Une interface claire** : une fonction `predict(records)` et un script en ligne de commande pour traiter un fichier CSV. Dans ce dépôt, aucun serveur externe n'est nécessaire ; le module standard `http.server` suffit pour un service sur `localhost` (c'est ce qu'utilise l'interface d'apprentissage `scripts/run_learning_environment.py`).
3. **La validation des entrées** : colonnes manquantes, types incorrects, valeurs hors de la plage vue à l'entraînement. Un modèle répond toujours quelque chose, même à une absurdité (une maison de −20 m²) : c'est au code de refuser ou d'avertir.
4. **La journalisation** : entrées (sans données personnelles inutiles), prédictions, avertissements, version du modèle ; indispensable pour diagnostiquer et pour surveiller la dérive.

### Surveiller

Les données changent : nouveaux quartiers, inflation, nouveaux types de spam, nouvel appareil photo. On compare régulièrement la distribution des entrées récentes à celle de l'entraînement : test de Kolmogorov-Smirnov pour une variable numérique, PSI par classes (règle usuelle : < 0,1 stable, 0,1 à 0,25 à surveiller, > 0,25 dérive importante), proportions des catégories. Quand les vraies valeurs deviennent disponibles (prix de vente réel), on suit aussi la métrique dans le temps. Une alerte déclenche une analyse, puis éventuellement un réentraînement, avec la même méthodologie d'évaluation que la première fois.

### Équité

Un modèle peut être globalement performant et nettement moins bon pour un groupe : parce que ce groupe est sous-représenté dans les données, parce que les étiquettes historiques reflètent des décisions biaisées, ou parce qu'une variable sert de **proxy** à une caractéristique sensible (le code postal peut refléter l'origine sociale). Mesures courantes par groupe :

- **taux de sélection** (proportion de décisions positives) — parité démographique ;
- **taux de vrais positifs** (rappel) et **taux de faux positifs** — égalité des chances ;
- **précision** par groupe — calibration.

Ces critères ne peuvent généralement pas être tous satisfaits en même temps lorsque les taux de base diffèrent entre groupes : le choix est un choix de **valeurs**, à discuter avec les personnes concernées, pas seulement un réglage technique. Retirer la variable sensible ne suffit pas (les proxies restent) ; il faut mesurer.

### Les autres dimensions d'une IA responsable

- **Vie privée** : collecter le minimum, pseudonymiser, ne pas journaliser de données personnelles inutiles, respecter le RGPD (finalité, durée de conservation, droits des personnes).
- **Sécurité** : ne charger que des modèles de confiance (risque d'exécution de code avec pickle/joblib, leçon 5.6), valider les entrées, limiter qui peut interroger le modèle.
- **Transparence** : fiche modèle, sources citées (RAG, leçon 8.4), explications des décisions importantes.
- **Contrôle humain** : pour les décisions à fort enjeu, le modèle assiste, une personne décide ; prévoir un recours.
- **Honnêteté sur les données** : un modèle entraîné sur des **données synthétiques** (comme tous ceux de ce cours) démontre une méthode ; il ne prouve rien sur le monde réel et ne doit pas être utilisé pour de vraies décisions.

## Intuition mathématique

```text
Kolmogorov-Smirnov : D = max_x |F_entraînement(x) − F_récent(x)|   (écart maximal entre fonctions de répartition)
PSI = Σ_classes (p_récent − p_entraînement) · ln(p_récent / p_entraînement)
taux de sélection du groupe g = P(ŷ = 1 | g)        TPR_g = P(ŷ = 1 | y = 1, g)        FPR_g = P(ŷ = 1 | y = 0, g)
```

## Applications concrètes

- Un script nocturne qui note un fichier de nouvelles demandes, journalise et alerte en cas de dérive.
- Un audit d'équité avant la mise en service d'un modèle de tri de candidatures ou de crédit.
- La fiche modèle jointe à chaque version livrée.

## Exemples de code Python

### Exemple 1 — un prédicteur qui valide ses entrées et traite des lots

```python
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

houses = pd.read_csv("datasets/generated/house_prices.csv")
numeric = ["surface_m2", "rooms", "age_years", "distance_center_km", "has_garden"]
features = numeric + ["neighborhood", "energy_class"]
pipeline = Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("hood", OneHotEncoder(handle_unknown="ignore"), ["neighborhood"]),
        ("energy", OrdinalEncoder(categories=[list("ABCDEFG")]), ["energy_class"]),
    ])),
    ("reg", LinearRegression()),
]).fit(houses[features], np.log(houses["price_keur"]))

model_dir = Path("outputs/module10/price_model")
model_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, model_dir / "model.joblib")
schema = {column: [float(houses[column].min()), float(houses[column].max())] for column in numeric}
metadata = {"version": "1.0.0", "features": features, "ranges": schema,
            "categories": {"neighborhood": sorted(houses["neighborhood"].unique()), "energy_class": list("ABCDEFG")},
            "scikit_learn": sklearn.__version__, "data": "synthetic - not for real decisions"}
(model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


class PricePredictor:
    def __init__(self, folder: Path):
        self.metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata["scikit_learn"] != sklearn.__version__:
            print("WARNING: model trained with another scikit-learn version")
        self.model = joblib.load(folder / "model.joblib")  # only load files you produced yourself

    def validate(self, frame: pd.DataFrame) -> list[str]:
        missing = [c for c in self.metadata["features"] if c not in frame.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        warnings = []
        for column, (low, high) in self.metadata["ranges"].items():
            outside = ~frame[column].between(low, high)
            warnings += [f"row {i}: {column}={frame.at[i, column]} outside training range [{low}, {high}]" for i in frame.index[outside]]
        for column, allowed in self.metadata["categories"].items():
            unknown = ~frame[column].isin(allowed)
            warnings += [f"row {i}: unknown {column} {frame.at[i, column]!r}" for i in frame.index[unknown]]
        return warnings

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        warnings = self.validate(frame)
        result = frame.assign(predicted_price_keur=np.exp(self.model.predict(frame[self.metadata["features"]])).round(1))
        return result, warnings


predictor = PricePredictor(model_dir)
new_houses = pd.DataFrame([
    {"surface_m2": 70, "rooms": 3, "age_years": 15, "distance_center_km": 4.0, "has_garden": 1, "neighborhood": "Residentiel", "energy_class": "C"},
    {"surface_m2": 900, "rooms": 30, "age_years": 5, "distance_center_km": 2.0, "has_garden": 1, "neighborhood": "Centre", "energy_class": "A"},
    {"surface_m2": 55, "rooms": 2, "age_years": 40, "distance_center_km": 8.0, "has_garden": 0, "neighborhood": "Montagne", "energy_class": "F"},
])
predictions, warnings = predictor.predict(new_houses)
print(predictions[["surface_m2", "neighborhood", "predicted_price_keur"]].to_string(index=False))
print("warnings:")
for message in warnings:
    print("  -", message)
```

### Explication du code

- Le pipeline est sauvegardé avec un fichier de métadonnées : version, colonnes, **plages vues à l'entraînement**, catégories connues, version de scikit-learn, et la mention « données synthétiques ».
- `PricePredictor` vérifie la version, valide les entrées et renvoie les prédictions **avec des avertissements**. La maison de 900 m² reçoit une prédiction absurde : 786 500 k€, soit près de 800 millions d'euros. Le modèle entraîné sur le logarithme du prix extrapole de façon **exponentielle** hors de la plage des données ; il « répond » sans aucun signe de doute. Heureusement, la validation la signale. Le quartier « Montagne », inconnu, est ignoré par l'encodeur one-hot (la prédiction reste plausible) et signalé lui aussi.
- En production, ces avertissements seraient journalisés, et les cas signalés transmis à une personne.

### Sortie attendue

```text output
 surface_m2 neighborhood  predicted_price_keur
         70  Residentiel                 261.9
        900       Centre              786500.4
         55     Montagne                 149.0
warnings:
  - row 1: surface_m2=900 outside training range [27.7, 320.0]
  - row 1: rooms=30 outside training range [1.0, 10.0]
  - row 2: unknown neighborhood 'Montagne'
```

### Exemple 2 — détecter une dérive des données

```python
from scipy import stats


def psi(reference: np.ndarray, recent: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    new = np.histogram(recent, edges)[0] / len(recent)
    ref, new = np.clip(ref, 1e-6, None), np.clip(new, 1e-6, None)
    return float(np.sum((new - ref) * np.log(new / ref)))


rng = np.random.default_rng(0)
reference = houses.sample(400, random_state=0)
stable = houses.drop(reference.index).sample(300, random_state=1)
drifted = stable.copy()
drifted["distance_center_km"] = drifted["distance_center_km"] * 1.6 + 2   # buyers move further away
drifted["surface_m2"] = drifted["surface_m2"] * rng.normal(1.0, 0.03, len(drifted))  # tiny change only

for name, batch in [("stable batch", stable), ("drifted batch", drifted)]:
    for column in ["surface_m2", "distance_center_km"]:
        ks = stats.ks_2samp(reference[column], batch[column])
        value = psi(reference[column].to_numpy(), batch[column].to_numpy())
        status = "DRIFT" if value > 0.25 else ("watch" if value > 0.1 else "ok")
        print(f"{name:<14} {column:<19} KS p-value={ks.pvalue:.3f}  PSI={value:.3f}  -> {status}")
```

### Explication du code

- On compare un échantillon de référence (les données d'entraînement) à deux lots récents : l'un tiré de la même distribution, l'autre où l'on a simulé un déplacement des acheteurs vers la périphérie.
- Sur le lot stable, les p-values du test de Kolmogorov-Smirnov sont élevées et le PSI faible : pas d'alerte. Sur le lot modifié, la distance au centre est détectée (PSI élevé, p-value minuscule), tandis que la très légère modification de la surface ne déclenche pas d'alerte.
- Remarquez que, sur le lot **stable**, la distance au centre obtient un PSI de 0,101, juste au-dessus du seuil « à surveiller » : avec 300 maisons, le hasard de l'échantillonnage suffit à franchir ce seuil. Une alerte isolée et faible déclenche une vérification, pas un réentraînement.
- Les seuils du PSI sont des conventions ; avec de très grands lots, le test KS signale des écarts minimes sans importance pratique : on combine toujours ampleur (PSI) et significativité.

### Sortie attendue

```text output
stable batch   surface_m2          KS p-value=0.568  PSI=0.042  -> ok
stable batch   distance_center_km  KS p-value=0.464  PSI=0.101  -> watch
drifted batch  surface_m2          KS p-value=0.464  PSI=0.027  -> ok
drifted batch  distance_center_km  KS p-value=0.000  PSI=2.856  -> DRIFT
```

### Exemple 3 — audit d'équité et fiche modèle

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Synthetic scenario: group B is under-represented and its useful signal is noisier.
rng = np.random.default_rng(42)
n = 4000
group = rng.choice(["A", "B"], size=n, p=[0.85, 0.15])
skill = rng.normal(0, 1, n)
noise_scale = np.where(group == "A", 0.5, 1.5)          # measurement is less reliable for group B
test_score = skill + rng.normal(0, 1, n) * noise_scale
outcome = (skill + rng.normal(0, 0.5, n) > 0).astype(int)
frame = pd.DataFrame({"group": group, "test_score": test_score, "outcome": outcome})

train, test = train_test_split(frame, test_size=0.4, stratify=frame["group"], random_state=0)
clf = LogisticRegression().fit(train[["test_score"]], train["outcome"])
test = test.assign(pred=clf.predict(test[["test_score"]]))


def group_metrics(df):
    positives, negatives = df[df["outcome"] == 1], df[df["outcome"] == 0]
    return pd.Series({
        "n": len(df),
        "accuracy": (df["pred"] == df["outcome"]).mean(),
        "selection_rate": df["pred"].mean(),
        "true_positive_rate": positives["pred"].mean(),
        "false_positive_rate": negatives["pred"].mean(),
    })


audit = test.groupby("group")[["outcome", "pred"]].apply(group_metrics)
print(audit.round(3).to_string())
print("overall accuracy:", round((test["pred"] == test["outcome"]).mean(), 3))

card = f"""# Fiche modèle - classifieur de démonstration (données synthétiques)

- **Usage prévu** : démonstration pédagogique d'un audit d'équité. Ne pas utiliser pour des décisions réelles.
- **Données** : {n} exemples synthétiques générés localement ; groupe B = {(frame['group'] == 'B').mean():.0%} des données.
- **Performance globale (test)** : accuracy {(test['pred'] == test['outcome']).mean():.3f}.
- **Performance par groupe** : accuracy A = {audit.loc['A', 'accuracy']:.3f}, B = {audit.loc['B', 'accuracy']:.3f} ; rappel A = {audit.loc['A', 'true_positive_rate']:.3f}, B = {audit.loc['B', 'true_positive_rate']:.3f}.
- **Limites connues** : la variable d'entrée est moins fiable pour le groupe B, sous-représenté ; les erreurs y sont plus fréquentes.
- **Recommandations** : collecter une mesure plus fiable pour B, présenter les résultats par groupe, garder une décision humaine.
"""
Path("outputs/module10").mkdir(parents=True, exist_ok=True)
Path("outputs/module10/model_card.md").write_text(card, encoding="utf-8")
print("model card written to outputs/module10/model_card.md")
```

### Explication du code

- Le scénario est **synthétique et volontairement construit** : la compétence réelle a la même distribution dans les deux groupes, mais la mesure disponible (`test_score`) est beaucoup plus bruitée pour le groupe B, minoritaire. C'est un mécanisme réaliste (outil de mesure mal adapté à une population).
- La précision globale paraît correcte, mais l'audit par groupe montre une précision nettement plus faible pour B, et des taux de vrais et de faux positifs différents : les erreurs ne sont pas réparties équitablement.
- La fiche modèle consigne ces résultats et les limites, avec les recommandations. Le remède n'est pas technique seulement : améliorer la mesure, collecter des données, décider des critères d'équité avec les personnes concernées.

### Sortie attendue

```text output
            n  accuracy  selection_rate  true_positive_rate  false_positive_rate
group                                                                           
A      1356.0     0.791           0.472               0.762                0.179
B       244.0     0.652           0.451               0.593                0.275
overall accuracy: 0.77
model card written to outputs/module10/model_card.md
```

## Erreurs fréquentes

- **Déployer un notebook** au lieu d'un artefact versionné et d'une fonction testée.
- **Ne pas valider les entrées** : le modèle prédit n'importe quoi sur n'importe quoi.
- **Ne jamais surveiller** après la mise en service.
- **Supprimer la variable sensible et déclarer le modèle équitable** sans mesurer par groupe.
- **Journaliser des données personnelles** sans nécessité.
- **Présenter des résultats sur données synthétiques comme une preuve d'efficacité réelle**.

## Exercices guidés

1. Ajoutez à `PricePredictor.validate` une vérification des valeurs manquantes.
2. Écrivez un script de prédiction par lots qui lit un CSV, écrit un CSV de prédictions et un fichier texte d'avertissements.
3. Calculez le PSI de `age_years` entre les maisons du centre et les maisons rurales. Interprétez.
4. Dans l'audit d'équité, que se passe-t-il si le bruit de mesure est le même pour les deux groupes ?

Corrigés : [corrections.md](corrections.md), section « Leçon 10.2 ».

## Défis autonomes

- Exposez `PricePredictor` via un petit serveur `http.server` sur `127.0.0.1` qui reçoit du JSON et renvoie la prédiction (inspirez-vous de `scripts/run_learning_environment.py`).
- Rédigez la fiche modèle de votre projet 6 en suivant le modèle de l'exemple 3.

## Quiz

1. Que doit contenir un artefact de modèle prêt à être utilisé ?
2. Pourquoi valider les entrées si le modèle « sait » prédire n'importe quelles valeurs ?
3. Différence entre dérive des données et dérive de concept ?
4. Que mesure le PSI ? Quels seuils sont habituellement utilisés ?
5. Pourquoi retirer une variable sensible ne garantit-il pas l'équité ?
6. Que doit indiquer une fiche modèle ?

Réponses : [corrections.md](corrections.md), section « Leçon 10.2 ».

## Résumé

- Déployer = artefact versionné + interface claire + validation des entrées + journalisation.
- Surveiller les entrées (KS, PSI) et la performance ; réentraîner avec la même rigueur.
- Mesurer les performances par groupe ; l'équité est un choix de valeurs documenté.
- Fiche modèle, vie privée, sécurité, contrôle humain, honnêteté sur les données.

## Pour aller plus loin

- Projet 6 (`projects/06_end_to_end_ml_project`).
- Retour au [parcours d'apprentissage](../../docs/learning_roadmap.md).
