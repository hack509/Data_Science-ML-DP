# Corrigés — Module 5 : Machine Learning

## Leçon 5.1

### Quiz

1. L'IA regroupe toutes les techniques qui imitent des capacités intelligentes (y compris des règles écrites à la main) ; le ML apprend ses règles à partir d'exemples ; le DL est la partie du ML qui utilise des réseaux de neurones profonds.
2. Régression (nombre de jours, une quantité).
3. La validation sert à comparer des modèles et régler les hyperparamètres ; le test sert une seule fois, à la fin, à estimer la performance sur des données jamais utilisées.
4. Surapprentissage (variance élevée). Remèdes : modèle plus simple, régularisation, plus de données, moins de variables.
5. La moyenne et l'écart-type utilisés contiennent de l'information des données de test, qui influence alors l'entraînement.
6. Un réglage choisi avant l'entraînement et non appris : `max_depth` d'un arbre, `k` du k-NN, `C` d'une régression logistique.

### Exercices guidés

```python
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

houses = pd.read_csv("datasets/generated/house_prices.csv")
train, test = train_test_split(houses, test_size=0.2, random_state=0)
print("mean price train/test:", round(train["price_keur"].mean(), 1), round(test["price_keur"].mean(), 1))

rng = np.random.default_rng(0)
x = np.sort(rng.uniform(-3, 3, 40))
y = np.sin(x) + rng.normal(0, 0.3, 40)
x_val = np.sort(rng.uniform(-3, 3, 200))
y_val = np.sin(x_val) + rng.normal(0, 0.3, 200)
for degree in [1, 2, 3, 4, 8, 12, 15]:
    model = make_pipeline(PolynomialFeatures(degree), StandardScaler(), LinearRegression()).fit(x.reshape(-1, 1), y)
    print(degree, round(mean_squared_error(y, model.predict(x.reshape(-1, 1))), 3),
          round(mean_squared_error(y_val, model.predict(x_val.reshape(-1, 1))), 3))

binary = pd.read_csv("datasets/generated/binary_classification.csv")
dummy = DummyClassifier(strategy="most_frequent").fit(binary.drop(columns="target"), binary["target"])
print("majority-class accuracy:", round(dummy.score(binary.drop(columns="target"), binary["target"]), 3))
```

```text output
mean price train/test: 284.6 289.6
1 0.25 0.301
2 0.248 0.307
3 0.108 0.109
4 0.106 0.113
8 0.087 0.131
12 0.066 0.158
15 0.062 0.147
majority-class accuracy: 0.598
```

Exercice 2 : l'erreur de validation est minimale vers les degrés 3 à 4, puis remonte. Exercice 4 : `final_score` (et les notes `math`, `physics`, `french`) déterminent exactement `passed` (`passed = final_score >= 10`) : c'est une fuite. Il faut exclure ces colonnes pour une prédiction réaliste (par exemple en début d'année).

## Leçon 5.2

### Quiz

1. La variation moyenne de la prédiction quand la variable augmente d'une unité, les autres variables restant fixes.
2. La MAE est la moyenne des erreurs absolues ; la RMSE la racine de la moyenne des carrés. La RMSE est plus sensible aux grosses erreurs.
3. Le modèle fait pire que la simple prédiction de la moyenne.
4. À supprimer une colonne indicatrice par variable catégorielle pour éviter une colinéarité parfaite.
5. Des erreurs qui augmentent avec la valeur prédite (hétéroscédasticité) ; modéliser le logarithme de la cible aide souvent.
6. Pour que les distances (k-NN) et la pénalité (Ridge) traitent toutes les variables à la même échelle.

### Exercices guidés

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

houses = pd.read_csv("datasets/generated/house_prices.csv")
X = pd.get_dummies(houses.drop(columns="price_keur"), columns=["neighborhood", "energy_class"], drop_first=True, dtype=float)
y = houses["price_keur"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
pred = LinearRegression().fit(X_train, y_train).predict(X_test)

err = y_test.to_numpy() - pred
mae, rmse = np.abs(err).mean(), np.sqrt((err**2).mean())
r2 = 1 - (err**2).sum() / ((y_test - y_test.mean()) ** 2).sum()
print(round(mae, 2), round(rmse, 2), round(r2, 4), "| sklearn:",
      round(mean_absolute_error(y_test, pred), 2), round(np.sqrt(mean_squared_error(y_test, pred)), 2), round(r2_score(y_test, pred), 4))

single = LinearRegression().fit(X_train[["surface_m2"]], y_train)
print("surface only R2:", round(single.score(X_test[["surface_m2"]], y_test), 3))

X2 = X.assign(log_distance=np.log1p(houses["distance_center_km"]))
X2_train, X2_test = X2.loc[X_train.index], X2.loc[X_test.index]
print("with log_distance R2:", round(LinearRegression().fit(X2_train, y_train).score(X2_test, y_test), 4))

reg = pd.read_csv("datasets/generated/regression.csv")
Xr_train, Xr_test, yr_train, yr_test = train_test_split(reg.drop(columns="y"), reg["y"], test_size=0.25, random_state=0)
for name, model in [("linear", LinearRegression()), ("kNN k=10", make_pipeline(StandardScaler(), KNeighborsRegressor(10)))]:
    print(name, "R2 =", round(model.fit(Xr_train, yr_train).score(Xr_test, yr_test), 3))
```

```text output
30.42 42.91 0.9157 | sklearn: 30.42 42.91 0.9157
surface only R2: 0.614
with log_distance R2: 0.9133
linear R2 = 0.923
kNN k=10 R2 = 0.936
```

Exercice 4 : le générateur contient des termes non linéaires (0,5·x3² et 2·sin(x4)) qu'une droite ne peut pas représenter. Le k-NN, qui les capture, obtient un meilleur R² (≈ 0,94 contre ≈ 0,92) ; l'écart reste modeste car la partie linéaire (3·x1 − 2·x2) domine. Un modèle linéaire auquel on ajoute les variables x3² et sin(x4) ferait mieux que les deux.

## Leçon 5.3

### Quiz

1. La sigmoïde.
2. Le rappel : on veut manquer le moins de malades possible (un faux négatif est très coûteux).
3. Il suppose les variables indépendantes entre elles sachant la classe.
4. Le compromis entre une marge large et le nombre d'erreurs tolérées : grand C = peu d'erreurs tolérées, risque de surapprentissage.
5. Le modèle ne classe pas mieux que le hasard.
6. Ses distances seraient dominées par les variables de grande amplitude.

### Exercices guidés

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

tp, fp, fn, tn = 40, 10, 20, 130
precision, recall = tp / (tp + fp), tp / (tp + fn)
print("accuracy", (tp + tn) / 200, "| precision", precision, "| recall", round(recall, 3), "| F1", round(2 * precision * recall / (precision + recall), 3))

df = pd.read_csv("datasets/generated/binary_classification.csv")
X, y = df.drop(columns="target"), df["target"]
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=0)
scores = {k: f1_score(y_val, make_pipeline(StandardScaler(), KNeighborsClassifier(k)).fit(X_tr, y_tr).predict(X_val)) for k in range(1, 51, 2)}
best_k = max(scores, key=scores.get)
print("best k on validation:", best_k, "F1 =", round(scores[best_k], 3))

multi = pd.read_csv("datasets/generated/multiclass_classification.csv")
Xm_train, Xm_test, ym_train, ym_test = train_test_split(multi.drop(columns="target"), multi["target"], test_size=0.25, stratify=multi["target"], random_state=0)
logit = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xm_train, ym_train)
print(confusion_matrix(ym_test, logit.predict(Xm_test)))

X_train, X_test2, y_train, y_test2 = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
binary_logit = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_train, y_train)
proba = binary_logit.predict_proba(X_test2)[:, 1]
for threshold in [t / 100 for t in range(30, 96)]:
    pred = (proba >= threshold).astype(int)
    if pred.sum() and precision_score(y_test2, pred) >= 0.75:
        print("smallest threshold with precision >= 0.75:", threshold, "| recall:", round((pred[y_test2 == 1]).mean(), 3))
        break
```

```text output
accuracy 0.85 | precision 0.8 | recall 0.667 | F1 0.727
best k on validation: 1 F1 = 0.805
[[72  1  1]
 [ 2 70  4]
 [ 0  5 70]]
smallest threshold with precision >= 0.75: 0.53 | recall: 0.5
```

Exercice 2 : k = 1 l'emporte ici, mais sur un jeu de validation de 113 exemples les écarts de F1 entre valeurs de k sont faibles et instables ; une validation croisée (leçon 5.6) donnerait un choix plus fiable. Exercice 3 : les erreurs portent surtout sur les classes 1 et 2.

## Leçon 5.4

### Quiz

1. Il teste toutes les variables et tous les seuils, et retient la coupure qui réduit le plus l'impureté (Gini, entropie ou variance).
2. Il continue à découper jusqu'à isoler chaque exemple : il mémorise le bruit.
3. Le tirage bootstrap des exemples pour chaque arbre et le tirage aléatoire des variables candidates à chaque nœud.
4. Non : plus d'arbres stabilise la moyenne ; cela coûte seulement du temps et de la mémoire.
5. Le score calculé, pour chaque exemple, uniquement avec les arbres qui ne l'ont pas vu à l'entraînement : une validation sans jeu séparé.
6. Elle mesure la perte réelle de performance sur des données de test quand on casse le lien entre la variable et la cible, au lieu de compter les coupures sur l'entraînement.

### Exercices guidés

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def gini(counts):
    p = np.array(counts) / sum(counts)
    return 1 - (p**2).sum()


parent, left, right = gini([30, 10]), gini([25, 2]), gini([5, 8])
gain = parent - (27 / 40 * left + 13 / 40 * right)
print(f"Gini parent={parent:.3f} left={left:.3f} right={right:.3f} gain={gain:.3f}")

multi = pd.read_csv("datasets/generated/multiclass_classification.csv")
X_train, X_test, y_train, y_test = train_test_split(multi.drop(columns="target"), multi["target"], test_size=0.25, stratify=multi["target"], random_state=0)
for leaf in [1, 5, 20, 50]:
    tree = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=0).fit(X_train, y_train)
    print(f"min_samples_leaf={leaf:<3} leaves={tree.get_n_leaves():<3} test={tree.score(X_test, y_test):.3f}")

binary = pd.read_csv("datasets/generated/binary_classification.csv")
Xb_train, Xb_test, yb_train, yb_test = train_test_split(binary.drop(columns="target"), binary["target"], test_size=0.25, stratify=binary["target"], random_state=0)
forest = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(Xb_train, yb_train)
svm = make_pipeline(StandardScaler(), SVC()).fit(Xb_train, yb_train)
print("forest:", round(forest.score(Xb_test, yb_test), 3), "| SVM:", round(svm.score(Xb_test, yb_test), 3))

houses = pd.read_csv("datasets/generated/house_prices.csv")
houses["energy_rank"] = houses["energy_class"].map({c: i for i, c in enumerate("ABCDEFG", start=1)})
Xh = pd.get_dummies(houses.drop(columns=["price_keur", "energy_class"]), columns=["neighborhood"], dtype=float)
Xh_train, Xh_test, yh_train, yh_test = train_test_split(Xh, houses["price_keur"], test_size=0.2, random_state=0)
for name, model in [("random forest", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1)),
                    ("hist gradient boosting", HistGradientBoostingRegressor(random_state=0))]:
    print(name, "MAE:", round(mean_absolute_error(yh_test, model.fit(Xh_train, yh_train).predict(Xh_test)), 1))
```

```text output
Gini parent=0.375 left=0.137 right=0.473 gain=0.129
min_samples_leaf=1   leaves=62  test=0.876
min_samples_leaf=5   leaves=38  test=0.902
min_samples_leaf=20  leaves=19  test=0.929
min_samples_leaf=50  leaves=12  test=0.898
forest: 0.88 | SVM: 0.853
random forest MAE: 26.4
hist gradient boosting MAE: 29.1
```

## Leçon 5.5

### Quiz

1. L'affectation de chaque point au centroïde le plus proche, puis le recalcul de chaque centroïde comme moyenne de ses points.
2. Le résultat dépend de l'initialisation (minimum local) : on garde la meilleure de plusieurs exécutions.
3. Si chaque point est plus proche des points de son groupe que de ceux du groupe voisin (proche de 1 : groupes nets ; proche de 0 : groupes qui se chevauchent).
4. La direction de plus grande variance des données (standardisées).
5. Sinon les variables de grande échelle dominent la variance et donc les composantes.
6. Non : k-means trouve toujours k groupes. Il faut vérifier la qualité (silhouette, stabilité) et l'utilité métier.

### Exercices guidés

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

customers = pd.read_csv("datasets/generated/customers.csv")
truth = pd.read_csv("datasets/generated/customers_true_segments.csv")["true_segment"]
raw = customers.drop(columns="customer_id")
X = StandardScaler().fit_transform(raw)

raw_labels = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(raw)
scaled_labels = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(X)
print("ARI without scaling:", round(adjusted_rand_score(truth, raw_labels), 3), "| with scaling:", round(adjusted_rand_score(truth, scaled_labels), 3))

pca = PCA().fit(X)
print("components for 90% variance:", int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.90) + 1))

uniform = np.random.default_rng(0).uniform(0, 1, size=(500, 2))
labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(uniform)
print("silhouette on uniform noise:", round(silhouette_score(uniform, labels), 3))

pca2 = PCA(n_components=2).fit(X)
reconstruction = pca2.inverse_transform(pca2.transform(X))
print("reconstruction MSE (standardized units):", round(((X - reconstruction) ** 2).mean(), 3))
```

```text output
ARI without scaling: 0.905 | with scaling: 0.985
components for 90% variance: 3
silhouette on uniform noise: 0.384
reconstruction MSE (standardized units): 0.225
```

Exercice 3 : la silhouette d'un nuage uniforme découpé en 3 n'est pas proche de 0 (k-means découpe proprement le carré en 3 régions), ce qui montre qu'un score « correct » ne prouve pas l'existence de vrais groupes. Exercice 4 : l'erreur de reconstruction (≈ 0,22) correspond à la part de variance non portée par les deux premières composantes (1 − 0,775 ≈ 0,225).

## Leçon 5.6

### Quiz

1. Pour que les paramètres du prétraitement soient appris uniquement sur les données d'entraînement de chaque pli, sans fuite, et pour déployer un seul objet.
2. Il applique des transformations différentes à différents groupes de colonnes et concatène les résultats.
3. La variabilité de la performance selon les données utilisées : une idée de l'incertitude du score moyen.
4. 4 × 3 × 5 = 60 entraînements.
5. Le chargement d'un pickle/joblib peut exécuter du code arbitraire.
6. À ignorer (encoder en zéros) les catégories jamais vues à l'entraînement au lieu de lever une erreur.

### Exercices guidés

```python
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

students = pd.read_csv("datasets/generated/students.csv")
numeric = ["study_hours_week", "attendance_rate", "sleep_hours", "previous_score"]
X, y = students[numeric + ["group"]], students["passed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["group"]),
])

for name, clf in [("logistic", LogisticRegression(max_iter=1000)), ("SVM", SVC(probability=True, random_state=0))]:
    scores = cross_val_score(Pipeline([("preprocess", preprocess), ("clf", clf)]), X_train, y_train, cv=5, scoring="roc_auc")
    print(f"{name:<8} CV AUC = {scores.mean():.3f} +/- {scores.std():.3f}")

selector_pipe = Pipeline([("preprocess", preprocess), ("select", SelectKBest(f_classif, k=3)), ("clf", LogisticRegression())]).fit(X_train, y_train)
names = selector_pipe.named_steps["preprocess"].get_feature_names_out()
print("selected:", list(names[selector_pipe.named_steps["select"].get_support()]))

search = RandomizedSearchCV(
    Pipeline([("preprocess", preprocess), ("clf", RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1))]),
    {"clf__max_depth": randint(2, 13), "clf__min_samples_leaf": randint(1, 31)},
    n_iter=10, cv=5, scoring="roc_auc", random_state=0, n_jobs=-1,
).fit(X_train, y_train)
print("random search best:", search.best_params_, round(search.best_score_, 3))
```

```text output
logistic CV AUC = 0.844 +/- 0.059
SVM      CV AUC = 0.803 +/- 0.078
selected: ['num__study_hours_week', 'num__attendance_rate', 'num__previous_score']
random search best: {'clf__max_depth': 5, 'clf__min_samples_leaf': 8} 0.819
```

Exercice 4 (dans un nouveau fichier, par exemple `outputs/predict.py`) :

```python no-run
import joblib
import pandas as pd

model = joblib.load("outputs/module05/models/pass_classifier.joblib")
student = pd.DataFrame([{"study_hours_week": 10, "attendance_rate": 0.9, "sleep_hours": 7, "previous_score": 11, "group": "A"}])
print(model.predict_proba(student)[0, 1])
```
