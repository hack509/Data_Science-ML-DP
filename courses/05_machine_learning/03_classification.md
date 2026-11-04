# Leçon 5.3 — Classification : logistique, k-NN, Naive Bayes, SVM et métriques

## Prérequis

- Leçons 5.1 et 5.2 ; leçon 2.1 (sigmoïde) ; leçon 2.4 (Bayes).

## Objectifs d'apprentissage

- Entraîner et interpréter une régression logistique.
- Comprendre et comparer k plus proches voisins, Naive Bayes et SVM.
- Lire une matrice de confusion et calculer accuracy, précision, rappel et F1.
- Utiliser les probabilités prédites : seuil de décision, courbe ROC et AUC.
- Choisir la métrique adaptée au coût des erreurs.

## Concepts clés

- **Classifieur** (*classifier*) : modèle qui prédit une classe ; souvent aussi une probabilité (`predict_proba`).
- **Frontière de décision** (*decision boundary*) : limite dans l'espace des variables entre les zones de chaque classe.
- **Matrice de confusion** (*confusion matrix*) : tableau réel × prédit ; VP, FP, FN, VN (*TP, FP, FN, TN*).
- **Exactitude** (*accuracy*) : proportion de bonnes prédictions.
- **Précision** (*precision*) : parmi les prédits positifs, part de vrais positifs.
- **Rappel** (*recall*, sensibilité) : parmi les positifs réels, part retrouvée.
- **F1-score** : moyenne harmonique de la précision et du rappel.
- **Courbe ROC / AUC** (*Receiver Operating Characteristic / Area Under the Curve*) : performance pour tous les seuils.

## Explication détaillée

### Fiche algorithme : régression logistique

1. **Intuition** : on calcule un score linéaire (comme en régression), puis on le transforme en probabilité avec la sigmoïde. Si p ≥ 0,5, on prédit la classe 1.
2. **Idée mathématique** : p(y = 1 | x) = σ(w·x + b). Les poids minimisent l'**entropie croisée** (*log loss*) −[y log p + (1 − y) log(1 − p)], qui pénalise fortement une probabilité confiante mais fausse. Pas de solution exacte : on optimise par une méthode itérative (cousine de la descente de gradient). Une régularisation L2 est appliquée par défaut (paramètre `C` = inverse de la force de régularisation).
3. **Implémentation** : `LogisticRegression` (variables standardisées recommandées).
4. **Données** : `binary_classification.csv` (600 exemples, 6 variables, 40 % de classe 1).
5. **Entraînement** : `fit` sur 75 % des données, découpage stratifié.
6. **Évaluation** : matrice de confusion, précision, rappel, F1, AUC.
7. **Interprétation** : un coefficient positif augmente la probabilité de la classe 1 ; exp(w) est le facteur multiplicatif sur la cote (*odds*) quand la variable augmente d'une unité (d'un écart-type si standardisée).
8. **Erreurs fréquentes** : oublier la standardisation, interpréter des coefficients de variables corrélées, utiliser le seuil 0,5 sans réfléchir aux coûts.
9-10. **Exercices et corrigé** : voir plus bas et `corrections.md`.

### Fiche algorithme : k plus proches voisins (k-NN)

1. **Intuition** : « dis-moi qui sont tes voisins, je te dirai qui tu es ». On prédit la classe majoritaire parmi les k exemples d'entraînement les plus proches.
2. **Mathématiques** : distance euclidienne d(x, x') = √Σ(xⱼ − x'ⱼ)² ; aucune phase d'apprentissage (on mémorise les données), tout le calcul se fait à la prédiction.
3. **Implémentation** : `KNeighborsClassifier(n_neighbors=k)` dans un pipeline avec `StandardScaler`.
4. à 6. Mêmes données et métriques que ci-dessus.
7. **Interprétation** : on peut montrer les voisins qui justifient une prédiction.
8. **Erreurs** : variables non standardisées ; trop de variables (en grande dimension, tous les points deviennent « loin » : *curse of dimensionality*) ; k = 1 (surapprentissage).

### Fiche algorithme : Naive Bayes

1. **Intuition** : appliquer la formule de Bayes (leçon 2.4) en supposant, « naïvement », que les variables sont indépendantes sachant la classe.
2. **Mathématiques** : P(classe | x) ∝ P(classe) · Π P(xⱼ | classe). La version gaussienne suppose chaque variable normale dans chaque classe ; la version multinomiale compte des mots (module 8).
3. **Implémentation** : `GaussianNB` (variables continues), `MultinomialNB` (comptages de mots).
4. à 6. Mêmes données et métriques.
7. **Interprétation** : moyennes et variances par classe (`theta_`, `var_`).
8. **Erreurs** : l'hypothèse d'indépendance est presque toujours fausse ; les probabilités sont souvent trop extrêmes (mal calibrées) même si le classement reste bon. Très rapide et efficace sur du texte.

### Fiche algorithme : machines à vecteurs de support (SVM)

1. **Intuition** : tracer la frontière qui sépare les classes avec la **plus grande marge** possible, c'est-à-dire la plus éloignée des points les plus proches (les *vecteurs de support*).
2. **Mathématiques** : maximiser la marge 2/‖w‖ sous contrainte de bien classer les points, en tolérant des erreurs (paramètre `C` : grand C = peu de tolérance, risque de surapprentissage). L'**astuce du noyau** (*kernel trick*, noyau RBF) permet des frontières courbes en calculant des similarités sans construire explicitement de nouvelles variables.
3. **Implémentation** : `SVC(kernel="rbf", C=1.0, gamma="scale")` ; `probability=True` pour obtenir des probabilités (calcul plus long).
4. à 6. Mêmes données et métriques.
7. **Interprétation** : difficile avec un noyau RBF ; on regarde les vecteurs de support et les performances.
8. **Erreurs** : variables non standardisées (indispensable) ; temps de calcul qui croît vite au-delà de quelques dizaines de milliers d'exemples ; C et gamma non réglés.

### Lire une matrice de confusion

|  | prédit 0 | prédit 1 |
|---|---|---|
| **réel 0** | VN (vrai négatif) | FP (faux positif) |
| **réel 1** | FN (faux négatif) | VP (vrai positif) |

- Accuracy = (VP + VN) / total. Trompeuse si les classes sont déséquilibrées.
- Précision = VP / (VP + FP) : « quand je dis spam, ai-je raison ? » Importante si les faux positifs coûtent cher (un vrai message envoyé dans les spams).
- Rappel = VP / (VP + FN) : « ai-je trouvé tous les spams / tous les malades ? » Important si les faux négatifs coûtent cher (un cancer non détecté).
- F1 = 2 · précision · rappel / (précision + rappel) : compromis quand les deux comptent.

### Seuil, ROC et AUC

Les classifieurs donnent un score ou une probabilité ; le seuil de 0,5 est un choix par défaut, pas une obligation. Baisser le seuil augmente le rappel et diminue la précision. La **courbe ROC** trace le taux de vrais positifs (rappel) contre le taux de faux positifs pour tous les seuils ; l'**AUC** (aire sous la courbe) vaut 0,5 pour un classement aléatoire et 1 pour un classement parfait. Elle s'interprète comme la probabilité qu'un exemple positif tiré au hasard reçoive un score plus élevé qu'un négatif. On choisit ensuite le seuil selon les coûts métier, **sur le jeu de validation**.

## Intuition mathématique

```text
logistique :   p = σ(w·x + b) = 1 / (1 + e^−(w·x + b))
log loss :     L = −(1/n) Σ [yᵢ log pᵢ + (1 − yᵢ) log(1 − pᵢ)]
Naive Bayes :  P(c | x) ∝ P(c) Πⱼ P(xⱼ | c)
SVM :          min ½‖w‖² + C Σ ξᵢ   sous yᵢ(w·xᵢ + b) ≥ 1 − ξᵢ
F1 :           2PR / (P + R)
```

## Applications concrètes

- Filtrage de spam (précision importante), dépistage médical (rappel important), détection de fraude (classes très déséquilibrées, AUC et précision/rappel), prédiction de désabonnement de clients.

## Exemples de code Python

### Exemple 1 — comparer quatre classifieurs

```python
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("datasets/generated/binary_classification.csv")
X, y = df.drop(columns="target"), df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

models = {
    "baseline": DummyClassifier(strategy="most_frequent"),
    "logistic": make_pipeline(StandardScaler(), LogisticRegression()),
    "kNN (k=15)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    "naive Bayes": GaussianNB(),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=0)),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"{name:<12} accuracy={accuracy_score(y_test, pred):.3f}  F1={f1_score(y_test, pred, zero_division=0):.3f}  "
          f"AUC={roc_auc_score(y_test, proba):.3f}")
```

### Explication du code

- Tous les modèles partagent la même interface `fit` / `predict` / `predict_proba` (leçon 1.3) : on les compare dans une simple boucle.
- La référence « classe majoritaire » obtient environ 60 % d'accuracy mais un F1 nul et une AUC de 0,5 : elle ne détecte aucun positif. D'où l'importance de regarder plusieurs métriques.
- Les modèles non linéaires (k-NN, SVM à noyau RBF) dépassent ici la régression logistique : les classes générées par `make_classification` forment des groupes que des frontières courbes séparent mieux.

### Sortie attendue

```text output
baseline     accuracy=0.600  F1=0.000  AUC=0.500
logistic     accuracy=0.740  F1=0.649  AUC=0.732
kNN (k=15)   accuracy=0.827  F1=0.740  AUC=0.916
naive Bayes  accuracy=0.773  F1=0.691  AUC=0.779
SVM (RBF)    accuracy=0.853  F1=0.792  AUC=0.899
```

### Exemple 2 — matrice de confusion et métriques en détail

```python
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from lab.plotting import apply_style, sequential_cmap

logistic = models["logistic"]
pred = logistic.predict(X_test)
cm = confusion_matrix(y_test, pred)
tn, fp, fn, tp = cm.ravel()
print(cm)
print(f"precision = {tp}/({tp}+{fp}) = {tp / (tp + fp):.3f}")
print(f"recall    = {tp}/({tp}+{fn}) = {tp / (tp + fn):.3f}")
print(classification_report(y_test, pred, digits=3))

coefficients = pd.Series(logistic[-1].coef_[0], index=X.columns).sort_values()
print("standardized coefficients:", coefficients.round(2).to_dict())

apply_style()
fig, ax = plt.subplots(figsize=(3.6, 3.2))
ConfusionMatrixDisplay(cm, display_labels=["classe 0", "classe 1"]).plot(ax=ax, cmap=sequential_cmap(), colorbar=False)
ax.grid(False)
ax.set_title("Régression logistique (test)")
Path("outputs/module05").mkdir(parents=True, exist_ok=True)
fig.savefig("outputs/module05/confusion_matrix.png", bbox_inches="tight")
plt.close(fig)
```

### Explication du code

- `confusion_matrix(...).ravel()` renvoie VN, FP, FN, VP dans cet ordre (pour des classes 0/1).
- On recalcule précision et rappel à la main pour vérifier qu'on a bien compris les formules.
- `classification_report` donne précision, rappel et F1 pour chaque classe, plus les moyennes (*macro* = moyenne simple des classes, *weighted* = pondérée par l'effectif).
- `logistic[-1]` accède au dernier élément du pipeline (le modèle) ; les coefficients portent sur des variables standardisées et sont donc comparables entre eux.

### Sortie attendue

```text output
[[75 15]
 [24 36]]
precision = 36/(36+15) = 0.706
recall    = 36/(36+24) = 0.600
              precision    recall  f1-score   support

           0      0.758     0.833     0.794        90
           1      0.706     0.600     0.649        60

    accuracy                          0.740       150
   macro avg      0.732     0.717     0.721       150
weighted avg      0.737     0.740     0.736       150

standardized coefficients: {'x6': -0.35, 'x5': -0.04, 'x3': 0.13, 'x4': 0.13, 'x1': 0.27, 'x2': 0.6}
```

### Exemple 3 — choisir un seuil, courbe ROC

```python
import numpy as np
from sklearn.metrics import RocCurveDisplay, precision_score, recall_score

from lab.plotting import CATEGORICAL

proba = logistic.predict_proba(X_test)[:, 1]
for threshold in [0.3, 0.5, 0.7]:
    pred_t = (proba >= threshold).astype(int)
    print(f"threshold {threshold}: precision={precision_score(y_test, pred_t):.3f}  recall={recall_score(y_test, pred_t):.3f}  "
          f"predicted positives={pred_t.sum()}")

fig, ax = plt.subplots(figsize=(4.2, 4))
for color, name in zip(CATEGORICAL, ["logistic", "SVM (RBF)", "naive Bayes"]):
    RocCurveDisplay.from_estimator(models[name], X_test, y_test, ax=ax, name=name, curve_kwargs={"color": color})
ax.plot([0, 1], [0, 1], linestyle="--", color="#8a8a85", label="hasard")
ax.set_title("Courbes ROC sur le jeu de test")
ax.legend(loc="lower right")
fig.savefig("outputs/module05/roc_curves.png", bbox_inches="tight")
plt.close(fig)
```

### Explication du code

Baisser le seuil à 0,3 fait prédire davantage de positifs : le rappel augmente, la précision baisse. Monter le seuil à 0,7 effondre le rappel (10 %) sans même améliorer la précision : avec seulement 9 positifs prédits, la précision devient très instable. En général la précision tend à monter avec le seuil, mais ce n'est pas garanti : il faut regarder la courbe précision-rappel. Le bon seuil dépend du coût relatif des erreurs et se choisit sur un jeu de validation (ici on illustre sur le test pour rester bref). Les courbes ROC (`outputs/module05/roc_curves.png`) résument tous les seuils : plus la courbe est proche du coin supérieur gauche, meilleur est le classement.

### Sortie attendue

```text output
threshold 0.3: precision=0.461  recall=0.783  predicted positives=102
threshold 0.5: precision=0.706  recall=0.600  predicted positives=51
threshold 0.7: precision=0.667  recall=0.100  predicted positives=9
```

## Erreurs fréquentes

- **Juger sur l'accuracy seule** avec des classes déséquilibrées.
- **Confondre précision et rappel**.
- **Garder le seuil de 0,5 sans réfléchir** au coût des erreurs.
- **Choisir le seuil sur le jeu de test**.
- **Oublier la standardisation** pour la logistique, k-NN et SVM.
- **Prendre les probabilités de Naive Bayes ou de SVM pour des probabilités fiables** : elles peuvent être mal calibrées.
- **Calculer l'AUC avec les classes prédites** au lieu des scores/probabilités.

## Exercices guidés

1. Calculez à la main accuracy, précision, rappel et F1 pour : VP = 40, FP = 10, FN = 20, VN = 130.
2. Faites varier k de 1 à 50 pour le k-NN et tracez le F1 de validation (utilisez un jeu de validation séparé). Quelle valeur de k retenir ?
3. Sur `multiclass_classification.csv`, entraînez une régression logistique et affichez la matrice de confusion 3 × 3. Quelles classes sont confondues ?
4. Avec la régression logistique de l'exemple 2, trouvez le plus petit seuil tel que la précision soit au moins de 0,75 sur le jeu de test, et le rappel correspondant.

Corrigés : [corrections.md](corrections.md), section « Leçon 5.3 ».

L'exercice **I08** vous fait coder toutes ces métriques sans scikit-learn.

## Défis autonomes

- Créez un jeu de données très déséquilibré (`make_classification(weights=[0.97])`), montrez que l'accuracy est trompeuse et comparez ROC-AUC et `average_precision_score`.
- Tracez la frontière de décision de la logistique, du k-NN et du SVM sur deux variables (`x1`, `x2`) avec une grille de points.

## Quiz

1. Quelle fonction transforme le score linéaire en probabilité dans la régression logistique ?
2. Dépistage d'une maladie grave : privilégier la précision ou le rappel ?
3. Pourquoi Naive Bayes est-il « naïf » ?
4. Que contrôle le paramètre `C` d'un SVM ?
5. Une AUC de 0,5 signifie quoi ?
6. Pourquoi k-NN a-t-il besoin de variables standardisées ?

Réponses : [corrections.md](corrections.md), section « Leçon 5.3 ».

## Résumé

- Logistique (linéaire, interprétable, probabilités), k-NN (voisins, non linéaire), Naive Bayes (rapide, texte), SVM (marge maximale, noyaux).
- La matrice de confusion est la base de toutes les métriques ; choisissez celle qui reflète le coût des erreurs.
- Les probabilités permettent d'ajuster le seuil ; ROC-AUC évalue le classement indépendamment du seuil.

## Pour aller plus loin

- Exercice **I08** (`exercises/intermediate/I08_classification_metrics`).
- Projet 4 (`projects/04_spam_detection`).
- Leçon suivante : [Arbres de décision et forêts aléatoires](04_arbres_forets.md).
