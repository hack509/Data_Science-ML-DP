# Leçon 8.2 — Sac de mots, TF-IDF et classification de textes

## Prérequis

- Leçon 8.1 ; leçons 5.3 (classification, métriques) et 5.6 (pipelines).

## Objectifs d'apprentissage

- Représenter des textes par un sac de mots (*bag of words*) et par TF-IDF.
- Entraîner des classifieurs de texte (Naive Bayes multinomial, régression logistique) dans un pipeline.
- Détecter et éviter les fuites dues aux doublons de textes.
- Utiliser les n-grammes pour capter des expressions comme « pas mal ».
- Interpréter un classifieur linéaire de texte (mots les plus discriminants) et analyser ses erreurs.

## Concepts clés

- **Sac de mots** (*bag of words*, BoW) : un document devient un vecteur de comptages de mots ; l'ordre des mots est ignoré.
- **Matrice document-terme** : une ligne par document, une colonne par mot du vocabulaire ; très **creuse** (*sparse*).
- **TF** (*term frequency*) : fréquence d'un mot dans un document.
- **IDF** (*inverse document frequency*) : rareté d'un mot dans le corpus.
- **TF-IDF** : produit des deux ; élevé pour un mot fréquent dans ce document mais rare ailleurs.
- **N-grammes** : séquences de n mots utilisées comme variables.
- **Analyse de sentiment** (*sentiment analysis*) : classer un texte selon l'opinion exprimée.

## Explication détaillée

### Du texte aux vecteurs

`CountVectorizer` apprend un vocabulaire sur les textes d'entraînement et transforme chaque texte en un vecteur de comptages. Avec 10 000 mots, chaque vecteur a 10 000 dimensions, presque toutes nulles : scikit-learn utilise des matrices creuses (`scipy.sparse`) qui ne stockent que les valeurs non nulles.

Le sac de mots ignore l'ordre : « le chien mord l'homme » et « l'homme mord le chien » ont la même représentation. Les **n-grammes** réintroduisent un peu d'ordre local : avec les bigrammes, « pas mal » et « pas décevant » deviennent des variables à part entière, ce qui aide pour les négations.

### TF-IDF

Les mots fréquents partout (« le », « de ») dominent les comptages sans rien apprendre au modèle. TF-IDF pondère chaque comptage par la rareté du mot dans le corpus :

```text
tf-idf(t, d) = tf(t, d) × idf(t)      idf(t) = ln((1 + N) / (1 + df(t))) + 1   (formule « lissée » de scikit-learn)
```

où N est le nombre de documents et df(t) le nombre de documents contenant t. Chaque vecteur est ensuite normalisé (norme L2 = 1), si bien que les longs documents ne sont pas avantagés et que le produit scalaire de deux vecteurs est leur similarité cosinus (leçon 2.2).

### Les classifieurs

- **Naive Bayes multinomial** : modélise la probabilité de chaque mot dans chaque classe ; très rapide, robuste avec peu de données, excellent point de départ pour le spam.
- **Régression logistique** sur TF-IDF : souvent la meilleure référence simple pour la classification de textes ; ses coefficients indiquent les mots qui poussent vers chaque classe.
- **SVM linéaire** : performances proches de la logistique.

Le tout dans un pipeline : `make_pipeline(TfidfVectorizer(...), LogisticRegression())`. Le vectoriseur est un transformateur comme un autre : il apprend vocabulaire et IDF sur l'entraînement uniquement.

### Fuites spécifiques au texte

Les corpus réels contiennent beaucoup de **doublons** : messages automatiques, copier-coller, retweets. Si le même texte est à la fois dans l'entraînement et le test, le modèle est évalué sur des exemples qu'il a déjà vus. Les SMS synthétiques de ce cours en contiennent énormément (ils sont générés à partir de modèles de phrases). Solutions : dédoublonner avant de découper, ou découper par **groupes** (tous les exemplaires d'un même texte du même côté, `GroupShuffleSplit`).

### Analyser les erreurs

Après l'évaluation, lisez les textes mal classés : ils révèlent les limites du modèle (négations, ironie, vocabulaire nouveau) et donnent des idées de variables ou de données à ajouter.

## Intuition mathématique

```text
Naive Bayes multinomial :  log P(c | d) = log P(c) + Σ_mots count(t, d) · log P(t | c) + constante
P(t | c) = (count(t, c) + α) / (Σ_t' count(t', c) + α |V|)     (lissage de Laplace α = 1)
logistique sur TF-IDF : score(d) = w · x_d + b ; le poids wₜ indique la direction et la force d'un mot
```

## Applications concrètes

- Filtres anti-spam, tri automatique de courriels et de tickets, détection de contenus abusifs.
- Analyse de sentiment d'avis clients, de réponses à un questionnaire.
- Recherche documentaire (TF-IDF + similarité cosinus, leçon 8.4).

## Exemples de code Python

### Exemple 1 — sac de mots et TF-IDF sur trois phrases

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

documents = [
    "le colis est arrivé",
    "le colis est en attente, payez les frais",
    "payez vite, offre gratuite",
]
counter = CountVectorizer()
counts = counter.fit_transform(documents)
print(pd.DataFrame(counts.toarray(), columns=counter.get_feature_names_out()).to_string())

tfidf = TfidfVectorizer()
weights = tfidf.fit_transform(documents)
idf = pd.Series(tfidf.idf_, index=tfidf.get_feature_names_out()).round(3)
print("idf:", idf.sort_values().to_dict())
print("cosine similarities:\n", np.round((weights @ weights.T).toarray(), 3))
print("matrix type:", type(weights).__name__, "| stored values:", weights.nnz, "of", weights.shape[0] * weights.shape[1])
```

### Explication du code

- La matrice de comptages a une colonne par mot du vocabulaire (appris sur ces trois phrases).
- « colis », « est », « le » et « payez » apparaissent dans deux documents : leur IDF est plus faible que celui des mots présents dans un seul document (« gratuite », « frais »...).
- Les vecteurs TF-IDF sont normalisés : leur produit scalaire est la similarité cosinus. Les phrases 1 et 2 sont les plus proches (« le colis est »).
- La matrice est stockée au format creux : seules les valeurs non nulles sont conservées.

### Sortie attendue

```text output
   arrivé  attente  colis  en  est  frais  gratuite  le  les  offre  payez  vite
0       1        0      1   0    1      0         0   1    0      0      0     0
1       0        1      1   1    1      1         0   1    1      0      1     0
2       0        0      0   0    0      0         1   0    0      1      1     1
idf: {'colis': 1.288, 'est': 1.288, 'le': 1.288, 'payez': 1.288, 'en': 1.693, 'attente': 1.693, 'frais': 1.693, 'arrivé': 1.693, 'gratuite': 1.693, 'les': 1.693, 'offre': 1.693, 'vite': 1.693}
cosine similarities:
 [[1.    0.418 0.   ]
 [0.418 1.    0.122]
 [0.    0.122 1.   ]]
matrix type: csr_matrix | stored values: 16 of 36
```

### Exemple 2 — détecteur de spam, avec et sans fuite par doublons

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

sms = pd.read_csv("datasets/generated/sms_spam.csv")
print("messages:", len(sms), "| distinct texts:", sms["text"].nunique(), "| spam share:", round((sms["label"] == "spam").mean(), 3))

# Naive split: identical texts may be on both sides.
tr, te = train_test_split(sms, test_size=0.25, stratify=sms["label"], random_state=0)
print("test messages whose exact text is also in train:", round(te["text"].isin(set(tr["text"])).mean(), 3))

# Group split: every copy of a text goes to the same side.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
train_idx, test_idx = next(splitter.split(sms, groups=sms["text"]))
train, test = sms.iloc[train_idx], sms.iloc[test_idx]
print("overlap after group split:", int(test["text"].isin(set(train["text"])).sum()))

for name, model in [
    ("naive Bayes (counts)", make_pipeline(CountVectorizer(), MultinomialNB())),
    ("logistic (TF-IDF 1-2 grams)", make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))),
]:
    model.fit(train["text"], train["label"])
    pred = model.predict(test["text"])
    print(f"{name:<28} F1(spam) = {f1_score(test['label'], pred, pos_label='spam'):.3f}")
print(classification_report(test["label"], pred, digits=3))
```

### Explication du code

- Les 1 200 messages ne contiennent que 742 textes distincts : avec un découpage aléatoire, près de la moitié des messages de test ont un jumeau exact dans l'entraînement.
- `GroupShuffleSplit(groups=texte)` place toutes les copies d'un même texte du même côté : aucun recouvrement.
- Les scores restent très élevés (F1 ≈ 0,98) : les spams synthétiques ont un vocabulaire typé (« gagné », `<url>`, « EUR »...). Les quelques erreurs viennent des cas difficiles ajoutés volontairement (messages légitimes au ton commercial, spams au ton familier) et des 2 % d'**erreurs d'annotation** introduites par le générateur, comme dans les vrais jeux de données. **Sur de vrais SMS, il faut s'attendre à des scores plus faibles** : ce résultat illustre la méthode, pas la difficulté réelle du problème.

### Sortie attendue

```text output
messages: 1200 | distinct texts: 742 | spam share: 0.259
test messages whose exact text is also in train: 0.453
overlap after group split: 0
naive Bayes (counts)         F1(spam) = 0.980
logistic (TF-IDF 1-2 grams)  F1(spam) = 0.980
              precision    recall  f1-score   support

         ham      0.988     0.994     0.991       163
        spam      0.986     0.973     0.980        74

    accuracy                          0.987       237
   macro avg      0.987     0.983     0.985       237
weighted avg      0.987     0.987     0.987       237
```

### Exemple 3 — sentiment : négations, n-grammes, interprétation et erreurs

```python
reviews = pd.read_csv("datasets/generated/reviews_fr.csv")
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1)
train_idx, test_idx = next(splitter.split(reviews, groups=reviews["text"]))
train, test = reviews.iloc[train_idx], reviews.iloc[test_idx]

for ngrams in [(1, 1), (1, 2)]:
    model = make_pipeline(TfidfVectorizer(ngram_range=ngrams, strip_accents="unicode"), LogisticRegression(max_iter=1000, C=10))
    model.fit(train["text"], train["sentiment"])
    accuracy = (model.predict(test["text"]) == test["sentiment"]).mean()
    print(f"ngram_range={ngrams}: test accuracy={accuracy:.3f}")

vectorizer, classifier = model.named_steps["tfidfvectorizer"], model.named_steps["logisticregression"]
coefficients = pd.Series(classifier.coef_[0], index=vectorizer.get_feature_names_out())
print("classes:", list(classifier.classes_), "-> positive coefficients push towards", classifier.classes_[1])
print("most 'positif':", coefficients.nlargest(6).round(2).to_dict())
print("most 'negatif':", coefficients.nsmallest(6).round(2).to_dict())

test = test.assign(prediction=model.predict(test["text"]))
errors = test[test["prediction"] != test["sentiment"]]
print(f"{len(errors)} errors out of {len(test)}; examples:")
for _, row in errors.head(3).iterrows():
    print(f"  [{row['sentiment']} predicted {row['prediction']}] {row['text']}")
```

### Explication du code

- Découpage par groupes, comme pour le spam.
- Les avis contiennent des tournures où l'ordre des mots inverse le sens : « Le casque n'est pas décevant » (positif) contre « Le casque n'est pas solide » (négatif), « Je craignais qu'il soit lent, mais il est rapide » contre l'inverse. Avec les seuls mots isolés, ces phrases sont ambiguës ; les bigrammes (« pas lent », « pas excellent ») apportent un gain, modeste ici (≈ 95 % → 95,5 %).
- `strip_accents="unicode"` applique la normalisation des accents de la leçon 8.1 à l'intérieur du vectoriseur.
- Les coefficients les plus positifs et les plus négatifs se lisent directement : ce sont les mots et bigrammes qui font pencher la décision. Les classes sont triées par ordre alphabétique (« négatif » puis « positif ») : un coefficient positif pousse vers « positif ».
- Certains bigrammes forts sont des artefacts (« ondes decevant », issu de « micro-ondes ») : un signe que le modèle apprend aussi des coïncidences de ce petit corpus.
- L'analyse des erreurs est instructive : on y trouve des négations mal comprises (« n'est pas décevant » classé négatif), mais aussi des **erreurs d'étiquette** (« Ce micro-ondes est rapide, je recommande » étiqueté négatif : le générateur inverse volontairement 3 % des étiquettes). Le modèle a raison et l'étiquette a tort ! Lire les erreurs, c'est aussi auditer les données.

### Sortie attendue

```text output
ngram_range=(1, 1): test accuracy=0.949
ngram_range=(1, 2): test accuracy=0.955
classes: ['négatif', 'positif'] -> positive coefficients push towards positif
most 'positif': {'je recommande': 5.15, 'pas trop': 4.47, 'pas inutilisable': 4.1, 'pas lent': 3.95, 'ondes decevant': 3.23, 'pas bruyant': 2.88}
most 'negatif': {'pas excellent': -4.54, 'je ne': -3.59, 'ne': -3.59, 'ne recommande': -3.59, 'recommande pas': -3.59, 'pas tres': -3.26}
8 errors out of 176; examples:
  [positif predicted négatif] Le casque n'est pas de mauvaise qualité.
  [négatif predicted positif] Ce micro-ondes est rapide, je recommande.
  [positif predicted négatif] Le casque n'est pas décevant.
```

## Erreurs fréquentes

- **Ajuster le vectoriseur sur tout le corpus** (vocabulaire et IDF qui voient le test).
- **Ignorer les doublons** : scores gonflés.
- **Retirer les négations** comme mots vides.
- **Convertir une matrice creuse en dense** (`toarray()`) sur un gros corpus : saturation de la mémoire.
- **Croire qu'un classifieur de sac de mots comprend le sens** : il compte des mots ; l'ironie et les tournures complexes lui échappent.
- **Se fier à des scores obtenus sur des données synthétiques** pour un usage réel.

## Exercices guidés

1. Calculez à la main l'IDF lissé de « colis » dans l'exemple 1 (N = 3, df = 2) et vérifiez avec `tfidf.idf_`.
2. Dans l'exemple 2, comparez le F1 obtenu avec un découpage aléatoire simple et avec le découpage par groupes.
3. Affichez les 10 mots les plus associés au spam selon la régression logistique.
4. Ajoutez au détecteur de spam l'option `min_df=2` du vectoriseur. Combien de colonnes sont supprimées ?

Corrigés : [corrections.md](corrections.md), section « Leçon 8.2 ».

## Défis autonomes

- Remplacez TF-IDF par des n-grammes de **caractères** (`analyzer="char_wb", ngram_range=(3, 5)`) : plus robustes aux fautes d'orthographe. Comparez.
- Écrivez une fonction `explain(text)` qui affiche, pour un message, la contribution (TF-IDF × coefficient) de chacun de ses mots à la décision.
- Exercice **A03** : implémentez TF-IDF de zéro.

## Quiz

1. Que perd-on avec un sac de mots ?
2. Pourquoi TF-IDF donne-t-il un faible poids aux mots très fréquents ?
3. Pourquoi les matrices document-terme sont-elles stockées au format creux ?
4. Comment les doublons de textes faussent-ils l'évaluation ? Comment l'éviter ?
5. En quoi les bigrammes aident-ils pour les négations ?
6. Comment interpréter les coefficients d'une régression logistique sur TF-IDF ?

Réponses : [corrections.md](corrections.md), section « Leçon 8.2 ».

## Résumé

- Sac de mots : comptages, ordre ignoré ; n-grammes : un peu d'ordre local.
- TF-IDF : pondère par la rareté, normalise ; produit scalaire = similarité cosinus.
- Pipeline vectoriseur + Naive Bayes ou logistique : références solides et interprétables.
- Dédoublonner ou découper par groupes ; analyser les erreurs.

## Pour aller plus loin

- Exercice **A03** (`exercises/advanced/A03_tfidf_from_scratch`).
- Projet 4 (`projects/04_spam_detection`).
- Leçon suivante : [Embeddings de mots et classifieur neuronal avec PyTorch](03_embeddings_pytorch.md).
