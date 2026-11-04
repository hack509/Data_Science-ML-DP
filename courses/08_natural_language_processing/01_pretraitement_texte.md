# Leçon 8.1 — Prétraitement du texte : normalisation, tokenisation, vocabulaire

## Prérequis

- Leçons 1.1 et 1.2 (chaînes de caractères, dictionnaires, fonctions), leçon 1.5 (pandas).

## Objectifs d'apprentissage

- Normaliser un texte : casse, accents, ponctuation, nombres, adresses web.
- Découper un texte en jetons avec des expressions régulières.
- Construire un vocabulaire avec des fréquences et des identifiants.
- Utiliser des mots vides (*stop words*) avec discernement.
- Comprendre la racinisation (*stemming*) et la lemmatisation, et leurs limites en français.

## Concepts clés

- **Corpus** : ensemble de documents (ici des SMS ou des avis).
- **Jeton** (*token*) : unité de texte après découpage (mot, ponctuation, sous-mot).
- **Tokenisation** (*tokenization*) : découpage d'un texte en jetons.
- **Vocabulaire** (*vocabulary*) : ensemble des jetons connus, chacun associé à un identifiant entier.
- **Mot vide** (*stop word*) : mot très fréquent et peu informatif (« le », « de », « et »).
- **Racinisation** (*stemming*) : couper les terminaisons par des règles (« achetés » → « achet »).
- **Lemmatisation** (*lemmatization*) : ramener un mot à sa forme de dictionnaire (« achetés » → « acheter »), ce qui exige un dictionnaire ou un modèle linguistique.
- **N-gramme** : suite de n jetons consécutifs (« pas mal » est un bigramme).
- **Hors vocabulaire** (*out-of-vocabulary*, OOV) : mot absent du vocabulaire d'entraînement.

## Explication détaillée

### Pourquoi prétraiter ?

Un modèle ne lit pas : il compte et compare. Pour lui, « Gratuit », « gratuit » et « GRATUIT ! » sont trois chaînes différentes. Le prétraitement réduit cette variabilité inutile afin que les mêmes idées soient représentées de la même manière. Mais chaque étape détruit aussi de l'information : les majuscules peuvent signaler un spam (« URGENT »), la ponctuation une émotion (« !!! »), et « pas » change le sens de « mauvais ». Le bon prétraitement dépend de la tâche, et se valide par l'expérience.

### Les étapes courantes

1. **Unicode** : en français, « é » peut être codé comme un seul caractère ou comme « e » + accent combinant. `unicodedata.normalize("NFC", texte)` unifie. Supprimer les accents (`NFKD` puis retirer les caractères combinants) rend « réclamez » et « reclamez » identiques : utile quand les utilisateurs écrivent sans accents (SMS), au prix de quelques ambiguïtés (« côte » / « cote »).
2. **Casse** : `lower()` (ou `casefold()`, plus complet).
3. **Remplacements** : les adresses web, numéros de téléphone et montants sont remplacés par des jetons spéciaux (`<url>`, `<tel>`, `<num>`). Le modèle apprend que « un numéro de téléphone » est suspect sans mémoriser chaque numéro.
4. **Tokenisation** : avec une expression régulière comme `\w+` (suites de lettres et chiffres, accents compris), ou une règle plus fine qui garde les apostrophes (« l'argent » → « l », « argent »).
5. **Mots vides** : on peut les retirer pour réduire le bruit dans les méthodes de comptage. **Attention aux négations** : « pas », « ne », « sans » sont dans beaucoup de listes de mots vides, mais en analyse de sentiment, les supprimer transforme « pas du tout décevant » en « décevant ».
6. **Racinisation ou lemmatisation** : regroupent les formes d'un même mot (« gagné », « gagnez », « gagner »). Utile avec peu de données ; souvent inutile, voire nuisible, avec beaucoup de données ou avec des modèles neuronaux.

### Le vocabulaire

On compte les jetons sur le jeu d'**entraînement** uniquement et on garde ceux qui apparaissent au moins `min_freq` fois. Chacun reçoit un identifiant ; deux identifiants spéciaux sont réservés : `<pad>` (remplissage pour mettre des textes à la même longueur) et `<unk>` (mot inconnu). La distribution des fréquences suit approximativement la **loi de Zipf** : quelques mots sont extrêmement fréquents, la grande majorité est rare.

### Outils et téléchargements

Les bibliothèques NLTK et spaCy offrent des tokeniseurs, des listes de mots vides, des raciniseurs (Snowball pour le français) et des lemmatiseurs de qualité, mais elles **téléchargent des ressources** au premier usage (`nltk.download(...)`, `python -m spacy download fr_core_news_sm`). Pour rester entièrement hors ligne et sans dépendance supplémentaire, cette leçon écrit des versions simples. Si vous installez ces outils plus tard (étape en ligne explicite), les ressources sont stockées sur le disque et fonctionnent ensuite hors ligne.

## Intuition mathématique

Loi de Zipf : la fréquence du mot de rang r est approximativement proportionnelle à 1/r.

```text
fréquence(r) ≈ C / r      ⇔      log(fréquence) ≈ log C − log r   (une droite de pente −1 en échelle log-log)
```

Conséquence pratique : en ne gardant que quelques milliers de mots les plus fréquents, on couvre l'essentiel des occurrences d'un corpus, mais une part non négligeable des mots d'un nouveau texte sera inconnue.

## Applications concrètes

- Préparer des textes pour un filtre anti-spam, un classifieur de tickets de support, un moteur de recherche.
- Normaliser des noms de produits ou d'adresses avant de les comparer.

## Exemples de code Python

### Exemple 1 — normaliser et tokeniser

```python
import re
import unicodedata


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize(text: str, remove_accents: bool = True) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"https?://\S+", " <url> ", text)
    text = re.sub(r"\b0\d(?:[ .]?\d{2}){4}\b", " <tel> ", text)
    text = re.sub(r"\d+(?:[.,]\d+)?", " <num> ", text)
    return strip_accents(text) if remove_accents else text


TOKEN_PATTERN = re.compile(r"<\w+>|\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize(text))


message = "FÉLICITATIONS ! Vous avez gagné 500 EUR. Appelez le 08 99 12 34 56 ou http://promo-123.example"
print(normalize(message))
print(tokenize(message))
print(tokenize("L'écran est très décevant, je ne le recommande pas."))
```

### Explication du code

- `strip_accents` décompose chaque caractère accentué en lettre + accent combinant, puis retire les accents.
- Les expressions régulières remplacent adresses web, numéros de téléphone français et nombres par des jetons spéciaux. L'ordre compte : le téléphone est traité avant les nombres.
- Le motif de tokenisation garde les jetons spéciaux (`<url>`) et les suites de caractères de mot. L'apostrophe sépare « l » de « ecran ».

### Sortie attendue

```text output
felicitations ! vous avez gagne  <num>  eur. appelez le  <tel>  ou  <url> 
['felicitations', 'vous', 'avez', 'gagne', '<num>', 'eur', 'appelez', 'le', '<tel>', 'ou', '<url>']
['l', 'ecran', 'est', 'tres', 'decevant', 'je', 'ne', 'le', 'recommande', 'pas']
```

### Exemple 2 — vocabulaire, loi de Zipf et mots hors vocabulaire

```python
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

sms = pd.read_csv("datasets/generated/sms_spam.csv")
train, test = train_test_split(sms, test_size=0.25, stratify=sms["label"], random_state=0)

counts = Counter(token for text in train["text"] for token in tokenize(text))
print("distinct tokens in training set:", len(counts), "| total tokens:", sum(counts.values()))
print("10 most frequent:", counts.most_common(10))
for rank in (1, 2, 4, 8):
    word, freq = counts.most_common()[rank - 1]
    print(f"rank {rank}: {word!r:<10} frequency={freq:<5} frequency x rank={freq * rank}")


def build_vocabulary(token_counts: Counter, min_freq: int = 2) -> dict[str, int]:
    vocabulary = {"<pad>": 0, "<unk>": 1}
    for token, freq in token_counts.most_common():
        if freq >= min_freq:
            vocabulary[token] = len(vocabulary)
    return vocabulary


vocab = build_vocabulary(counts, min_freq=2)
encode = lambda text: [vocab.get(token, vocab["<unk>"]) for token in tokenize(text)]
print("vocabulary size:", len(vocab))
print("encoded:", encode("Salut Hugo, on se retrouve au cinéma ?"))
test_tokens = [t for text in test["text"] for t in tokenize(text)]
print(f"out-of-vocabulary rate on test: {sum(t not in vocab for t in test_tokens) / len(test_tokens):.2%}")
```

### Explication du code

- Le vocabulaire est construit sur l'**entraînement** uniquement, comme tout paramètre appris (leçon 5.1).
- Le produit fréquence × rang reste du même ordre de grandeur pour les premiers rangs : c'est la loi de Zipf (approximative sur un si petit corpus synthétique).
- `min_freq=2` écarte les mots vus une seule fois ; ils deviendront `<unk>`.
- Le corpus ne contient que 306 jetons distincts et le taux de mots hors vocabulaire sur le test est d'environ 1 % : les SMS synthétiques sont construits à partir de quelques modèles de phrases, et les rares mots inconnus viennent surtout des fautes de frappe simulées (un caractère supprimé dans 15 % des messages). Sur de vrais SMS (fautes, abréviations, émojis), le vocabulaire compterait des milliers de mots et le taux d'inconnus serait bien plus élevé.
- Le jeton « h » parmi les plus fréquents vient des horaires (« 18h » devient « <num> h ») : un effet de bord du prétraitement, à repérer en regardant les jetons les plus fréquents. On pourrait ajouter une règle `\d+h\d*` → `<heure>`.

### Sortie attendue

```text output
distinct tokens in training set: 306 | total tokens: 9770
10 most frequent: [('<num>', 556), ('le', 439), ('est', 303), ('votre', 289), ('de', 284), ('a', 282), ('au', 247), ('h', 184), ('on', 173), ('du', 167)]
rank 1: '<num>'    frequency=556   frequency x rank=556
rank 2: 'le'       frequency=439   frequency x rank=878
rank 4: 'votre'    frequency=289   frequency x rank=1156
rank 8: 'h'        frequency=184   frequency x rank=1472
vocabulary size: 219
encoded: [70, 181, 10, 146, 147, 8, 108]
out-of-vocabulary rate on test: 1.09%
```

### Exemple 3 — mots vides et racinisation simple

```python
FRENCH_STOP_WORDS = {
    "le", "la", "les", "l", "un", "une", "des", "de", "du", "d", "au", "aux", "et", "ou", "a", "à",
    "en", "dans", "sur", "pour", "par", "avec", "ce", "cet", "cette", "ces", "je", "j", "tu", "il",
    "elle", "on", "nous", "vous", "ils", "elles", "me", "te", "se", "lui", "y", "est", "c", "qu", "que", "qui",
}
NEGATIONS = {"ne", "n", "pas", "plus", "jamais", "rien", "aucun", "sans"}  # kept on purpose


def remove_stop_words(tokens):
    return [t for t in tokens if t not in FRENCH_STOP_WORDS or t in NEGATIONS]


SUFFIXES = sorted(["ements", "ement", "ations", "ation", "ables", "able", "euses", "euse", "eux", "ees", "ee", "es", "er", "ez", "e", "s"], key=len, reverse=True)


def simple_stem(token: str) -> str:
    """Very small rule-based stemmer: removes the longest known suffix, keeps at least 3 letters."""
    for suffix in SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


review = "Ce produit n'est pas du tout décevant, livraison rapide et je recommande vraiment."
tokens = tokenize(review)
print("tokens:          ", tokens)
print("without stopwords:", remove_stop_words(tokens))
print("stems:           ", [simple_stem(t) for t in remove_stop_words(tokens)])
print([simple_stem(w) for w in ["gagne", "gagnez", "gagner", "gagnees", "rapidement", "rapide"]])
```

### Explication du code

- La liste de mots vides est volontairement courte et écrite à la main ; les **négations sont conservées**, sinon « pas du tout décevant » perdrait son sens.
- Le raciniseur retire le plus long suffixe connu en gardant au moins 3 lettres. Il regroupe « gagne », « gagnez », « gagner », « gagnees » sous « gagn ». Il est grossier : « rapidement » et « rapide » donnent « rapid », mais beaucoup d'autres familles de mots ne seraient pas regroupées, et certains mots seraient abîmés. Un vrai raciniseur (Snowball) ou un lemmatiseur (spaCy) est bien plus précis.

### Sortie attendue

```text output
tokens:           ['ce', 'produit', 'n', 'est', 'pas', 'du', 'tout', 'decevant', 'livraison', 'rapide', 'et', 'je', 'recommande', 'vraiment']
without stopwords: ['produit', 'n', 'pas', 'tout', 'decevant', 'livraison', 'rapide', 'recommande', 'vraiment']
stems:            ['produit', 'n', 'pas', 'tout', 'decevant', 'livraison', 'rapid', 'recommand', 'vraiment']
['gagn', 'gagn', 'gagn', 'gagn', 'rapid', 'rapid']
```

## Erreurs fréquentes

- **Construire le vocabulaire sur toutes les données** (test compris) : fuite.
- **Supprimer les négations** en analyse de sentiment.
- **Appliquer un prétraitement différent** à l'entraînement et à la prédiction.
- **Oublier le jeton `<unk>`** : erreur à la première phrase contenant un mot nouveau.
- **Problèmes d'encodage** : lire un fichier sans `encoding="utf-8"` sous Windows (« Ã© »).
- **Sur-normaliser** : retirer majuscules et ponctuation alors qu'ils sont informatifs pour la tâche (spam).

## Exercices guidés

1. Ajoutez à `normalize` le remplacement des montants (« 500 EUR », « 2,49 EUR ») par `<montant>` avant le remplacement des nombres.
2. Calculez le nombre de jetons moyen par message pour les spams et les messages normaux.
3. Quelle proportion des occurrences de jetons du corpus d'entraînement est couverte par les 50 jetons les plus fréquents ?
4. Écrivez `ngrams(tokens, n)` qui renvoie la liste des n-grammes, et affichez les 5 bigrammes les plus fréquents des spams.

Corrigés : [corrections.md](corrections.md), section « Leçon 8.1 ».

## Défis autonomes

- Tracez la loi de Zipf (fréquence en fonction du rang, échelles logarithmiques) pour les avis `reviews_fr.csv`.
- Ajoutez une caractéristique « proportion de majuscules » dans le message d'origine et comparez-la entre spams et messages normaux.

## Quiz

1. Qu'est-ce qu'un jeton ? Donnez deux façons de tokeniser « l'argent ».
2. Pourquoi remplacer les URL par `<url>` ?
3. Pourquoi ne faut-il pas supprimer « pas » en analyse de sentiment ?
4. Différence entre racinisation et lemmatisation ?
5. Pourquoi le vocabulaire se construit-il sur le seul jeu d'entraînement ?
6. Que dit la loi de Zipf ?

Réponses : [corrections.md](corrections.md), section « Leçon 8.1 ».

## Résumé

- Normaliser (Unicode, casse, accents, jetons spéciaux), tokeniser, construire un vocabulaire sur l'entraînement.
- Mots vides et racinisation réduisent la variabilité mais détruisent de l'information : à valider par l'expérience.
- `<pad>` et `<unk>` sont indispensables ; les mêmes traitements s'appliquent à l'entraînement et à la prédiction.

## Pour aller plus loin

- Leçon suivante : [Sac de mots, TF-IDF et classification de textes](02_bow_tfidf_classification.md).
