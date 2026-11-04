# Projet 4 — Détection de spam

**Difficulté** : ★★★☆ — **Modules** : 5, 8, 10 — **Durée indicative** : 5 à 8 heures

## Énoncé du problème

Un opérateur veut filtrer automatiquement les SMS indésirables (arnaques, faux colis, faux remboursements). Contrainte forte : **un message légitime classé à tort comme spam est très coûteux** (un code de vérification ou un rendez-vous perdu). Le filtre doit donc privilégier la précision sur la classe « spam », tout en attrapant le plus de spams possible, et chaque décision doit pouvoir être expliquée.

## Objectifs d'apprentissage

- Construire un découpage sans fuite pour du texte contenant des doublons (`GroupShuffleSplit`, leçon 8.2).
- Comparer des représentations (mots, n-grammes de caractères) et des modèles (Naive Bayes, régression logistique).
- Choisir un seuil de décision sur le jeu de validation selon une contrainte métier (précision minimale).
- Analyser les erreurs et expliquer une décision (contribution des termes).
- Livrer un modèle utilisable en ligne de commande.

## Données

`datasets/generated/sms_spam.csv` : 1 200 SMS **synthétiques** en français, colonnes `message_id`, `text`, `label` (`ham` ou `spam`), environ 26 % de spams.

Le générateur (`scripts/generate_datasets.py`) assemble des modèles de phrases ; il inclut des cas difficiles (messages légitimes au ton commercial : codes de vérification, livraisons, factures ; spams au ton familier), des messages sans accents ou avec une faute de frappe, et **2 % d'étiquettes volontairement inversées** pour simuler les erreurs d'annotation des vrais jeux de données. Seuls 742 textes sont distincts : les doublons imposent un découpage par groupes.

## Architecture

```text
04_spam_detection/
├── run.py                       entraîner/évaluer, ou classify "message..."
├── spam_filter/detector.py      normalisation, découpage, candidats, seuil, métriques, explication
├── tests/test_project.py
└── outputs/                     spam_model.joblib, metrics.json, errors.csv
```

## Étapes

1. **Normaliser** les textes : minuscules, accents retirés, URL, téléphones et nombres remplacés par des jetons (leçon 8.1).
2. **Découper** 60/20/20 avec `GroupShuffleSplit` sur le texte : aucune copie d'un même message des deux côtés (vérifier que le recouvrement entraînement/test vaut 0).
3. **Comparer** sur la validation : Naive Bayes sur les comptages de mots, régression logistique sur TF-IDF de mots (1-2 grammes), régression logistique sur TF-IDF de caractères (3-5 grammes). En cas d'égalité, préférer le modèle explicable.
4. **Choisir le seuil** sur la validation : le plus bas seuil qui atteint une précision d'au moins 98 % (pour maximiser le rappel sous cette contrainte).
5. **Évaluer une fois** sur le test : précision, rappel, F1, faux positifs, faux négatifs ; comparer seuil par défaut et seuil choisi.
6. **Analyser les erreurs** (`errors.csv`) et **expliquer** une décision par les termes qui y contribuent le plus.
7. **Livrer** : modèle et seuil sauvegardés, commande `classify`.

## Lancer la solution de référence

```bash
python projects/04_spam_detection/run.py
python projects/04_spam_detection/run.py classify "Votre colis est en attente, payez 1,99 EUR sur http://x.example" "On mange ensemble à midi ?"
python -m pytest projects/04_spam_detection -q
```

## Indicateurs et résultats attendus

| Indicateur | Valeur |
|---|---|
| Tailles entraînement / validation / test | 760 / 247 / 193 |
| Recouvrement de textes entraînement-test | 0 |
| F1 de validation des 3 candidats | ≈ 0,969 (égalité) → régression logistique (mots) retenue |
| Test : précision / rappel / F1 | ≈ 0,98 / 0,96 / 0,97 (1 faux positif, 2 faux négatifs) |

**Lecture des erreurs** : les 3 erreurs du test sont des messages dont l'étiquette a été volontairement inversée par le générateur (par exemple « Salut Maman, on se retrouve à 17h15 au stade ? » étiqueté spam). Le modèle a raison et l'étiquette a tort : l'analyse d'erreurs sert aussi à **auditer les données**. Sur de vrais SMS, les scores seraient plus faibles : ces résultats illustrent la méthode.

## Défis pour aller plus loin

- Ajoutez des variables non textuelles (longueur, part de majuscules, présence d'un montant) avec un `ColumnTransformer` et mesurez l'effet.
- Remplacez le découpage par groupes par un découpage aléatoire : de combien les scores changent-ils ? Pourquoi est-ce trompeur ?
- Écrivez 20 nouveaux messages difficiles vous-même et mesurez la performance du filtre sur eux.
- Entraînez le classifieur `EmbeddingBag` de la leçon 8.3 et comparez-le.

## Solution de référence

`spam_filter/detector.py` : `normalize`, `split`, `candidates`, `choose_threshold`, `metrics`, `explain`, `run`, `classify`.

## Pistes d'amélioration

- Seuil choisi par validation croisée plutôt que sur un seul jeu de validation.
- Calibration des probabilités (`CalibratedClassifierCV`).
- Boucle de retour : les messages signalés par les utilisateurs alimentent un nouvel entraînement, après vérification des étiquettes.
