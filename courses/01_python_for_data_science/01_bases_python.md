# Leçon 1.1 — Bases de Python : types, collections, conditions et boucles

## Prérequis

- Savoir ouvrir un terminal dans VS Code et lancer `python mon_script.py`.
- Environnement virtuel activé (`.venv`), voir `docs/installation_guide.md`.

## Objectifs d'apprentissage

À la fin de cette leçon, vous saurez :

- distinguer les types de base (`int`, `float`, `str`, `bool`, `None`) et convertir de l'un à l'autre ;
- choisir entre liste, tuple, ensemble (set) et dictionnaire selon le problème ;
- écrire des conditions et des boucles lisibles ;
- calculer des statistiques simples sur des données d'étudiants sans bibliothèque externe.

## Concepts clés

- **Variable** : un nom qui référence une valeur en mémoire. En Python, c'est la valeur qui a un type, pas le nom.
- **Type** (*data type*) : la nature d'une valeur (entier, réel, texte...). Il détermine les opérations possibles.
- **Mutable / immuable** (*mutable / immutable*) : une valeur mutable peut être modifiée sur place (liste, dictionnaire) ; une valeur immuable ne le peut pas (tuple, chaîne, nombre).
- **Itérable** (*iterable*) : tout objet que l'on peut parcourir avec `for` (liste, chaîne, dictionnaire, fichier...).
- **Hachable** (*hashable*) : une valeur dont on peut calculer une empreinte stable ; seules les valeurs hachables peuvent être clés de dictionnaire ou éléments d'un ensemble.

## Explication détaillée

### Les types de base

Python est un langage à **typage dynamique** : on n'annonce pas le type d'une variable, il est déduit de la valeur. `age = 21` crée un entier, `moyenne = 12.5` un réel (*float*), `nom = "Léa"` une chaîne de caractères. La fonction `type()` permet de vérifier.

Deux pièges reviennent sans cesse en analyse de données :

1. **Les nombres lus dans un fichier sont d'abord du texte.** `"12" + "3"` donne `"123"`, pas `15`. Il faut convertir avec `int()` ou `float()`.
2. **Les réels ne sont pas exacts.** `0.1 + 0.2` vaut `0.30000000000000004`, car les nombres sont stockés en binaire. On compare donc des réels avec une tolérance (`math.isclose`) et on arrondit à l'affichage, pas pendant les calculs.

La valeur spéciale `None` signifie « absence de valeur ». On la teste avec `is None`, jamais avec `== None`.

### Les quatre collections fondamentales

| Collection | Syntaxe | Ordonnée | Modifiable | Doublons | Usage typique |
|---|---|---|---|---|---|
| liste (*list*) | `[1, 2, 2]` | oui | oui | oui | séquence de mesures |
| tuple | `(48.85, 2.35)` | oui | non | oui | enregistrement fixe (coordonnées) |
| ensemble (*set*) | `{"A", "B"}` | non | oui | non | valeurs uniques, appartenance rapide |
| dictionnaire (*dict*) | `{"math": 14}` | oui (ordre d'insertion) | oui | clés uniques | associer une clé à une valeur |

Une bonne règle : si vous cherchez souvent « est-ce que x est dedans ? », utilisez un ensemble ou un dictionnaire. Le test `x in liste` parcourt toute la liste (temps proportionnel à sa taille), alors que `x in ensemble` est quasi instantané grâce au hachage.

### Conditions

`if / elif / else` exécute le premier bloc dont la condition est vraie. En Python, beaucoup de valeurs sont « fausses » dans un test : `0`, `0.0`, `""`, `[]`, `{}`, `None`. C'est pratique (`if notes:` signifie « si la liste n'est pas vide ») mais dangereux quand `0` est une valeur légitime : une note de 0 n'est pas une note manquante ! Dans ce cas, écrivez `if note is not None:`.

### Boucles

- `for element in iterable:` parcourt chaque élément. C'est la boucle de loin la plus utilisée.
- `enumerate(iterable)` donne en plus l'indice ; `zip(a, b)` parcourt deux séquences en parallèle ; `dict.items()` donne les couples (clé, valeur).
- `while condition:` répète tant que la condition est vraie ; utile quand on ne connaît pas le nombre d'itérations à l'avance (par exemple, un algorithme qui converge).
- `break` sort de la boucle, `continue` passe à l'itération suivante.

En data science, on évite d'écrire des boucles sur des millions de lignes : NumPy et pandas font ces calculs bien plus vite (leçons 1.4 et 1.5). Mais il faut maîtriser les boucles pour comprendre ce que ces bibliothèques font à notre place.

## Intuition mathématique

La moyenne arithmétique d'une série de notes x₁, ..., xₙ est :

```text
moyenne = (x₁ + x₂ + ... + xₙ) / n = (1/n) · Σ xᵢ
```

Une **moyenne pondérée** donne plus d'importance à certaines valeurs : avec des coefficients w₁, ..., wₙ,

```text
moyenne pondérée = Σ (wᵢ · xᵢ) / Σ wᵢ
```

C'est exactement ce que calcule une boucle qui accumule une somme : chaque passage ajoute un terme wᵢ · xᵢ. La même idée (somme pondérée) est au cœur de la régression linéaire et des neurones artificiels que vous verrez plus tard.

## Applications concrètes

- Calculer la moyenne d'un bulletin avec coefficients.
- Compter les occurrences de catégories (produits vendus, mots d'un texte) avec un dictionnaire.
- Dédoublonner une liste d'identifiants avec un ensemble.
- Filtrer des enregistrements selon une règle métier (par exemple, étudiants admis).

## Exemples de code Python

### Exemple 1 — types et conversions

```python
import math

age = 21
average = 12.5
name = "Léa"
is_admitted = True
missing = None

for value in [age, average, name, is_admitted, missing]:
    print(f"{value!r:>8} -> {type(value).__name__}")

# Values read from a file are strings: convert them before computing.
raw_score = "14.5"
print(raw_score + "1", "|", float(raw_score) + 1)

# Floating point numbers are approximations.
print(0.1 + 0.2 == 0.3, math.isclose(0.1 + 0.2, 0.3))
```

### Explication du code

La boucle parcourt une liste qui contient des valeurs de types différents. Dans la f-string, `!r` affiche la représentation de la valeur (avec les guillemets pour les chaînes) et `:>8` l'aligne à droite sur 8 caractères. On voit ensuite qu'additionner deux chaînes les concatène, et que la comparaison exacte de réels échoue alors que `math.isclose` réussit.

### Sortie attendue

```text output
      21 -> int
    12.5 -> float
   'Léa' -> str
    True -> bool
    None -> NoneType
14.51 | 15.5
False True
```

### Exemple 2 — collections : notes d'étudiants

```python
# Each student is a dictionary: keys are subjects, values are scores out of 20.
students = {
    "S01": {"math": 14.0, "physics": 12.5, "french": 11.0},
    "S02": {"math": 8.5, "physics": 9.0, "french": 13.5},
    "S03": {"math": 17.0, "physics": 15.5, "french": 9.5},
    "S04": {"math": 0.0, "physics": 10.0, "french": 12.0},
}
coefficients = {"math": 4, "physics": 3, "french": 3}

averages = {}
for student_id, scores in students.items():
    weighted_sum = 0.0
    total_weight = 0
    for subject, score in scores.items():
        weighted_sum += coefficients[subject] * score
        total_weight += coefficients[subject]
    averages[student_id] = weighted_sum / total_weight

for student_id, avg in averages.items():
    status = "admis" if avg >= 10 else "ajourné"
    print(f"{student_id}: {avg:5.2f} -> {status}")

admitted = {sid for sid, avg in averages.items() if avg >= 10}
print("Admis :", sorted(admitted))
best = max(averages, key=averages.get)
print("Meilleure moyenne :", best, round(averages[best], 2))
```

### Explication du code

- `students.items()` renvoie les couples (identifiant, dictionnaire de notes).
- La boucle interne accumule la somme pondérée et la somme des coefficients : c'est la formule de la moyenne pondérée.
- L'expression `"admis" if avg >= 10 else "ajourné"` est une **expression conditionnelle** (opérateur ternaire).
- `{... for ... if ...}` est une **compréhension d'ensemble** : elle construit l'ensemble des admis (détaillé en leçon 1.2).
- `max(averages, key=averages.get)` cherche la clé dont la valeur est maximale.
- Le 0 en mathématiques de S04 est une vraie note et est bien pris en compte.

### Sortie attendue

```text output
S01: 12.65 -> admis
S02: 10.15 -> admis
S03: 14.30 -> admis
S04:  6.60 -> ajourné
Admis : ['S01', 'S02', 'S03']
Meilleure moyenne : S03 14.3
```

### Exemple 3 — compter des catégories et boucle while

```python
sales = ["Souris", "Clavier", "Souris", "Ecran", "Souris", "Clavier", "Casque"]

counts = {}
for product in sales:
    counts[product] = counts.get(product, 0) + 1
print(counts)

# Sort by decreasing count: sorted() accepts a key function.
for product, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
    print(f"{product:<8} {'#' * count}")

# while: how many years for savings to double at 3% interest?
capital, years = 1000.0, 0
while capital < 2000:
    capital *= 1.03
    years += 1
print(f"Capital doublé après {years} ans : {capital:.2f} EUR")
```

### Explication du code

`counts.get(product, 0)` renvoie 0 si le produit n'a pas encore été vu, ce qui évite une `KeyError`. Le tri utilise une petite fonction anonyme (`lambda`) qui renvoie le nombre d'occurrences. La boucle `while` s'arrête dès que le capital dépasse 2000 : on ne savait pas d'avance combien d'itérations seraient nécessaires.

### Sortie attendue

```text output
{'Souris': 3, 'Clavier': 2, 'Ecran': 1, 'Casque': 1}
Souris   ###
Clavier  ##
Ecran    #
Casque   #
Capital doublé après 24 ans : 2032.79 EUR
```

## Erreurs fréquentes

- **Concaténer au lieu d'additionner** : `"12" + "3"`. Convertissez avec `float()` dès la lecture.
- **Tester un réel avec `==`** : utilisez `math.isclose` ou une tolérance.
- **Confondre 0 et absence de valeur** : `if note:` est faux pour une note de 0. Utilisez `if note is not None:`.
- **Modifier une liste pendant qu'on la parcourt** : construisez plutôt une nouvelle liste filtrée.
- **Utiliser une liste comme clé de dictionnaire** : `TypeError: unhashable type: 'list'`. Utilisez un tuple.
- **`KeyError`** en lisant une clé absente : utilisez `dict.get(cle, valeur_par_defaut)` ou testez `cle in dico`.
- **Indentation incohérente** : en Python l'indentation fait partie de la syntaxe ; configurez VS Code pour 4 espaces.

## Exercices guidés

1. **Conversion** : la liste `["12", "15.5", "abs", "9"]` provient d'un fichier. Construisez la liste des notes numériques en ignorant les valeurs non convertibles. *Indice : testez `valeur.replace(".", "", 1).isdigit()`.*
2. **Moyenne simple** : sans utiliser `sum()`, calculez avec une boucle la moyenne de `[12, 15.5, 9, 18]`.
3. **Dédoublonnage** : à partir de `["S01", "S02", "S01", "S03", "S02"]`, obtenez la liste des identifiants uniques triés. *Indice : un ensemble, puis `sorted()`.*
4. **Fréquences** : comptez les lettres de `"statistiques"` avec un dictionnaire et affichez la plus fréquente.

Corrigés : [corrections.md](corrections.md), section « Leçon 1.1 ».

Pour aller plus loin, l'exercice **B01** (`exercises/beginner/B01_student_averages`) automatise le calcul des moyennes avec des tests.

## Défis autonomes

- Écrivez un programme qui, pour chaque étudiant de l'exemple 2, affiche la matière où il est le plus faible.
- Simulez un livret d'épargne avec versement mensuel de 50 EUR et un taux annuel de 3 % (taux mensuel = 3 % / 12). Combien de mois pour atteindre 5000 EUR ?
- Sans `max()`, trouvez les deux meilleures moyennes en un seul parcours.

## Quiz

1. Que vaut `type(3 / 2)` ? Et `type(3 // 2)` ?
2. Quelle collection choisir pour stocker des identifiants uniques et tester rapidement l'appartenance ?
3. Pourquoi `0.1 + 0.2 == 0.3` est-il faux ?
4. Quelle est la différence entre `if x:` et `if x is not None:` quand `x = 0` ?
5. Un tuple peut-il être une clé de dictionnaire ? Et une liste ?
6. Que renvoie `{"a": 1}.get("b", 0)` ?

Réponses : [corrections.md](corrections.md), section « Leçon 1.1 ».

## Résumé

- Le type appartient à la valeur ; convertissez explicitement les données lues dans les fichiers.
- Liste = séquence modifiable ; tuple = enregistrement fixe ; ensemble = valeurs uniques ; dictionnaire = association clé → valeur.
- Attention aux valeurs « fausses » (`0`, `""`, `[]`) : testez `None` explicitement.
- `for` + `enumerate`, `zip`, `items()` couvrent l'essentiel des parcours.
- Une moyenne pondérée est une somme de produits divisée par la somme des poids : une idée que l'on retrouvera dans tout le Machine Learning.

## Pour aller plus loin

- Exercice **B01** — moyennes d'étudiants (`exercises/beginner/B01_student_averages`).
- Exercice **B02** — analyse de ventes (`exercises/beginner/B02_sales_records`).
- Tapez `help(dict)` ou `help(str.split)` dans un interpréteur Python : la documentation intégrée est disponible hors ligne.
- Leçon suivante : [Fonctions, modules, fichiers et exceptions](02_fonctions_modules_fichiers.md).
