# Leçon 1.2 — Fonctions, modules, fichiers et exceptions

## Prérequis

- Leçon 1.1 (types, collections, boucles).

## Objectifs d'apprentissage

- Écrire des fonctions avec paramètres par défaut, valeurs de retour et docstrings.
- Utiliser les expressions `lambda` et les compréhensions de listes et de dictionnaires.
- Organiser son code en modules et comprendre `import`.
- Lire et écrire des fichiers texte et CSV avec `pathlib` et `csv`.
- Gérer les erreurs avec `try / except / else / finally` et lever ses propres exceptions.

## Concepts clés

- **Fonction** (*function*) : bloc de code nommé, réutilisable, qui reçoit des arguments et renvoie un résultat.
- **Docstring** : chaîne de documentation placée en première ligne d'une fonction ; affichée par `help()`.
- **Lambda** : petite fonction anonyme d'une seule expression, souvent passée en argument (`key=` de `sorted`).
- **Compréhension** (*comprehension*) : syntaxe compacte pour construire une liste, un ensemble ou un dictionnaire à partir d'un itérable.
- **Module / paquet** (*module / package*) : un fichier `.py` / un dossier de modules que l'on importe.
- **Gestionnaire de contexte** (*context manager*) : objet utilisé avec `with` qui garantit une action de fin (fermer un fichier).
- **Exception** : objet qui signale une erreur et interrompt le flux normal tant qu'elle n'est pas capturée.

## Explication détaillée

### Pourquoi des fonctions ?

Un script d'analyse grossit vite. Si vous copiez-collez le même calcul trois fois, vous aurez trois endroits à corriger le jour où une erreur est découverte. Une fonction donne un **nom** à une opération (`compute_average`), isole ses **entrées** (les paramètres) et sa **sortie** (le `return`). Une bonne fonction :

- fait une seule chose et porte un nom qui la décrit (un verbe) ;
- ne modifie pas en cachette ses arguments (pas d'effet de bord surprenant) ;
- renvoie une valeur plutôt que de l'afficher, pour pouvoir la réutiliser et la tester ;
- est documentée par une docstring et, idéalement, des annotations de type (`def f(x: float) -> float`).

**Piège classique** : ne jamais utiliser une liste ou un dictionnaire comme valeur par défaut (`def f(x, cache=[])`). La valeur par défaut est créée une seule fois et partagée entre tous les appels. Utilisez `None` puis créez la liste dans la fonction.

### Arguments positionnels, nommés, `*args` et `**kwargs`

`def scale(values, factor=1.0)` peut être appelée `scale(v, 2)` ou `scale(v, factor=2)`. Les arguments nommés rendent les appels lisibles. `*args` collecte les arguments positionnels supplémentaires dans un tuple, `**kwargs` les arguments nommés supplémentaires dans un dictionnaire : vous les rencontrerez dans les bibliothèques (pandas, scikit-learn) qui transmettent des options à d'autres fonctions.

### Lambda et compréhensions

`lambda x: x ** 2` est équivalent à une fonction `def carre(x): return x ** 2`, sans nom. On l'utilise quand une fonction attend une autre fonction en paramètre : `sorted(etudiants, key=lambda e: e["moyenne"])`.

Une compréhension de liste remplace le motif « liste vide + boucle + append » :

```python no-run
squares = []
for x in range(5):
    if x % 2 == 0:
        squares.append(x ** 2)
# équivalent :
squares = [x ** 2 for x in range(5) if x % 2 == 0]
```

Elle est plus courte et souvent plus rapide. Restez raisonnable : si la compréhension ne tient pas sur une ligne lisible, revenez à une boucle.

### Modules et paquets

`import math` charge le module standard `math`. `from statistics import mean` importe un seul nom. `import numpy as np` donne un alias conventionnel. Quand vous écrivez `utils.py` dans votre dossier, `import utils` fonctionne si le dossier est dans le chemin de recherche (`sys.path`), ce qui est le cas du dossier du script lancé. Un **paquet** est un dossier contenant des modules (et souvent un fichier `__init__.py`). Ce dépôt contient par exemple le paquet `lab/`.

Le bloc `if __name__ == "__main__":` contient le code qui ne doit s'exécuter que lorsque le fichier est lancé directement, et pas lorsqu'il est importé. C'est indispensable pour écrire des scripts réutilisables.

### Fichiers

`pathlib.Path` représente un chemin de façon indépendante du système (`/` sous Linux, `\` sous Windows). On lit un petit fichier avec `Path("f.txt").read_text(encoding="utf-8")`. Pour les fichiers plus gros ou ligne par ligne, on utilise `with open(...) as f:` : le fichier est fermé automatiquement, même en cas d'erreur.

**Précisez toujours `encoding="utf-8"`** : sous Windows, l'encodage par défaut peut être différent et les accents deviennent illisibles (« Ã© » au lieu de « é »).

Le module standard `csv` lit les fichiers CSV en gérant correctement les guillemets et les virgules dans les valeurs. `csv.DictReader` renvoie chaque ligne sous forme de dictionnaire. Avant pandas, c'est l'outil de base.

### Exceptions

Une erreur à l'exécution lève une exception (`ValueError`, `KeyError`, `FileNotFoundError`, `ZeroDivisionError`...). On la capture avec :

```python no-run
try:
    value = float(text)
except ValueError:
    value = None          # only if the conversion failed
else:
    print("conversion ok")  # only if no exception
finally:
    print("toujours exécuté")
```

Règles d'or : capturez l'exception **la plus précise possible** (jamais un `except:` nu qui masque toutes les erreurs, y compris les fautes de frappe), et ne capturez que ce que vous savez traiter. Pour signaler une donnée invalide dans votre propre code, levez une exception explicite : `raise ValueError("score must be between 0 and 20")`.

## Applications concrètes

- Une fonction `clean_price("19,90 EUR") -> 19.9` réutilisée dans tous les scripts de l'équipe.
- Un module `io_utils.py` qui centralise lecture et écriture des fichiers du projet.
- Une boucle de lecture qui ignore et journalise les lignes corrompues au lieu de planter.

## Exemples de code Python

### Exemple 1 — fonctions documentées et compréhensions

```python
def weighted_average(scores: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """Return the weighted average of the scores.

    If weights is None, every subject has the same weight.
    """
    if not scores:
        raise ValueError("scores must not be empty")
    if weights is None:
        weights = {subject: 1.0 for subject in scores}
    total_weight = sum(weights[subject] for subject in scores)
    return sum(weights[subject] * score for subject, score in scores.items()) / total_weight


scores = {"math": 14.0, "physics": 12.5, "french": 11.0}
print(round(weighted_average(scores), 2))
print(round(weighted_average(scores, {"math": 4, "physics": 3, "french": 3}), 2))

# Comprehensions
values = [3, 8, 12, 5, 20]
print([v * 2 for v in values if v > 4])
print({v: v ** 2 for v in values[:3]})

students = [("S01", 12.4), ("S02", 15.1), ("S03", 9.8)]
print(sorted(students, key=lambda item: item[1], reverse=True))
help(weighted_average)
```

### Explication du code

La fonction lève une `ValueError` si on lui donne un dictionnaire vide plutôt que de renvoyer un résultat absurde (division par zéro). La valeur par défaut `None` de `weights` évite le piège du dictionnaire mutable par défaut. Les deux `sum(...)` utilisent des **expressions génératrices**, cousines des compréhensions qui ne construisent pas de liste intermédiaire. `help()` affiche la docstring.

### Sortie attendue

```text output
12.5
12.65
[16, 24, 10, 40]
{3: 9, 8: 64, 12: 144}
[('S02', 15.1), ('S01', 12.4), ('S03', 9.8)]
Help on function weighted_average in module __main__:

weighted_average(
    scores: dict[str, float],
    weights: dict[str, float] | None = None
) -> float
    Return the weighted average of the scores.

    If weights is None, every subject has the same weight.
```

### Exemple 2 — écrire puis relire un CSV, en gérant les erreurs

```python
import csv
from pathlib import Path

output_dir = Path("outputs/module01")
output_dir.mkdir(parents=True, exist_ok=True)
csv_path = output_dir / "scores_demo.csv"

rows = [
    {"student_id": "S01", "score": "14.5"},
    {"student_id": "S02", "score": "abs"},
    {"student_id": "S03", "score": "9"},
    {"student_id": "S04", "score": "25"},
]
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["student_id", "score"])
    writer.writeheader()
    writer.writerows(rows)


def parse_score(text: str) -> float:
    """Convert a score to float and check that it lies in [0, 20]."""
    value = float(text)  # may raise ValueError
    if not 0 <= value <= 20:
        raise ValueError(f"score out of range: {value}")
    return value


valid, errors = {}, []
with csv_path.open(encoding="utf-8") as handle:
    for line_number, row in enumerate(csv.DictReader(handle), start=2):
        try:
            valid[row["student_id"]] = parse_score(row["score"])
        except ValueError as exc:
            errors.append(f"line {line_number}: {exc}")

print("Valid scores:", valid)
print("Errors:")
for message in errors:
    print("  -", message)
```

### Explication du code

- `mkdir(parents=True, exist_ok=True)` crée le dossier s'il n'existe pas, sans erreur s'il existe.
- `newline=""` est recommandé par le module `csv` pour éviter des lignes vides supplémentaires sous Windows.
- `parse_score` signale deux problèmes différents avec la même exception (`ValueError`) : texte non numérique et valeur hors bornes.
- La boucle capture l'erreur ligne par ligne : une mauvaise ligne n'empêche pas de traiter les autres, et on conserve un journal des erreurs. Le numéro de ligne commence à 2 car la ligne 1 est l'en-tête.

### Sortie attendue

```text output
Valid scores: {'S01': 14.5, 'S03': 9.0}
Errors:
  - line 3: could not convert string to float: 'abs'
  - line 5: score out of range: 25.0
```

### Exemple 3 — un module importable

```python
from pathlib import Path
import importlib
import sys

module_dir = Path("outputs/module01")
(module_dir / "grading.py").write_text(
    '"""Tiny grading module written by the lesson."""\n'
    "PASS_MARK = 10\n\n"
    "def grade(score):\n"
    "    if score >= 16:\n"
    "        return 'tres bien'\n"
    "    if score >= PASS_MARK:\n"
    "        return 'admis'\n"
    "    return 'ajourne'\n\n"
    "if __name__ == '__main__':\n"
    "    print('Run directly: demo', grade(12))\n",
    encoding="utf-8",
)
sys.path.insert(0, str(module_dir.resolve()))
grading = importlib.import_module("grading")
print(grading.PASS_MARK, [grading.grade(s) for s in (8, 12, 17)])
```

### Explication du code

Le code écrit un vrai fichier `grading.py`, ajoute son dossier au chemin de recherche des modules puis l'importe. Le bloc `if __name__ == '__main__':` n'est **pas** exécuté lors de l'import : le message « Run directly » n'apparaît pas. Lancez `python outputs/module01/grading.py` dans un terminal pour le voir s'afficher. Dans un vrai projet, on n'écrit pas un module depuis du code : on crée simplement le fichier dans VS Code.

### Sortie attendue

```text output
10 ['ajourne', 'admis', 'tres bien']
```

## Erreurs fréquentes

- **Afficher au lieu de renvoyer** : une fonction qui fait `print(resultat)` sans `return` renvoie `None`.
- **Valeur par défaut mutable** : `def f(acc=[])` accumule les valeurs entre les appels.
- **`except:` ou `except Exception:` trop larges** : ils masquent les vrais bugs. Capturez `ValueError`, `KeyError`...
- **Oublier `encoding="utf-8"`** sous Windows : accents corrompus.
- **Chemins écrits à la main avec `\`** : `"data\new.csv"` contient `\n`, un retour à la ligne ! Utilisez `pathlib` ou `/`.
- **Nommer son fichier comme un module existant** (`csv.py`, `numpy.py`) : il masque le vrai module à l'import.

## Exercices guidés

1. Écrivez `clean_price(text: str) -> float` qui transforme `"19,90 EUR"`, `" 24.9 "` et `"189"` en réels. *Indice : `strip()`, `replace(",", ".")`, `removesuffix("EUR")`.*
2. Avec une compréhension de dictionnaire, construisez `{produit: prix}` à partir de deux listes `["Souris", "Clavier"]` et `[24.9, 39.9]`. *Indice : `zip`.*
3. Écrivez `safe_divide(a, b)` qui renvoie `None` et affiche un avertissement si `b == 0`, en utilisant `try / except ZeroDivisionError`.
4. Écrivez un fichier texte de 3 lignes dans `outputs/module01/notes.txt`, relisez-le et comptez les lignes non vides.

Corrigés : [corrections.md](corrections.md), section « Leçon 1.2 ».

## Défis autonomes

- Écrivez une fonction `describe(values)` qui renvoie un dictionnaire `{"count", "min", "max", "mean"}` et lève une `ValueError` claire si la liste est vide ou contient autre chose que des nombres.
- Transformez l'exemple 2 en module `score_reader.py` utilisable à la fois comme script (`python score_reader.py fichier.csv`) et comme module importable. *Piste : `sys.argv` ou `argparse`.*

## Quiz

1. Que renvoie une fonction qui n'a pas d'instruction `return` ?
2. Pourquoi ne faut-il pas écrire `def f(x, items=[])` ?
3. Réécrivez en compréhension : `r = []; for w in mots: r.append(w.lower())`.
4. À quoi sert `if __name__ == "__main__":` ?
5. Quel est l'intérêt de `with open(...) as f:` ?
6. Quelle exception lève `float("abc")` ? Et `{}["x"]` ?

Réponses : [corrections.md](corrections.md), section « Leçon 1.2 ».

## Résumé

- Une fonction = un nom, des entrées, une sortie, une docstring. Renvoyez, n'affichez pas.
- Les compréhensions construisent des collections de façon concise ; `lambda` sert de petite fonction jetable.
- `import` charge des modules ; `if __name__ == "__main__":` protège le code de démonstration.
- `pathlib` + `with open(..., encoding="utf-8")` pour des fichiers fiables sur tous les systèmes.
- Capturez des exceptions précises et levez des erreurs explicites pour les données invalides.

## Pour aller plus loin

- Exercice **B02** (`exercises/beginner/B02_sales_records`) : agrégation de ventes avec des fonctions.
- Exercice **I01** (`exercises/intermediate/I01_data_processing_script`) : script réutilisable en ligne de commande.
- Explorez la bibliothèque standard hors ligne : `python -m pydoc pathlib`.
- Leçon suivante : [Programmation orientée objet et environnements virtuels](03_poo_et_environnements.md).
