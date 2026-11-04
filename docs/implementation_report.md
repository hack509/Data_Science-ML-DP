# Rapport d'implémentation et de vérification

Ce rapport décrit ce qui a été construit, ce qui a été réellement vérifié et ce qui reste à vérifier.

## Ce qui a été livré

| Élément | Contenu |
|---|---|
| Cours | 10 modules, 39 leçons, 10 fichiers de corrigés (quiz et exercices guidés) |
| Exercices | 21 (6 débutants, 9 intermédiaires, 6 avancés), avec énoncé, fichier à compléter, tests et solution séparée |
| Projets | 6 projets avec README, solution de référence, `run.py` et tests |
| Données | 17 jeux de données synthétiques reproductibles, dont 300 images, documentés dans `datasets/README.md` |
| Notebooks | 39 notebooks générés à partir des leçons |
| Interface locale | `scripts/run_learning_environment.py` : lecture, recherche, progression, sur 127.0.0.1, bibliothèque standard uniquement |
| Fichier HTML autonome | `offline_ai_learning_lab.html` : tout le contenu dans un seul fichier, sans ressource externe |
| PDF | un PDF par module et un cours complet, avec sommaire et numéros de page |
| Outils | vérification de l'environnement, installation, vérification hors ligne, progression, index du cours |

## Environnement de vérification

Windows 10 Pro 64 bits, Python 3.13.17, processeur 4 cœurs sans GPU, PyTorch 2.15.0+cpu. Versions complètes : [guide d'installation](installation_guide.md).

Modifications apportées à la machine pendant le projet :

- paquets ajoutés à `.venv` : seaborn, sympy, pytest, reportlab, torch (CPU) ;
- **Microsoft Visual C++ Redistributable x64 installé sur le système**, avec accord préalable, car `import torch` échouait sans lui ;
- fichier `offline_ai_learning_lab.pth` ajouté à `.venv` pour que `import lab` fonctionne depuis n'importe quel dossier.

## Résultats des tests

Suite complète (`python -m pytest -q`, 22 minutes sur la machine de test) : **264 tests réussis, 1 échec**. Cet échec était un lien du README vers le présent rapport, qui n'existait pas encore. Le résultat après sa création et les dernières corrections figure dans la dernière section.

| Domaine | Ce que vérifient les tests |
|---|---|
| Leçons | structure obligatoire, longueur, et **exécution réelle de tout le code** des 39 leçons |
| Corrigés | exécution réelle du code des 10 fichiers |
| Exercices | les solutions passent tous les tests ; les fichiers à compléter ne contiennent pas la solution ; sections de l'énoncé présentes |
| Projets | suite de tests de chacun des 6 projets, en mode réduit pour le projet 5 |
| Données | génération identique octet par octet pour les CSV et JSON ; tailles ; problèmes documentés présents ; message clair si un fichier manque |
| Modèles | entraînement et sauvegarde / rechargement scikit-learn et PyTorch sur CPU |
| PDF | génération, polices embarquées, couverture des symboles, découpe des lignes de code |
| Progression | aucune écriture sans action explicite ; ligne de commande ; validation |
| Hors ligne | blocage du réseau, analyse du code source à la recherche de téléchargements |
| Interface locale | pages, recherche, refus des chemins hors dépôt, écoute limitée à 127.0.0.1 |
| Fichier HTML | contenu complet, liens internes valides, aucune ressource réseau, fichiers générés exclus |
| Documentation | tous les liens Markdown internes existent ; l'index est à jour |

`python setup/verify_offline.py` affiche **OFFLINE READY** : imports, jeux de données, entraînements scikit-learn et PyTorch et rendu de figures réussissent avec les connexions non locales bloquées.

Les sorties affichées dans les leçons sont les sorties réelles, insérées par `python -m lab.lesson_runner --update-outputs`. Ce mécanisme et les relectures ont conduit à corriger de nombreuses explications écrites avant exécution, par exemple sur l'effet des bigrammes ou la comparaison d'architectures.

## Vérifications manuelles

- **PDF** : pages rendues et examinées (sommaire, accents, symboles, listes).
- **Figures** : plusieurs figures ouvertes et corrigées (titres inexacts, étiquettes qui se chevauchaient) ; palette catégorielle contrôlée avec un simulateur de daltonisme.
- **Fichier HTML dans Chrome** : accueil, leçon, exercice, recherche, galerie, thème sombre, largeur de 380 px ; bouton de progression testé depuis `file://` ; solutions fermées par défaut.
- **Notebook** : exécution réelle d'un notebook généré, avec nbconvert et avec nbclient.

## Critères d'acceptation

| N° | Critère | État |
|---|---|---|
| 1 | Structure claire | Rempli |
| 2 | Leçons substantielles | Rempli (longueur minimale testée, contenu rédigé) |
| 3 | Exemples exécutables | Rempli (exécutés par les tests) |
| 4 | Données générées sans Internet | Rempli (génération sous blocage réseau) |
| 5 | Modèles de ML sur CPU | Rempli |
| 6 | Modèle PyTorch sur CPU | Rempli |
| 7 | Solutions séparées | Rempli |
| 8 | Progression locale | Rempli |
| 9 | PDF hors ligne | Rempli pour la génération ; non exécutée réseau coupé (voir ci-dessous) |
| 10 | README complet | Rempli |
| 11 | Tests réels | Rempli |
| 12 | Aucun service externe | Rempli (analyse du code et blocage réseau) |

## Ce qui n'a pas été vérifié

- **Réseau physiquement coupé** : le blocage s'applique dans le processus Python de `verify_offline.py` ; la suite de tests n'a pas été lancée Wi-Fi coupé, et l'export PDF n'a pas tourné sous blocage (seules ses polices sont contrôlées).
- **Installation neuve** : `setup_environment.py --install` n'a pas été exécuté de bout en bout dans un environnement vide, et l'installation depuis des wheels n'a pas été essayée.
- **Autres plateformes** : Linux, macOS et Python 3.11 ou 3.12 ne sont pas testés, malgré leur mention dans les versions prises en charge. Les minimums de `requirements.txt` sont des estimations ; seules les versions du guide d'installation ont été exécutées.
- **Autres navigateurs** : le fichier HTML n'a été ouvert que dans Chrome.
- **GPU** : les instructions optionnelles n'ont pas été essayées (pas de GPU sur la machine).
- **Poids pré-entraînés** : le bloc de la leçon 8.4 est illustratif et n'a pas été exécuté.
- **Parcours apprenant** : aucun exercice n'a été résolu par une personne ; seule B01 a été lancée en mode apprenant (les tests échouent comme prévu). La difficulté et les durées indiquées sont des estimations.
- **Export et import de progression** du fichier HTML, option `--jupyter` de l'interface, apparence de l'interface locale : non testés dans un navigateur (les réponses HTTP le sont).
- **Rendu de la coloration et du texte des PDF complets** : échantillon de pages seulement.

## Limites connues

- Tous les résultats portent sur des données synthétiques et ne prouvent rien sur le monde réel.
- Les PNG et le fichier `.npz` sont régénérés mais leur identité octet par octet n'est pas testée.
- Avertissement interne à seaborn 0.13 avec matplotlib 3.11 (`vert: bool`), sans effet sur les résultats.
- Le fichier HTML permet de lire, pas d'exécuter le code ; sa progression reste dans le navigateur (export JSON possible).
- La galerie du fichier HTML dépend des figures présentes dans `outputs/` au moment de la génération.
- Le dépôt contient un ancien dossier `__pycache__/` à la racine, antérieur à ce projet et conservé tel quel.

## Dernière vérification

Après création de ce rapport et dernières corrections (version minimale de scikit-learn, lien vers les notebooks dans l'interface, commentaire de `requirements.txt`), les tests des outils, des utilitaires et des données (`tests/test_tools.py`, `tests/test_lab_utilities.py`, `tests/test_datasets.py`) ont été relancés : **61 tests réussis, 0 échec**, dont celui qui échouait. Les tests des leçons, exercices et projets n'ont pas été relancés après ces changements, qui ne les concernent pas.
