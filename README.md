# Offline AI Learning Lab

Un environnement d'apprentissage complet, en français et **hors ligne**, pour apprendre Python pour la data science, les mathématiques et statistiques, l'analyse et la visualisation de données, le Machine Learning, le Deep Learning, les réseaux de neurones, le traitement du langage et la vision par ordinateur, jusqu'à des projets de bout en bout.

**10 modules · 39 leçons · 21 exercices avec tests et solutions · 6 projets · 17 jeux de données générés localement.**

Les leçons et énoncés sont en français ; le code, les noms de variables et les commentaires sont en anglais. Tous les exemples s'exécutent sur un processeur ordinaire : **aucun GPU, aucune clé d'API, aucun service externe.**

## Consulter le cours sans rien installer

Téléchargez le fichier **[`offline_ai_learning_lab.html`](offline_ai_learning_lab.html)** (environ 2,5 Mo) et ouvrez-le dans un navigateur, même sans Internet. Ce fichier unique contient les leçons, corrigés, exercices (solutions masquées), projets, aperçus des données, figures et le code du dépôt, avec recherche, thème sombre et suivi de progression.

Sur GitHub, utilisez le bouton « Download raw file » : GitHub n'affiche que le code source des fichiers HTML. Ce fichier sert à **lire** ; pour exécuter le code, installez le dépôt comme ci-dessous.

## Installation (en ligne, une seule fois)

Prérequis : Python 3.11 à 3.13, 64 bits (testé sous Windows 10 avec Python 3.13).

```bash
python setup/setup_environment.py --install   # creates .venv, installs packages, generates datasets
.venv\Scripts\Activate.ps1                     # Windows PowerShell
# source .venv/bin/activate                    # Linux / macOS
python setup/check_environment.py              # packages and versions
python setup/verify_offline.py                 # real tasks with the network blocked
```

Sous Windows, PyTorch a besoin du Microsoft Visual C++ Redistributable x64 (voir le [dépannage](docs/troubleshooting.md)). Détails et versions testées : [guide d'installation](docs/installation_guide.md). Installation sur une machine sans Internet : [utilisation hors ligne](docs/offline_usage.md).

## Usage quotidien (sans Internet)

```bash
python scripts/run_learning_environment.py --open   # local interface on http://127.0.0.1:8765
python -m jupyter lab                                # notebooks
python -m pytest exercises/beginner/B01_student_averages -q   # check an exercise
python projects/02_house_price_prediction/run.py     # run a reference project
python scripts/progress.py show                      # your progress
```

Lancez toujours les commandes depuis la racine du dépôt. Pour bien commencer : lisez le [parcours d'apprentissage](docs/learning_roadmap.md), puis la [leçon 1.1](courses/01_python_for_data_science/01_bases_python.md).

## Contenu

| Module | Thème | Leçons |
|---|---|---|
| [1](courses/01_python_for_data_science/README.md) | Python pour la Data Science | 5 |
| [2](courses/02_mathematics_and_statistics/README.md) | Mathématiques et statistiques | 5 |
| [3](courses/03_data_analysis/README.md) | Analyse de données | 4 |
| [4](courses/04_data_visualization/README.md) | Visualisation de données | 3 |
| [5](courses/05_machine_learning/README.md) | Machine Learning | 6 |
| [6](courses/06_deep_learning/README.md) | Deep Learning avec PyTorch | 4 |
| [7](courses/07_neural_networks/README.md) | Réseaux de neurones | 3 |
| [8](courses/08_natural_language_processing/README.md) | Traitement du langage (NLP) | 4 |
| [9](courses/09_computer_vision/README.md) | Vision par ordinateur | 3 |
| [10](courses/10_ai_projects/README.md) | Projets, déploiement, IA responsable | 2 |

Index complet : [docs/course_index.md](docs/course_index.md).

## Organisation du dépôt

```text
courses/      lessons (Markdown), one corrections.md per module
exercises/    statements, files to complete, tests (beginner, intermediate, advanced)
solutions/    reference solutions and answer key, kept separate
projects/     six end-to-end projects (README, code, tests, run.py)
datasets/     raw/ and generated/ local data; documented in datasets/README.md
notebooks/    one notebook per lesson, generated from the Markdown files
scripts/      datasets, index, PDF, notebooks, local interface, offline HTML, progress
setup/        environment check, installation, offline verification
lab/          shared Python utilities
docs/         roadmap, installation, offline usage, glossary, troubleshooting, report
tests/        automated tests
progress/     progress.json and your learning log
```

Les dossiers `outputs/` (figures, modèles) et `exports/` (PDF) sont recréés localement et ignorés par Git.

## Générer les documents

```bash
python scripts/generate_datasets.py                 # datasets
python scripts/generate_course_index.py             # docs/course_index.md
python scripts/build_notebooks.py                   # notebooks/
python scripts/export_lessons_to_pdf.py --all-in-one  # exports/pdf/
python scripts/build_offline_html.py                # offline_ai_learning_lab.html
```

## Tests

```bash
python -m pytest -m "not slow" -q    # fast tests
python -m pytest -q                  # everything, including neural network training
```

Le code de chaque leçon et de chaque corrigé est réellement exécuté par les tests, et les sorties affichées dans les leçons sont les sorties réelles. Résultats et limites : [rapport d'implémentation](docs/implementation_report.md).

## Limites

- **Données synthétiques** : les jeux de données sont générés pour enseigner. Les scores obtenus illustrent des méthodes et ne prouvent rien sur le monde réel ; ne les utilisez pas pour de vraies décisions.
- **Hors ligne seulement après installation** : l'installation des paquets demande Internet ; lancez `python setup/verify_offline.py` pour confirmer.
- **Pas de modèle pré-entraîné** : la leçon 8.4 explique comment utiliser des poids téléchargés séparément, de façon optionnelle.
- **Plateforme** : testé sous Windows 10 ; Linux et macOS n'ont pas été testés.
- **Résultats numériques** : d'autres versions de bibliothèques ou un autre processeur peuvent changer les derniers chiffres des sorties, surtout pour les réseaux de neurones.
- **Export PDF** : pas d'images intégrées ni de coloration du code (voir [utilisation hors ligne](docs/offline_usage.md)).

En cas de problème : [dépannage](docs/troubleshooting.md).
