# Utilisation hors ligne

## Ce qui demande Internet, et ce qui ne le demande pas

| Activité | Internet ? |
|---|---|
| Installer Python, les paquets (`pip install`), PyTorch | **Oui, une seule fois** |
| Installer le Visual C++ Redistributable (Windows) | **Oui, une seule fois** |
| Installer des extensions VS Code | **Oui, une seule fois** |
| Lire les leçons (VS Code, interface locale, PDF) | Non |
| Générer les jeux de données | Non |
| Exécuter le code des leçons, les exercices, les tests | Non |
| Entraîner les modèles (scikit-learn, PyTorch CPU) | Non |
| Interface web locale, JupyterLab | Non (serveurs sur 127.0.0.1) |
| Générer les PDF et les notebooks | Non |
| Utiliser des modèles pré-entraînés (optionnel, leçon 8.4) | Oui pour le téléchargement initial, non ensuite |

**Ne considérez l'environnement comme hors ligne qu'après avoir lancé `python setup/verify_offline.py` avec succès.**

## Vérifier le fonctionnement hors ligne

```bash
python setup/verify_offline.py
```

Le script bloque toute connexion réseau non locale **dans le processus Python**, puis importe les bibliothèques, recharge et régénère les jeux de données, entraîne un modèle scikit-learn et un réseau PyTorch, enregistre une figure, vérifie les polices du PDF, et analyse le code source à la recherche de fonctions de téléchargement. Pour un test complet, coupez aussi le Wi-Fi et relancez la suite de tests :

```bash
python -m pytest -m "not slow" -q
```

## Usage quotidien (sans Internet)

```bash
.venv\Scripts\Activate.ps1                          # Windows (source .venv/bin/activate on Linux/macOS)
python scripts/run_learning_environment.py --open   # local interface on http://127.0.0.1:8765
python -m jupyter lab                               # notebooks
python -m lab.lesson_runner courses/05_machine_learning/02_regression.md   # run all code of a lesson
python -m pytest exercises/beginner/B01_student_averages -q               # check your exercise
python scripts/progress.py show                     # progress
```

## Ressources locales qui remplacent Internet

- **Documentation Python** : `python -m pydoc <module>` ou `help(objet)` dans un interpréteur ; `python -m pydoc -b` ouvre un navigateur de documentation local.
- **Documentation des bibliothèques** : `help(pd.DataFrame.merge)`, `help(torch.nn.Linear)`, ou Maj+clic / survol dans VS Code.
- **Glossaire** du cours : [glossary.md](glossary.md).
- **Index** de toutes les leçons : [course_index.md](course_index.md).

## Préparer une installation hors ligne (machine sans Internet)

Sur une machine **connectée**, avec la même version de Python et le même système d'exploitation que la machine cible :

```bash
python -m pip download -r requirements.txt -r requirements-dev.txt -d wheels/
python -m pip download -r requirements-torch.txt -d wheels/
```

Copiez le dépôt **et** le dossier `wheels/` (plusieurs centaines de Mo) sur la machine cible, puis :

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --no-index --find-links wheels/ -r requirements.txt -r requirements-dev.txt
python -m pip install --no-index --find-links wheels/ torch
python setup/setup_environment.py
python scripts/generate_datasets.py
python setup/verify_offline.py
```

`--no-index` interdit à pip de contacter Internet. Pensez aussi au Visual C++ Redistributable sous Windows (copiez `vc_redist.x64.exe`).

## Export PDF

```bash
python scripts/export_lessons_to_pdf.py                  # one PDF per module in exports/pdf/
python scripts/export_lessons_to_pdf.py --module 05      # a single module
python scripts/export_lessons_to_pdf.py --all-in-one     # + exports/pdf/cours_complet.pdf
python scripts/export_lessons_to_pdf.py --with-corrections
```

Le convertisseur (ReportLab + analyseur Markdown du dépôt) utilise les polices DejaVu fournies avec matplotlib : accents, lettres grecques et symboles mathématiques courants sont pris en charge, avec sommaire et numéros de page. **Limites connues** : les images référencées dans le Markdown ne sont pas intégrées (les leçons enregistrent leurs figures dans `outputs/`), les listes imbriquées sur plus d'un niveau sont aplaties, et le code n'a pas de coloration syntaxique. Alternative fiable pour une mise en page différente : ouvrir la leçon dans l'aperçu Markdown de VS Code, ou dans l'interface locale, puis « Imprimer » → « Enregistrer en PDF » du navigateur (hors ligne).

## Modèles pré-entraînés (optionnel)

Aucun exercice ne les exige. Pour expérimenter (module 8), téléchargez une fois les poids dans `models/` (ignoré par Git) pendant une phase connectée, puis chargez-les avec `local_files_only=True` et `HF_HUB_OFFLINE=1` : voir la leçon 8.4.
