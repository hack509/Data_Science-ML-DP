# Dépannage

## Installation et environnement

**`python` n'est pas reconnu** — Python n'est pas dans le PATH. Réinstallez en cochant « Add python.exe to PATH », ou utilisez le lanceur `py` sous Windows (`py -3.13 -m venv .venv`).

**`Activate.ps1 cannot be loaded because running scripts is disabled`** (PowerShell) — autorisez les scripts pour votre utilisateur : `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, ou utilisez `.venv\Scripts\activate.bat` dans `cmd`, ou appelez directement `.venv\Scripts\python.exe`.

**`ModuleNotFoundError: No module named 'numpy'` (ou un autre paquet)** — l'environnement virtuel n'est pas actif, ou VS Code utilise un autre interpréteur. Vérifiez avec `python setup/check_environment.py` (ligne « Virtual env ») et sélectionnez `.venv` dans VS Code.

**`ModuleNotFoundError: No module named 'lab'`** — lancez les scripts depuis la racine du dépôt, ou enregistrez le dépôt dans l'environnement (opération hors ligne) : `python setup/setup_environment.py`.

**`OSError: [WinError 126] ... c10.dll` en important torch** — le Microsoft Visual C++ Redistributable x64 manque. Installez `vc_redist.x64.exe` (site de Microsoft, droits administrateur), puis relancez `python setup/check_environment.py`.

**pip échoue pendant l'installation** — mettez pip à jour (`python -m pip install --upgrade pip`) ; vérifiez la version de Python (3.11 à 3.13, 64 bits) ; derrière un proxy, configurez `pip config set global.proxy http://...`.

## Données

**`MissingDatasetError: Dataset 'students' is missing`** — générez les données (hors ligne) : `python scripts/generate_datasets.py`.

**`FileNotFoundError: datasets/generated/...`** — le code des leçons utilise des chemins relatifs à la racine : lancez Python depuis la racine du dépôt (les notebooks générés s'y placent automatiquement).

**Les accents s'affichent mal (« Ã© »)** — un fichier a été lu sans `encoding="utf-8"`. Dans un terminal Windows qui affiche mal les accents, utilisez `set PYTHONUTF8=1` (cmd) ou `$env:PYTHONUTF8=1` (PowerShell) ; le terminal intégré de VS Code les affiche correctement.

## Exécution du code

**Les figures ne s'affichent pas** — les leçons enregistrent les figures dans `outputs/<module>/` : ouvrez les fichiers PNG dans VS Code. Dans un script, ajoutez `plt.show()` ; dans Jupyter, elles s'affichent sous la cellule.

**`RuntimeWarning: More than 20 figures have been opened`** — fermez les figures avec `plt.close(fig)` après les avoir enregistrées.

**`MatplotlibDeprecationWarning: vert: bool was deprecated`** lors d'un `sns.boxplot` — avertissement interne à seaborn 0.13 avec matplotlib 3.11, sans conséquence sur le résultat. Il disparaîtra avec une version de seaborn compatible.

**Le code d'une leçon est lent** — les leçons des modules 6 à 9 entraînent des réseaux : comptez de 10 secondes à 1 minute et demie par leçon sur un processeur de portable. Fermez les autres applications lourdes ; `torch.get_num_threads()` indique le nombre de cœurs utilisés.

**Résultats légèrement différents de la sortie attendue** — les sorties des leçons ont été produites avec les versions de la configuration testée. Une autre version de NumPy, scikit-learn ou PyTorch, ou un autre processeur, peut changer les derniers chiffres des résultats aléatoires ou des entraînements.

## Tests

**Les tests d'un exercice échouent avec `NotImplementedError`** — c'est normal tant que vous n'avez pas complété `exercise.py`.

**Vérifier la solution de référence d'un exercice** — `LAB_EXERCISE_TARGET=solution python -m pytest exercises/...` (PowerShell : `$env:LAB_EXERCISE_TARGET="solution"; python -m pytest exercises/...`).

**Tests trop longs** — `python -m pytest -m "not slow" -q` ignore les entraînements de réseaux et les exemples les plus coûteux.

## Interface locale et Jupyter

**`OSError: [Errno 98] Address already in use` / port occupé** — choisissez un autre port : `python scripts/run_learning_environment.py --port 9000`.

**Le navigateur ne s'ouvre pas** — ouvrez manuellement l'adresse affichée (`http://127.0.0.1:8765/`).

**JupyterLab affiche un avertissement « Kernel is running over TCP without encryption »** — message informatif : le noyau communique uniquement sur 127.0.0.1 (votre machine).

## PDF

**`reportlab is not installed`** — installez les dépendances (étape en ligne) : `python -m pip install -r requirements.txt`.

**Un symbole apparaît comme un carré** — la police DejaVu ne contient pas ce caractère (émoji, symbole rare). Remplacez-le dans le Markdown par un équivalent courant.
