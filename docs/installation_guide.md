# Guide d'installation

L'installation est la **seule étape qui nécessite Internet** (téléchargement des paquets Python). Une fois terminée, tout le laboratoire fonctionne hors ligne : leçons, données, exercices, notebooks, interface locale, PDF.

## Configuration testée

| Élément | Version testée |
|---|---|
| Système | Windows 10 Pro 64 bits (19045) |
| Python | 3.13.17 (64 bits) |
| numpy / pandas | 2.5.4 / 3.1.0 |
| matplotlib / seaborn | 3.11.2 / 0.13.2 |
| scipy / scikit-learn | 1.18.1 / 1.9.1 |
| sympy / pillow | 1.14.0 / 13.0.0 |
| jupyterlab / nbformat | 4.6.5 / 5.11.1 |
| reportlab | 5.1.0 |
| torch | 2.15.0+cpu |
| pytest | 9.2.0 |

Versions de Python prises en charge : **3.11 à 3.13**, 64 bits. Linux et macOS doivent fonctionner avec les mêmes commandes (chemins `bin/` au lieu de `Scripts\`), mais n'ont pas été testés pour cette version du dépôt.

Matériel : un processeur récent et 4 Go de mémoire suffisent ; **aucune carte graphique n'est nécessaire**. Espace disque : environ 1,5 Go pour l'environnement virtuel (dont ≈ 700 Mo pour PyTorch CPU).

## Installation pas à pas (en ligne, une seule fois)

### 1. Installer Python

Téléchargez Python 3.11, 3.12 ou 3.13 (64 bits) depuis le site officiel et cochez « Add python.exe to PATH ». Vérifiez dans un terminal :

```bash
python --version
```

### 2. Créer l'environnement virtuel et installer les paquets

Depuis la racine du dépôt :

```bash
python setup/setup_environment.py --install
```

Ce script crée `.venv/`, enregistre le dépôt dans l'environnement (pour que `import lab` fonctionne depuis n'importe quel dossier), installe `requirements.txt`, `requirements-torch.txt` (PyTorch **CPU**) et `requirements-dev.txt` (pytest), génère les jeux de données et vérifie l'environnement.

Installation manuelle équivalente :

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1                    # Windows PowerShell
# source .venv/bin/activate                   # Linux / macOS
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-torch.txt
python -m pip install -r requirements-dev.txt
python setup/setup_environment.py             # registers the repository (offline step)
python scripts/generate_datasets.py           # offline step
```

Sans PyTorch (modules 6 à 9 limités) : `python setup/setup_environment.py --install --no-torch`.

### 3. Windows : Visual C++ Redistributable

PyTorch a besoin du **Microsoft Visual C++ Redistributable x64**. S'il manque, `import torch` échoue avec `OSError: [WinError 126] ... c10.dll`. Installez `vc_redist.x64.exe` depuis le site de Microsoft (droits administrateur requis), puis relancez la vérification. Sur la machine de test, il a fallu l'installer.

### 4. Vérifier

```bash
python setup/check_environment.py     # packages, versions, datasets
python setup/verify_offline.py        # runs real tasks with the network blocked
python -m pytest -m "not slow" -q     # fast tests (about 10 min)
python -m pytest -q                   # all tests (about 25 min on a 4-core laptop CPU)
```

### 5. Configurer VS Code

- Ouvrez le dossier du dépôt dans VS Code.
- Commande « Python: Select Interpreter » → choisissez `.venv`.
- Extensions utiles (à installer pendant la phase en ligne) : Python, Jupyter.
- L'aperçu Markdown de VS Code (Ctrl+Maj+V) affiche les leçons hors ligne.

## GPU (optionnel)

Le cours n'en a pas besoin. Avec une carte NVIDIA et des pilotes récents, vous pouvez remplacer PyTorch CPU par une version CUDA (téléchargement de plusieurs Go) :

```bash
python -m pip uninstall torch
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124   # choose the CUDA version supported by your driver
python -c "import torch; print(torch.cuda.is_available())"
```

Les leçons fonctionnent alors à l'identique ; pour utiliser le GPU dans votre propre code : `device = "cuda" if torch.cuda.is_available() else "cpu"` puis `model.to(device)` et `tensor.to(device)`.

## Installer sur une machine sans aucun accès Internet

Voir [offline_usage.md](offline_usage.md), section « Préparer une installation hors ligne ».

## Après l'installation

- Lancer l'interface : `python scripts/run_learning_environment.py --open`
- Lancer JupyterLab : `python -m jupyter lab` (ou `python scripts/run_learning_environment.py --jupyter`)
- Générer les notebooks : `python scripts/build_notebooks.py`
- Exporter les PDF : `python scripts/export_lessons_to_pdf.py --all-in-one`
