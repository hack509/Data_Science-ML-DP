"""Markdown report and model card."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .config import Config


def _markdown_table(frame: pd.DataFrame) -> str:
    frame = frame.reset_index()
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def write_report(output_dir: Path, config: Config, data_log: dict, cv_table: pd.DataFrame, chosen: str, metrics: dict) -> Path:
    subgroups = pd.DataFrame(metrics["subgroups"]).T
    subgroups["n"] = subgroups["n"].astype(int)
    subgroups.index.name = config.subgroup_column
    lines = [
        "# Rapport — prédiction de la réussite (pipeline de bout en bout)",
        "",
        "> Données synthétiques générées localement : ce rapport illustre une méthode, il ne décrit pas de vrais étudiants.",
        "",
        "## Données",
        "",
        f"- Lignes brutes : {data_log['rows_raw']} ; doublons supprimés : {data_log['duplicates_removed']} ; "
        f"valeurs impossibles mises à manquant : {data_log['impossible_values_set_to_missing']}.",
        f"- Lignes sans étiquette (note manquante) exclues : {data_log['rows_without_label']} ; lignes utilisées : {data_log['rows_labelled']} ; "
        f"taux de réussite : {data_log['pass_rate']:.1%}.",
        f"- Contrôles de qualité réussis : {len(data_log['quality_checks_passed'])}.",
        f"- Variables : {', '.join(config.features)}. Les notes de matières, qui définissent la cible, sont exclues (fuite).",
        "",
        "## Sélection du modèle (validation croisée, ROC-AUC)",
        "",
        _markdown_table(cv_table.round(4)),
        "",
        f"Modèle retenu : **{chosen}** (le plus simple à moins d'une erreur standard du meilleur).",
        "",
        "## Évaluation finale (jeu de test, utilisé une seule fois)",
        "",
        f"- n = {metrics['n_test']} ; ROC-AUC = {metrics['roc_auc']} (IC 95 % {metrics['ci95']['roc_auc']}) ; "
        f"accuracy = {metrics['accuracy']} (IC 95 % {metrics['ci95']['accuracy']}).",
        f"- Précision = {metrics['precision']} ; rappel = {metrics['recall']} ; F1 = {metrics['f1']} (seuil {config.decision_threshold}).",
        "",
        f"### Par sous-groupe ({config.subgroup_column})",
        "",
        _markdown_table(subgroups),
        "",
        "Les sous-groupes contiennent peu d'exemples de test : leurs métriques sont très incertaines.",
    ]
    path = output_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_model_card(output_dir: Path, config: Config, chosen: str, metrics: dict, versions: dict) -> Path:
    lines = [
        f"# Fiche modèle — {chosen}",
        "",
        f"- **Date** : {date.today().isoformat()} ; versions : {versions}.",
        "- **Usage prévu** : exercice pédagogique de pipeline de bout en bout. **Ne pas utiliser pour évaluer de vrais étudiants.**",
        "- **Données** : `datasets/raw/students_raw.csv` et `student_activities.csv`, synthétiques (graine fixe).",
        f"- **Cible** : note finale ≥ {config.pass_mark} (calculée à partir des notes, qui ne sont pas des variables du modèle).",
        f"- **Performance (test)** : ROC-AUC {metrics['roc_auc']} (IC 95 % {metrics['ci95']['roc_auc']}), accuracy {metrics['accuracy']}.",
        f"- **Sous-groupes** : {metrics['subgroups']}.",
        "- **Limites** : moins de 300 exemples ; corrélations synthétiques ; aucune validation sur des données réelles ; les petits groupes ont des métriques instables.",
        "- **Risques** : utilisée pour orienter des étudiants, une telle prédiction pourrait devenir auto-réalisatrice ; toute décision doit rester humaine.",
    ]
    path = output_dir / "model_card.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
