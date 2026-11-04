"""Cleaning, analysis, figures and report for the student dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from lab.data import load_csv  # noqa: E402
from lab.plotting import CATEGORICAL, apply_style, diverging_cmap  # noqa: E402

SUBJECTS = ["math", "physics", "french"]
WEIGHTS = {"math": 0.4, "physics": 0.3, "french": 0.3}


def _parse_rate(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    return float(text.removesuffix("%")) / 100 if text.endswith("%") else float(text)


def clean_students(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the raw export; return the clean table and a quality report."""
    report = {"rows_raw": len(raw)}
    df = raw.drop_duplicates().copy()
    report["duplicates_removed"] = report["rows_raw"] - len(df)
    df["group"] = df["group"].str.strip().str.upper()
    df["attendance_rate"] = df["attendance_rate"].map(_parse_rate)
    df["math_absent"] = df["math"].eq("absent")
    df["math"] = pd.to_numeric(df["math"], errors="coerce")
    impossible = df["study_hours_week"].notna() & ~df["study_hours_week"].between(0, 80)
    report["impossible_values"] = int(impossible.sum())
    df.loc[impossible, "study_hours_week"] = np.nan
    report["imputed"] = {}
    for column in ["study_hours_week", "sleep_hours", "physics"]:
        report["imputed"][column] = int(df[column].isna().sum())
    df["study_hours_week"] = df["study_hours_week"].fillna(df.groupby("group")["study_hours_week"].transform("median"))
    df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].median())
    df["physics"] = df["physics"].fillna(df["physics"].median())
    # Absent students have no math score: their final score cannot be computed.
    df["final_score"] = sum(df[s] * w for s, w in WEIGHTS.items()).round(2)
    df["passed"] = (df["final_score"] >= 10).astype("Int64").where(df["final_score"].notna())
    report["math_absent"] = int(df["math_absent"].sum())
    report["rows_clean"] = len(df)
    return df.sort_values("student_id").reset_index(drop=True), report


def add_activities(students: pd.DataFrame, activities: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    orphans = sorted(set(activities["student_id"]) - set(students["student_id"]))
    merged = students.merge(activities, on="student_id", how="left", validate="one_to_one")
    merged["club"] = merged["club"].fillna("inconnu")
    return merged, orphans


def summaries(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    scored = df[df["final_score"].notna()].copy()
    scored["study_level"] = pd.qcut(scored["study_hours_week"], 4, labels=["Q1 (faible)", "Q2", "Q3", "Q4 (élevé)"])
    agg = dict(n=("student_id", "size"), mean_final=("final_score", "mean"), pass_rate=("passed", "mean"))
    return {
        "by_group": scored.groupby("group").agg(**agg).round(3),
        "by_study_level": scored.groupby("study_level", observed=True).agg(**agg).round(3),
        "by_club": scored.groupby("club").agg(**agg).sort_values("pass_rate", ascending=False).round(3),
        "correlations": scored[["study_hours_week", "attendance_rate", "sleep_hours", "previous_score", "final_score"]]
        .corr()["final_score"].drop("final_score").sort_values(ascending=False).round(3).to_frame("correlation"),
    }


def group_test(df: pd.DataFrame, a: str = "C", b: str = "A") -> dict:
    scored = df[df["final_score"].notna()]
    x = scored.loc[scored["group"] == a, "final_score"]
    y = scored.loc[scored["group"] == b, "final_score"]
    result = stats.ttest_ind(x, y, equal_var=False)
    return {"groups": f"{a} vs {b}", "difference": round(float(x.mean() - y.mean()), 3),
            "t": round(float(result.statistic), 3), "p_value": round(float(result.pvalue), 4)}


def make_figures(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    apply_style()
    scored = df[df["final_score"].notna()]
    paths = []

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(scored["final_score"], bins=20, color=CATEGORICAL[0], edgecolor="white")
    ax.axvline(10, color="#2b2b29", linestyle="--", linewidth=1)
    ax.set_title(f"{(scored['final_score'] < 10).mean():.0%} des étudiants sont sous la moyenne")
    ax.set_xlabel("Note finale /20")
    ax.set_ylabel("Étudiants")
    paths.append(output_dir / "01_final_score_distribution.png")

    def save(figure, name):
        figure.savefig(output_dir / name, bbox_inches="tight")
        plt.close(figure)
        return output_dir / name

    save(fig, paths[-1].name)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for color, (value, label) in zip(CATEGORICAL, [(1, "admis"), (0, "non admis")]):
        subset = scored[scored["passed"] == value]
        ax.scatter(subset["study_hours_week"], subset["final_score"], s=12, alpha=0.7, color=color, label=label)
    ax.set_xlabel("Heures d'étude par semaine")
    ax.set_ylabel("Note finale")
    ax.set_title("Plus d'heures d'étude, meilleures notes en moyenne")
    ax.legend()
    paths.append(save(fig, "02_study_hours_vs_score.png"))

    import seaborn as sns

    fig, ax = plt.subplots(figsize=(5, 3.4))
    sns.boxplot(data=scored, x="group", y="final_score", order=["A", "B", "C"], color="#cde2fb", ax=ax)
    ax.set_title("Distribution des notes par groupe")
    paths.append(save(fig, "03_score_by_group.png"))

    cols = ["study_hours_week", "attendance_rate", "sleep_hours", "previous_score"] + SUBJECTS + ["final_score"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(scored[cols].corr(), cmap=diverging_cmap(), vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
    ax.set_title("Corrélations")
    ax.grid(False)
    paths.append(save(fig, "04_correlations.png"))
    return paths


def _table(frame: pd.DataFrame) -> str:
    frame = frame.reset_index()
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    separator = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, separator, *rows])


def write_report(df, quality, orphans, tables, test, figures, output_dir: Path) -> Path:
    scored = df[df["final_score"].notna()]
    lines = [
        "# Rapport — performance des étudiants (données synthétiques)",
        "",
        "> Les données sont générées par `scripts/generate_datasets.py` : les constats illustrent une méthode d'analyse et ne décrivent pas de vrais étudiants.",
        "",
        "## Qualité des données",
        "",
        f"- Lignes brutes : {quality['rows_raw']} ; doublons supprimés : {quality['duplicates_removed']} ; lignes finales : {quality['rows_clean']}.",
        f"- Valeurs impossibles d'heures d'étude remplacées : {quality['impossible_values']}.",
        f"- Valeurs imputées : {quality['imputed']}.",
        f"- Étudiants absents en mathématiques (note finale non calculable) : {quality['math_absent']}.",
        f"- Identifiants d'activités inconnus : {len(orphans)} ({', '.join(orphans[:3])}...).",
        "",
        "## Constats principaux",
        "",
        f"- Taux de réussite : {scored['passed'].mean():.1%} sur {len(scored)} étudiants notés.",
        f"- Variable la plus corrélée à la note finale : {tables['correlations'].index[0]} (r = {tables['correlations'].iloc[0, 0]}).",
        f"- Écart entre groupes {test['groups']} : {test['difference']} point (p = {test['p_value']}). "
        + ("Différence non significative au seuil de 5 %." if test["p_value"] >= 0.05 else "Différence significative au seuil de 5 %."),
        "",
        "## Par groupe",
        "",
        _table(tables["by_group"]),
        "",
        "## Par niveau d'heures d'étude",
        "",
        _table(tables["by_study_level"]),
        "",
        "## Par activité",
        "",
        _table(tables["by_club"]),
        "",
        "## Figures",
        "",
        *[f"- `{p.name}`" for p in figures],
        "",
        "## Limites",
        "",
        "- Analyse observationnelle : les associations ne prouvent aucun effet causal.",
        "- Les petits groupes (activités peu suivies) ont des taux très incertains.",
    ]
    path = output_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean, quality = clean_students(load_csv("students_raw"))
    merged, orphans = add_activities(clean, load_csv("student_activities"))
    merged.to_csv(output_dir / "students_clean.csv", index=False)
    tables = summaries(merged)
    test = group_test(merged)
    figures = make_figures(merged, output_dir)
    report = write_report(merged, quality, orphans, tables, test, figures, output_dir)
    return {"quality": quality, "orphans": orphans, "tables": tables, "test": test, "figures": figures, "report": report}
