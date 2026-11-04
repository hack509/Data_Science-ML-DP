"""Choose k, cluster customers, describe and name the segments, write a report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from lab.data import load_csv  # noqa: E402
from lab.plotting import CATEGORICAL, apply_style  # noqa: E402

FEATURES = ["age", "annual_income_keur", "spending_score", "visits_per_month", "online_ratio", "avg_basket_eur"]
SEED = 0


def choose_k(X: np.ndarray, k_values=range(2, 9)) -> pd.DataFrame:
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": silhouette_score(X, model.labels_)})
    return pd.DataFrame(rows).set_index("k")


def stability(X: np.ndarray, k: int, n_runs: int = 5) -> float:
    """Mean adjusted Rand index between runs with different seeds (1 = perfectly stable)."""
    labels = [KMeans(n_clusters=k, n_init=1, random_state=seed).fit_predict(X) for seed in range(n_runs)]
    scores = [adjusted_rand_score(labels[i], labels[j]) for i in range(n_runs) for j in range(i + 1, n_runs)]
    return float(np.mean(scores))


def name_segment(profile: pd.Series, overall: pd.Series) -> str:
    """Rule-based business names from how a segment differs from the average customer."""
    if profile["age"] > overall["age"] + 15:
        return "seniors fidèles"
    if profile["annual_income_keur"] > overall["annual_income_keur"] * 1.5:
        return "clients premium"
    if profile["online_ratio"] > overall["online_ratio"] + 0.2 and profile["age"] < overall["age"]:
        return "jeunes connectés"
    if profile["spending_score"] < overall["spending_score"] - 5 and profile["online_ratio"] < overall["online_ratio"]:
        return "familles économes"
    return "clients standard"


def profile_segments(customers: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df = customers.assign(cluster=labels)
    profiles = df.groupby("cluster")[FEATURES].mean()
    overall = df[FEATURES].mean()
    profiles.insert(0, "n", df.groupby("cluster").size())
    profiles.insert(1, "share", (profiles["n"] / len(df)).round(3))
    profiles.insert(0, "segment", [name_segment(profiles.loc[c], overall) for c in profiles.index])
    return profiles.round(2)


def make_figures(X: np.ndarray, labels: np.ndarray, selection: pd.DataFrame, profiles: pd.DataFrame, output_dir: Path) -> list[Path]:
    apply_style()
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].plot(selection.index, selection["inertia"], marker="o", color=CATEGORICAL[0])
    axes[0].set_title("Inertie (méthode du coude)")
    axes[1].plot(selection.index, selection["silhouette"], marker="o", color=CATEGORICAL[0])
    axes[1].set_title("Score de silhouette")
    for ax in axes:
        ax.set_xlabel("nombre de segments k")
    paths.append(output_dir / "01_choose_k.png")
    fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)

    coords = PCA(n_components=2, random_state=SEED).fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for color, cluster in zip(CATEGORICAL, sorted(np.unique(labels))):
        mask = labels == cluster
        ax.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.7, color=color, label=profiles.loc[cluster, "segment"])
    ax.set_xlabel("composante principale 1")
    ax.set_ylabel("composante principale 2")
    ax.set_title("Segments de clientèle (projection ACP)")
    ax.legend()
    paths.append(output_dir / "02_segments_pca.png")
    fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)
    return paths


def write_report(selection, k, stab, profiles, ari, figures, output_dir: Path) -> Path:
    def table(frame):
        frame = frame.reset_index()
        rows = ["| " + " | ".join(str(c) for c in frame.columns) + " |", "|" + "---|" * len(frame.columns)]
        rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in frame.itertuples(index=False)]
        return "\n".join(rows)

    lines = [
        "# Rapport — segmentation de clientèle (données synthétiques)",
        "",
        f"- Nombre de segments retenu : **k = {k}** (silhouette maximale = {selection.loc[k, 'silhouette']:.3f}).",
        f"- Stabilité entre initialisations (ARI moyen) : {stab:.3f}.",
        f"- Contrôle avec les segments du générateur (impossible avec des données réelles) : ARI = {ari:.3f}.",
        "",
        "## Choix de k",
        "",
        table(selection.round(3)),
        "",
        "## Profils des segments (moyennes)",
        "",
        table(profiles),
        "",
        "## Figures",
        "",
        *[f"- `{p.name}`" for p in figures],
        "",
        "## Limites",
        "",
        "- Les noms de segments sont attribués par des règles simples ; ils doivent être validés par les équipes métier.",
        "- k-means suppose des groupes compacts de tailles comparables.",
        "- Données synthétiques : la structure en 4 segments a été construite par le générateur.",
    ]
    path = output_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    customers = load_csv("customers")
    X = StandardScaler().fit_transform(customers[FEATURES])
    selection = choose_k(X)
    k = int(selection["silhouette"].idxmax())
    model = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    profiles = profile_segments(customers, model.labels_)
    truth = load_csv("customers_true_segments").set_index("customer_id").loc[customers["customer_id"], "true_segment"]
    ari = adjusted_rand_score(truth, model.labels_)
    stab = stability(X, k)
    figures = make_figures(X, model.labels_, selection, profiles, output_dir)
    customers.assign(cluster=model.labels_, segment=profiles.loc[model.labels_, "segment"].to_numpy()).to_csv(
        output_dir / "customers_with_segments.csv", index=False
    )
    report = write_report(selection, k, stab, profiles, ari, figures, output_dir)
    return {"k": k, "selection": selection, "profiles": profiles, "ari": ari, "stability": stab, "report": report, "figures": figures}
