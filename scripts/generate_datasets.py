"""Generate every dataset used by the lessons, exercises and projects.

All data is synthetic and created locally with fixed random seeds, so running
this script twice produces identical files. No network access is needed.

Usage (from the repository root):
    python scripts/generate_datasets.py            # generate everything
    python scripts/generate_datasets.py --list     # show the datasets
    python scripts/generate_datasets.py --output-dir /tmp/data  # other folder
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab.paths import DATASETS_DIR  # noqa: E402

SEED = 42


def _rng(offset: int) -> np.random.Generator:
    # One independent, reproducible generator per dataset.
    return np.random.default_rng(SEED + offset)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _write_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------
def make_students(n: int = 300) -> pd.DataFrame:
    rng = _rng(1)
    ability = rng.normal(0, 1, n)
    study_hours = np.clip(rng.gamma(4.0, 2.5, n), 0, 30)
    attendance = np.clip(rng.beta(8, 1.5, n), 0.4, 1.0)
    sleep_hours = np.clip(rng.normal(7.2, 0.9, n), 4.5, 10)

    def score(weight_ability: float, noise: float) -> np.ndarray:
        raw = (
            9.5
            + weight_ability * ability
            + 0.22 * (study_hours - 10)
            + 6.0 * (attendance - 0.82)
            + 0.4 * (sleep_hours - 7.2)
            + rng.normal(0, noise, n)
        )
        return np.round(np.clip(raw, 0, 20), 1)

    math = score(2.6, 1.6)
    physics = score(2.3, 1.8)
    french = score(1.6, 2.0)
    previous = np.round(np.clip(10 + 2.2 * ability + rng.normal(0, 2.0, n), 0, 20), 1)
    final = np.round(0.4 * math + 0.3 * physics + 0.3 * french, 2)
    return pd.DataFrame(
        {
            "student_id": [f"S{i:04d}" for i in range(1, n + 1)],
            "group": rng.choice(["A", "B", "C"], size=n, p=[0.35, 0.35, 0.30]),
            "study_hours_week": np.round(study_hours, 1),
            "attendance_rate": np.round(attendance, 3),
            "sleep_hours": np.round(sleep_hours, 1),
            "previous_score": previous,
            "math": math,
            "physics": physics,
            "french": french,
            "final_score": final,
            "passed": (final >= 10).astype(int),
        }
    )


def make_students_raw(students: pd.DataFrame) -> pd.DataFrame:
    """Copy of the students table with realistic data-quality problems."""
    rng = _rng(2)
    raw = students.drop(columns=["final_score", "passed"]).copy()
    raw = raw.astype({"study_hours_week": object, "attendance_rate": object, "math": object, "group": object})
    n = len(raw)

    # Missing values.
    for column, rate in [("study_hours_week", 0.06), ("sleep_hours", 0.05), ("physics", 0.03)]:
        idx = rng.choice(n, size=int(rate * n), replace=False)
        raw.loc[idx, column] = np.nan
    # Inconsistent text formatting in the group column.
    idx = rng.choice(n, size=25, replace=False)
    raw.loc[idx, "group"] = [f" {g.lower()} " if i % 2 else g.lower() for i, g in enumerate(raw.loc[idx, "group"])]
    # Attendance written as percentage strings for some rows.
    idx = rng.choice(n, size=30, replace=False)
    raw.loc[idx, "attendance_rate"] = [f"{float(v) * 100:.1f}%" for v in raw.loc[idx, "attendance_rate"]]
    # Non-numeric marker for absent students.
    idx = rng.choice(n, size=6, replace=False)
    raw.loc[idx, "math"] = "absent"
    # Impossible values (data-entry errors).
    raw.loc[rng.choice(n, size=3, replace=False), "study_hours_week"] = [120.0, 95.0, -3.0]
    # Exact duplicated rows.
    duplicates = raw.iloc[rng.choice(n, size=8, replace=False)]
    raw = pd.concat([raw, duplicates], ignore_index=True)
    return raw.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def make_student_activities(students: pd.DataFrame) -> pd.DataFrame:
    rng = _rng(3)
    ids = rng.choice(students["student_id"].to_numpy(), size=250, replace=False)
    extra = [f"S{i:04d}" for i in range(901, 911)]  # unknown students (merge lesson)
    all_ids = np.concatenate([np.sort(ids), extra])
    clubs = rng.choice(["sport", "musique", "theatre", "robotique", "aucun"], size=len(all_ids), p=[0.3, 0.2, 0.1, 0.15, 0.25])
    hours = np.where(clubs == "aucun", 0, rng.integers(1, 40, size=len(all_ids)))
    return pd.DataFrame({"student_id": all_ids, "club": clubs, "volunteer_hours_year": hours})


# ---------------------------------------------------------------------------
# Sales
# ---------------------------------------------------------------------------
PRODUCTS = {
    "Clavier": ("Peripheriques", 39.90),
    "Souris": ("Peripheriques", 24.90),
    "Ecran": ("Affichage", 189.00),
    "Casque": ("Audio", 59.90),
    "Webcam": ("Video", 69.00),
}
REGIONS = ["Nord", "Sud", "Est", "Ouest"]


def make_sales(n: int = 3000) -> pd.DataFrame:
    rng = _rng(4)
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    # Seasonality: more orders in November and December.
    weights = np.where(days.month >= 11, 2.0, 1.0) * np.where(days.day_of_week >= 5, 0.7, 1.0)
    dates = rng.choice(days, size=n, p=weights / weights.sum())
    products = rng.choice(list(PRODUCTS), size=n, p=[0.25, 0.3, 0.12, 0.2, 0.13])
    regions = rng.choice(REGIONS, size=n, p=[0.3, 0.25, 0.2, 0.25])
    quantity = rng.integers(1, 6, size=n)
    discount = rng.choice([0.0, 0.0, 0.0, 0.1, 0.2], size=n)
    unit_price = np.array([PRODUCTS[p][1] for p in products]) * (1 - discount)
    df = pd.DataFrame(
        {
            "order_id": [f"O{i:05d}" for i in range(1, n + 1)],
            "date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
            "region": regions,
            "product": products,
            "category": [PRODUCTS[p][0] for p in products],
            "quantity": quantity,
            "unit_price": np.round(unit_price, 2),
        }
    )
    return df.sort_values(["date", "order_id"]).reset_index(drop=True)


def make_sales_raw(sales: pd.DataFrame, n: int = 400) -> pd.DataFrame:
    """A messy export of 400 sales lines, used in the cleaning exercises."""
    rng = _rng(5)
    raw = sales.sample(n=n, random_state=SEED).reset_index(drop=True)
    raw = raw.astype({"unit_price": object, "quantity": object, "product": object, "date": object})
    idx = rng.choice(n, size=40, replace=False)
    raw.loc[idx, "product"] = [f"  {p.upper()} " if i % 2 else p.lower() for i, p in enumerate(raw.loc[idx, "product"])]
    idx = rng.choice(n, size=50, replace=False)
    raw.loc[idx, "unit_price"] = [f"{float(p):.2f}".replace(".", ",") + " EUR" for p in raw.loc[idx, "unit_price"]]
    raw.loc[rng.choice(n, size=12, replace=False), "quantity"] = np.nan
    raw.loc[rng.choice(n, size=5, replace=False), "quantity"] = -1
    raw.loc[rng.choice(n, size=4, replace=False), "date"] = np.nan
    raw = pd.concat([raw, raw.iloc[rng.choice(n, size=10, replace=False)]], ignore_index=True)
    return raw.sample(frac=1.0, random_state=SEED + 1).reset_index(drop=True)


def make_orders_json(n_customers: int = 40) -> list[dict]:
    rng = _rng(6)
    cities = ["Lyon", "Lille", "Nantes", "Marseille", "Bordeaux", "Paris", "Rennes"]
    customers = []
    order_counter = 1
    for i in range(1, n_customers + 1):
        orders = []
        for _ in range(int(rng.integers(0, 5))):
            items = []
            for product in rng.choice(list(PRODUCTS), size=int(rng.integers(1, 4)), replace=False):
                items.append(
                    {"product": str(product), "quantity": int(rng.integers(1, 4)), "unit_price": PRODUCTS[str(product)][1]}
                )
            day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 366)))
            orders.append({"order_id": f"W{order_counter:04d}", "date": day.strftime("%Y-%m-%d"), "items": items})
            order_counter += 1
        customers.append(
            {
                "customer_id": f"C{i:03d}",
                "city": str(rng.choice(cities)),
                "member_since": int(rng.integers(2015, 2025)),
                "newsletter": bool(rng.random() < 0.4),
                "orders": orders,
            }
        )
    return customers


# ---------------------------------------------------------------------------
# House prices
# ---------------------------------------------------------------------------
def make_house_prices(n: int = 800) -> pd.DataFrame:
    rng = _rng(7)
    neighborhoods = np.array(["Centre", "Residentiel", "Peripherie", "Rural"])
    hood = rng.choice(neighborhoods, size=n, p=[0.2, 0.35, 0.3, 0.15])
    base_distance = {"Centre": 1.5, "Residentiel": 5.0, "Peripherie": 12.0, "Rural": 25.0}
    distance = np.array([max(0.3, rng.normal(base_distance[h], base_distance[h] * 0.35)) for h in hood])
    surface = np.clip(rng.lognormal(np.log(85), 0.38, n), 20, 320)
    rooms = np.clip(np.round(surface / 25 + rng.normal(0, 0.7, n)), 1, 10).astype(int)
    age = np.clip(rng.gamma(2.0, 18.0, n), 0, 120).round().astype(int)
    garden = (rng.random(n) < np.where(np.isin(hood, ["Centre"]), 0.1, 0.6)).astype(int)
    energy = rng.choice(list("ABCDEFG"), size=n, p=[0.05, 0.12, 0.25, 0.28, 0.17, 0.08, 0.05])
    price_m2 = {"Centre": 4.6, "Residentiel": 3.4, "Peripherie": 2.5, "Rural": 1.7}  # k EUR per m2
    energy_effect = {k: v for k, v in zip("ABCDEFG", [0.08, 0.05, 0.02, 0.0, -0.04, -0.09, -0.14])}
    price = (
        np.array([price_m2[h] for h in hood]) * surface
        * (1 + np.array([energy_effect[e] for e in energy]))
        * (1 - 0.0025 * age)
        * (1 - 0.08 * np.log1p(distance) / np.log1p(30))
        + 18 * garden
        + 6 * rooms
    )
    price = price * rng.lognormal(0, 0.08, n)
    return pd.DataFrame(
        {
            "surface_m2": np.round(surface, 1),
            "rooms": rooms,
            "age_years": age,
            "distance_center_km": np.round(distance, 2),
            "neighborhood": hood,
            "has_garden": garden,
            "energy_class": energy,
            "price_keur": np.round(price, 1),
        }
    )


# ---------------------------------------------------------------------------
# Customers (segmentation)
# ---------------------------------------------------------------------------
SEGMENTS = {
    # name: (share, age, income kEUR, spending score, visits/month, online ratio, basket EUR)
    "jeunes_connectes": (0.30, (24, 4), (24, 6), (68, 10), (6, 2), (0.85, 0.08), (35, 10)),
    "familles_econes": (0.30, (41, 6), (38, 8), (38, 10), (4, 1.5), (0.35, 0.12), (85, 20)),
    "cadres_premium": (0.20, (45, 7), (85, 15), (80, 8), (3, 1), (0.55, 0.15), (180, 40)),
    "seniors_fideles": (0.20, (67, 6), (30, 7), (45, 10), (8, 2), (0.10, 0.06), (50, 12)),
}


def make_customers(n: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = _rng(8)
    names = list(SEGMENTS)
    shares = np.array([SEGMENTS[s][0] for s in names])
    segment = rng.choice(names, size=n, p=shares / shares.sum())

    def draw(position: int, low: float, high: float, decimals: int) -> np.ndarray:
        values = np.array([rng.normal(*SEGMENTS[s][position]) for s in segment])
        return np.round(np.clip(values, low, high), decimals)

    ids = [f"K{i:04d}" for i in range(1, n + 1)]
    customers = pd.DataFrame(
        {
            "customer_id": ids,
            "age": draw(1, 18, 90, 0).astype(int),
            "annual_income_keur": draw(2, 8, 200, 1),
            "spending_score": draw(3, 1, 100, 0).astype(int),
            "visits_per_month": draw(4, 0, 30, 0).astype(int),
            "online_ratio": draw(5, 0, 1, 2),
            "avg_basket_eur": draw(6, 5, 500, 1),
        }
    )
    truth = pd.DataFrame({"customer_id": ids, "true_segment": segment})
    return customers, truth


# ---------------------------------------------------------------------------
# Text datasets (French)
# ---------------------------------------------------------------------------
def make_sms_spam(n: int = 1200, spam_share: float = 0.25) -> pd.DataFrame:
    rng = _rng(9)
    names = ["Lea", "Hugo", "Ines", "Tom", "Sarah", "Nina", "Paul", "Maman", "Karim", "Julie"]
    places = ["café", "cinéma", "stade", "bureau", "parc", "restaurant", "marché"]
    times = ["18h", "19h30", "midi", "20h", "8h", "17h15"]
    ham_templates = [
        "Salut {name}, on se retrouve à {time} au {place} ?",
        "Je suis en retard de 10 minutes, désolé !",
        "Tu peux acheter du pain en rentrant stp",
        "Merci pour hier soir, c'était super",
        "{name} a dit que la réunion est décalée à {time}",
        "Bien arrivé au {place}, je t'appelle ce soir",
        "N'oublie pas le rendez-vous chez le médecin demain à {time}",
        "Tu as le code du wifi ? Je l'ai encore perdu",
        "Le colis est arrivé, je le pose sur ton bureau",
        "On mange ensemble à {time} ? Le {place} à côté du travail",
        "Bon anniversaire {name} ! Grosses bises",
        "Je t'envoie le document ce soir, c'est presque fini",
        "Le match commence à {time}, viens vers le {place}",
        "Tu as gagné au tirage du club, bravo {name} ! On fête ça au {place}",
        "C'est gratuit le musée le premier dimanche, on y va ?",
        "Peux-tu rappeler {name} quand tu as une minute ?",
    ]
    spam_templates = [
        "FÉLICITATIONS ! Vous avez gagné {amount} EUR. Appelez le {phone} pour recevoir votre prix",
        "URGENT : votre compte sera bloqué. Confirmez vos informations sur {url}",
        "Offre exclusive : -{pct}% sur tout le site, cliquez ici {url}",
        "Vous êtes sélectionné pour un iPhone GRATUIT ! Répondez OUI au {short}",
        "Credit rapide sans justificatif jusqu'à {amount} EUR, appelez vite le {phone}",
        "Votre colis est en attente, payez {fee} EUR de frais sur {url}",
        "Gagnez de l'argent depuis chez vous, {amount} EUR par semaine ! Infos : {url}",
        "Dernier rappel : réclamez votre bon d'achat de {amount} EUR avant minuit {url}",
        "Rencontres célibataires près de chez vous, inscription gratuite {url}",
        "Votre remboursement de {amount} EUR est disponible, cliquez {url}",
    ]
    rows = []
    n_spam = int(n * spam_share)
    labels = np.array(["spam"] * n_spam + ["ham"] * (n - n_spam))
    rng.shuffle(labels)
    for label in labels:
        template = str(rng.choice(spam_templates if label == "spam" else ham_templates))
        text = template.format(
            name=rng.choice(names),
            place=rng.choice(places),
            time=rng.choice(times),
            amount=int(rng.choice([100, 250, 500, 1000, 5000])),
            phone=f"08 99 {rng.integers(10, 99)} {rng.integers(10, 99)} {rng.integers(10, 99)}",
            url=f"http://promo-{rng.integers(100, 999)}.example",
            pct=int(rng.choice([50, 70, 80, 90])),
            short=int(rng.integers(30000, 39999)),
            fee=float(rng.choice([1.99, 2.49, 2.99])),
        )
        rows.append((text, label))
    df = pd.DataFrame(rows, columns=["text", "label"])
    df.insert(0, "message_id", [f"M{i:04d}" for i in range(1, n + 1)])
    return df


def make_reviews(n: int = 800) -> pd.DataFrame:
    rng = _rng(10)
    products = ["casque", "clavier", "livre", "téléphone", "sac à dos", "grille-pain", "micro-ondes", "chargeur"]
    positive = ["excellent", "très pratique", "solide", "agréable", "parfait", "rapide", "de bonne qualité", "efficace"]
    negative = ["décevant", "fragile", "bruyant", "lent", "inutilisable", "de mauvaise qualité", "trop cher", "cassé"]
    pos_templates = [
        "Ce {product} est {adj}, je recommande.",
        "Vraiment {adj} et {adj2}, très content de mon achat.",
        "Livraison rapide et {product} {adj}.",
        "Pas mal du tout, {adj} pour le prix.",
        "Je l'utilise tous les jours, {adj} et {adj2}.",
        "Pas du tout {neg}, au contraire il est {adj}.",
    ]
    neg_templates = [
        "Ce {product} est {neg}, je ne recommande pas.",
        "Très {neg} et {neg2}, je vais le renvoyer.",
        "Produit {neg}, service client injoignable.",
        "Pas {adj} du tout, plutôt {neg}.",
        "Au bout d'une semaine il était {neg}. Dommage.",
        "{product} {neg}, remboursement demandé.",
    ]
    rows = []
    for _ in range(n):
        is_positive = rng.random() < 0.55
        template = str(rng.choice(pos_templates if is_positive else neg_templates))
        adj, adj2 = rng.choice(positive, size=2, replace=False)
        neg, neg2 = rng.choice(negative, size=2, replace=False)
        text = template.format(product=rng.choice(products), adj=adj, adj2=adj2, neg=neg, neg2=neg2)
        text = text[0].upper() + text[1:]
        rating = int(rng.choice([4, 5], p=[0.4, 0.6])) if is_positive else int(rng.choice([1, 2], p=[0.5, 0.5]))
        rows.append((text, "positif" if is_positive else "négatif", rating))
    df = pd.DataFrame(rows, columns=["text", "sentiment", "rating"])
    df.insert(0, "review_id", [f"R{i:04d}" for i in range(1, n + 1)])
    return df


KNOWLEDGE_BASE = [
    ("Régression linéaire", "La régression linéaire prédit une valeur numérique continue comme une somme pondérée des variables. Les poids sont choisis pour minimiser l'erreur quadratique moyenne (MSE)."),
    ("Régression logistique", "La régression logistique est un modèle de classification. Elle applique la fonction sigmoïde à une combinaison linéaire des variables pour obtenir une probabilité entre 0 et 1."),
    ("Surapprentissage", "Le surapprentissage (overfitting) se produit quand un modèle mémorise le jeu d'entraînement : l'erreur d'entraînement est faible mais l'erreur de validation est élevée. La régularisation, plus de données ou un modèle plus simple aident à le réduire."),
    ("Validation croisée", "La validation croisée (cross-validation) découpe les données en k plis. Le modèle est entraîné k fois, à chaque fois évalué sur un pli différent, ce qui donne une estimation plus stable de la performance."),
    ("Fuite de données", "La fuite de données (data leakage) arrive quand une information du jeu de test influence l'entraînement, par exemple en normalisant avant de séparer les données. Les scores obtenus sont alors trop optimistes."),
    ("Descente de gradient", "La descente de gradient met à jour les paramètres dans la direction opposée au gradient de la fonction de perte. Le taux d'apprentissage (learning rate) contrôle la taille du pas."),
    ("K-means", "K-means est un algorithme de clustering non supervisé. Il alterne l'affectation de chaque point au centre le plus proche et le recalcul des centres comme moyenne des points affectés."),
    ("ACP", "L'analyse en composantes principales (PCA) projette les données sur les directions de plus grande variance. Elle sert à réduire la dimension et à visualiser des données."),
    ("TF-IDF", "TF-IDF pondère chaque mot par sa fréquence dans le document et par la rareté du mot dans le corpus. Les mots très fréquents partout, comme les articles, reçoivent un poids faible."),
    ("Réseau convolutif", "Un réseau de neurones convolutif (CNN) applique des filtres qui glissent sur l'image pour détecter des motifs locaux comme des bords. Le pooling réduit la taille des cartes de caractéristiques."),
    ("Dropout", "Le dropout désactive aléatoirement des neurones pendant l'entraînement. Cela empêche le réseau de dépendre de quelques neurones et réduit le surapprentissage."),
    ("Attention", "Le mécanisme d'attention calcule, pour chaque élément d'une séquence, une moyenne pondérée des autres éléments. Les poids viennent du produit scalaire entre requêtes (queries) et clés (keys), passé dans un softmax."),
    ("Précision et rappel", "La précision mesure la part de vrais positifs parmi les prédictions positives. Le rappel mesure la part de positifs réels retrouvés. Le F1-score est leur moyenne harmonique."),
    ("IA responsable", "Une IA responsable documente ses données, mesure ses performances par sous-groupe, protège la vie privée et explique ses limites. Un modèle entraîné sur des données synthétiques ne prouve rien sur le monde réel."),
]


def make_knowledge_base() -> list[dict]:
    return [{"doc_id": f"D{i:02d}", "title": t, "text": x} for i, (t, x) in enumerate(KNOWLEDGE_BASE, start=1)]


# ---------------------------------------------------------------------------
# Generic scikit-learn datasets
# ---------------------------------------------------------------------------
def make_tabular() -> dict[str, pd.DataFrame]:
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=600, n_features=6, n_informative=4, n_redundant=1, weights=[0.6, 0.4],
        class_sep=1.0, flip_y=0.03, random_state=SEED,
    )
    binary = pd.DataFrame(np.round(X, 4), columns=[f"x{i}" for i in range(1, 7)])
    binary["target"] = y

    X, y = make_classification(
        n_samples=900, n_features=5, n_informative=3, n_redundant=0, n_classes=3,
        n_clusters_per_class=1, class_sep=1.2, random_state=SEED + 1,
    )
    multiclass = pd.DataFrame(np.round(X, 4), columns=[f"x{i}" for i in range(1, 6)])
    multiclass["target"] = y

    rng = _rng(11)
    n = 500
    X = rng.uniform(-3, 3, size=(n, 4))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] ** 2 + 2 * np.sin(X[:, 3]) + rng.normal(0, 1.0, n)
    regression = pd.DataFrame(np.round(X, 4), columns=["x1", "x2", "x3", "x4"])
    regression["y"] = np.round(y, 4)
    return {"binary": binary, "multiclass": multiclass, "regression": regression}


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
SHAPE_CLASSES = ["circle", "square", "triangle"]


def make_shapes(per_class: int = 100, size: int = 28) -> tuple[np.ndarray, np.ndarray]:
    """Grayscale 28x28 images of circles, squares and triangles."""
    from PIL import Image, ImageDraw

    rng = _rng(12)
    images, labels = [], []
    for label, shape in enumerate(SHAPE_CLASSES):
        for _ in range(per_class):
            image = Image.new("L", (size, size), color=0)
            draw = ImageDraw.Draw(image)
            radius = int(rng.integers(6, 11))
            cx, cy = (int(v) for v in rng.integers(radius + 1, size - radius - 1, size=2))
            fill = int(rng.integers(150, 256))
            if shape == "circle":
                draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=fill)
            elif shape == "square":
                draw.rectangle([cx - radius, cy - radius, cx + radius, cy + radius], fill=fill)
            else:
                angle = rng.uniform(0, 2 * np.pi)
                points = [
                    (cx + radius * np.cos(angle + k * 2 * np.pi / 3), cy + radius * np.sin(angle + k * 2 * np.pi / 3))
                    for k in range(3)
                ]
                draw.polygon(points, fill=fill)
            array = np.asarray(image, dtype=np.float32)
            array = np.clip(array + rng.normal(0, 12, array.shape), 0, 255).astype(np.uint8)
            images.append(array)
            labels.append(label)
    return np.stack(images), np.array(labels, dtype=np.int64)


def save_shapes(images: np.ndarray, labels: np.ndarray, datasets_dir: Path) -> None:
    from PIL import Image

    shapes_dir = datasets_dir / "generated" / "shapes"
    rows = []
    counters = {name: 0 for name in SHAPE_CLASSES}
    for array, label in zip(images, labels):
        name = SHAPE_CLASSES[int(label)]
        relative = f"generated/shapes/{name}/{name}_{counters[name]:03d}.png"
        counters[name] += 1
        path = datasets_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(path)
        rows.append((relative, name, int(label)))
    _write_csv(pd.DataFrame(rows, columns=["path", "label", "label_id"]), shapes_dir / "index.csv")
    np.savez_compressed(
        datasets_dir / "generated" / "shapes.npz", images=images, labels=labels, class_names=np.array(SHAPE_CLASSES)
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def generate_all(datasets_dir: Path = DATASETS_DIR, verbose: bool = True) -> dict[str, Path]:
    """Generate every dataset into ``datasets_dir`` and return the created files."""
    datasets_dir = Path(datasets_dir)
    created: dict[str, Path] = {}

    def save(name: str, relative: str, obj) -> None:
        path = datasets_dir / relative
        if isinstance(obj, pd.DataFrame):
            _write_csv(obj, path)
            detail = f"{len(obj)} rows x {obj.shape[1]} columns"
        else:
            _write_json(obj, path)
            detail = f"{len(obj)} records"
        created[name] = path
        if verbose:
            print(f"  [OK] {relative:<45} {detail}")

    if verbose:
        print(f"Generating datasets in {datasets_dir} (seed={SEED})")
    students = make_students()
    save("students", "generated/students.csv", students)
    save("students_raw", "raw/students_raw.csv", make_students_raw(students))
    save("student_activities", "generated/student_activities.csv", make_student_activities(students))
    sales = make_sales()
    save("sales", "generated/sales.csv", sales)
    save("sales_raw", "raw/sales_raw.csv", make_sales_raw(sales))
    save("orders_json", "raw/orders.json", make_orders_json())
    save("house_prices", "generated/house_prices.csv", make_house_prices())
    customers, truth = make_customers()
    save("customers", "generated/customers.csv", customers)
    save("customers_true_segments", "generated/customers_true_segments.csv", truth)
    save("sms_spam", "generated/sms_spam.csv", make_sms_spam())
    save("reviews", "generated/reviews_fr.csv", make_reviews())
    for name, frame in make_tabular().items():
        filename = {"binary": "binary_classification", "multiclass": "multiclass_classification"}.get(name, name)
        save(name, f"generated/{filename}.csv", frame)
    save("knowledge_base", "generated/knowledge_base.json", make_knowledge_base())

    images, labels = make_shapes()
    save_shapes(images, labels, datasets_dir)
    created["shapes_index"] = datasets_dir / "generated/shapes/index.csv"
    created["shapes_npz"] = datasets_dir / "generated/shapes.npz"
    if verbose:
        print(f"  [OK] generated/shapes/ + shapes.npz              {len(images)} images 28x28")
        (datasets_dir / "processed").mkdir(parents=True, exist_ok=True)
        print("Done. All files were created locally, without network access.")
    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DATASETS_DIR, help="target datasets folder")
    parser.add_argument("--list", action="store_true", help="list the registered datasets and exit")
    args = parser.parse_args(argv)
    if args.list:
        from lab.data import DATASETS

        for info in DATASETS.values():
            status = "present" if info.path.exists() else "MISSING"
            print(f"{info.name:<25} {info.relative_path:<42} {status:<8} {info.description}")
        return 0
    generate_all(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
