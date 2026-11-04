import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from segmentation.clustering import run  # noqa: E402


def test_segmentation(tmp_path):
    result = run(tmp_path)
    assert result["k"] == 4
    assert result["ari"] > 0.95
    assert result["stability"] > 0.5
    profiles = result["profiles"]
    assert profiles["n"].sum() == 500
    assert set(profiles["segment"]) == {"seniors fidèles", "clients premium", "jeunes connectés", "familles économes"}
    assert result["report"].exists() and all(p.exists() for p in result["figures"])
    assert (tmp_path / "customers_with_segments.csv").exists()
