import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from student_analysis.analysis import clean_students, run  # noqa: E402

from lab.data import load_csv  # noqa: E402


def test_cleaning_report():
    clean, quality = clean_students(load_csv("students_raw"))
    assert quality["rows_raw"] == 308 and quality["duplicates_removed"] == 8 and quality["rows_clean"] == 300
    assert quality["math_absent"] == 6
    assert clean["student_id"].is_unique
    assert clean["final_score"].isna().sum() == 6  # absent students only


def test_full_run(tmp_path):
    result = run(tmp_path)
    assert result["report"].exists()
    text = result["report"].read_text(encoding="utf-8")
    assert "Qualité des données" in text and "données synthétiques" in text
    assert len(result["figures"]) == 4 and all(p.exists() for p in result["figures"])
    assert result["tables"]["correlations"].index[0] == "previous_score"
    assert result["test"]["p_value"] > 0.05  # groups have no effect in the generator
    assert len(result["orphans"]) == 10
