"""Check that every lesson follows the common template (docs/lesson_template.md)."""

import pytest

from lab.catalog import list_lessons
from lab.markdown_lite import headings, inline_to_plain, slugify

# Section headings required in every lesson (compared after slugification,
# so accents and case do not matter). A heading only has to START with the
# required words, e.g. "Exemples de code Python" or "Quiz (réponses : ...)".
REQUIRED_SECTIONS = [
    "prerequis",
    "objectifs",
    "concepts cles",
    "explication detaillee",
    "applications",
    "exemples de code",
    "erreurs frequentes",
    "exercices guides",
    "defis",
    "quiz",
    "resume",
    "pour aller plus loin",
]


@pytest.mark.parametrize("lesson", list_lessons(), ids=lambda lesson: lesson.id)
def test_lesson_has_required_sections(lesson):
    text = lesson.path.read_text(encoding="utf-8")
    level2 = [slugify(inline_to_plain(h.text)) for h in headings(text) if h.level == 2]
    missing = [s for s in REQUIRED_SECTIONS if not any(h.startswith(slugify(s)) for h in level2)]
    assert not missing, f"{lesson.relative_path} is missing sections: {missing}"


@pytest.mark.parametrize("lesson", list_lessons(), ids=lambda lesson: lesson.id)
def test_lesson_is_substantive(lesson):
    text = lesson.path.read_text(encoding="utf-8")
    words = len(text.split())
    assert words >= 900, f"{lesson.relative_path} looks too short ({words} words)"
    assert "```python" in text, f"{lesson.relative_path} has no Python example"
