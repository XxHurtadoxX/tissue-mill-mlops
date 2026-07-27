"""Tests del modelo de planta: tags, cronograma de fallas y determinismo."""
import re
from datetime import date

from src.simulator import plant


TAG_RE = re.compile(r"^TIS1\.\d{2}(II|TI|VI|PI)\d{4}\.PV$")


def test_roster_no_duplicate_codes():
    codes = [eq.code for eq in plant.build_plant()]
    assert len(codes) == len(set(codes))


def test_tag_grammar_is_isa_compliant():
    for eq in plant.build_plant():
        for sig in eq.signals:
            assert TAG_RE.match(eq.tag(sig)), eq.tag(sig)


def test_pulper2_current_tag_matches_case():
    p2 = next(eq for eq in plant.build_plant() if eq.code == "P2")
    cur = next(s for s in p2.signals if s.kind == "current")
    assert p2.tag(cur) == "TIS1.20II0205.PV"


def test_failure_schedule_is_deterministic():
    eq = next(e for e in plant.build_plant() if e.code == "BV2")
    horizon = date(2026, 1, 1)
    a = plant.failure_schedule(eq, seed=42, horizon=horizon)
    b = plant.failure_schedule(eq, seed=42, horizon=horizon)
    assert a == b
    assert len(a) > 0


def test_failure_schedule_changes_with_seed():
    eq = next(e for e in plant.build_plant() if e.code == "BV2")
    horizon = date(2026, 1, 1)
    a = plant.failure_schedule(eq, seed=1, horizon=horizon)
    b = plant.failure_schedule(eq, seed=2, horizon=horizon)
    assert a != b


def test_progress_bounds_and_window():
    f = plant.Failure("X", date(2026, 1, 1), date(2026, 2, 1), "test")
    assert plant.progress(f, date(2025, 12, 31)) is None
    assert plant.progress(f, date(2026, 1, 1)) == 0.0
    assert plant.progress(f, date(2026, 2, 1)) == 1.0
    mid = plant.progress(f, date(2026, 1, 16))
    assert 0.4 < mid < 0.6
