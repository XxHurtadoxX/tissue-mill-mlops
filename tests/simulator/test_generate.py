"""Tests del generador: esquema, determinismo, suciedad y firmas de falla."""
import statistics
from datetime import date, timedelta

from src.simulator import generate, plant
from src.simulator.dirty import BAD_VALUE


SEED = 42


def _plant_and_scheds(day):
    p = plant.build_plant()
    scheds = {eq.code: plant.failure_schedule(eq, SEED, day) for eq in p}
    return p, scheds


def test_historian_schema():
    p, scheds = _plant_and_scheds(date(2026, 6, 1))
    rows = generate.historian_rows(date(2026, 6, 1), p, scheds, SEED,
                                   sample_min=60, dirty=True)
    assert rows
    for r in rows:
        assert len(r) == 4
        assert r[3] in ("GOOD", "BAD")


def test_generation_is_deterministic():
    p, scheds = _plant_and_scheds(date(2026, 6, 1))
    a = generate.historian_rows(date(2026, 6, 1), p, scheds, SEED, 60, True)
    b = generate.historian_rows(date(2026, 6, 1), p, scheds, SEED, 60, True)
    assert a == b


def test_clean_mode_has_no_bad_values():
    p, scheds = _plant_and_scheds(date(2026, 6, 1))
    rows = generate.historian_rows(date(2026, 6, 1), p, scheds, SEED, 60,
                                   dirty=False)
    assert all(r[3] == "GOOD" for r in rows)
    assert all(r[2] != int(BAD_VALUE) for r in rows)


def test_dirty_mode_injects_bad_values():
    # Con muchos días acumulamos suficientes muestras para ver BAD/-9999.
    p, scheds = _plant_and_scheds(date(2026, 6, 30))
    bad = 0
    for i in range(30):
        d = date(2026, 6, 1) + timedelta(days=i)
        rows = generate.historian_rows(d, p, scheds, SEED, 60, dirty=True)
        bad += sum(1 for r in rows if r[3] == "BAD")
    assert bad > 0


def test_failure_drives_vibration_up():
    """Al final de una degradación no silenciosa la vibración supera su base."""
    eq = next(e for e in plant.build_plant() if e.code == "BV2")
    sched = plant.failure_schedule(eq, SEED, date(2027, 1, 1))
    f = next(x for x in sched if not x.silent)
    vib = next(s for s in eq.signals if s.kind == "vibration")
    late = f.fail - timedelta(days=2)       # casi al final de la degradación
    healthy = f.onset - timedelta(days=40)  # bastante antes del onset

    def mean_vib(day):
        vals = []
        for slot in range(24):
            import datetime as dt
            ts = dt.datetime(day.year, day.month, day.day, slot)
            vals.append(generate.signal_value(eq, vib, ts,
                        plant.active_failure(sched, day), SEED))
        return statistics.mean(vals)

    assert mean_vib(late) > mean_vib(healthy) * 1.2


def test_silent_failure_leaves_no_trace():
    """Una falla silenciosa no altera la señal: es el techo del modelo."""
    eq = next(e for e in plant.build_plant() if e.code == "BV2")
    vib = next(s for s in eq.signals if s.kind == "vibration")
    silent = plant.Failure("BV2", date(2026, 1, 1), date(2026, 2, 1), "test", True)
    ruidosa = plant.Failure("BV2", date(2026, 1, 1), date(2026, 2, 1), "test", False)
    import datetime as dt
    ts = dt.datetime(2026, 1, 30, 12)
    assert (generate.signal_value(eq, vib, ts, silent, SEED)
            < generate.signal_value(eq, vib, ts, ruidosa, SEED))


def test_rotor_wear_increases_kwh_per_kg():
    """El desgaste del rotor sube el consumo específico (kWh/kg) del lote."""
    # Busca un día de bajo desgaste y uno de alto en el primer ciclo del rotor.
    low = high = None
    d = plant.PLANT_EPOCH
    while d < date(2025, 1, 1):
        w = generate._rotor_wear("P2", SEED, d)
        if w < 0.1 and low is None:
            low = d
        if w > 0.9 and high is None:
            high = d
        d += timedelta(days=1)
    assert low and high

    def mean_kwh_per_kg(day):
        rows = generate.batch_rows(day, SEED)
        p2 = [r for r in rows if r[1] == "P2"]
        return statistics.mean(r[6] / r[5] for r in p2)

    assert mean_kwh_per_kg(high) > mean_kwh_per_kg(low) * 1.1


def test_work_order_emitted_on_failure_day(tmp_path):
    eq = next(e for e in plant.build_plant() if e.code == "BV2")
    sched = plant.failure_schedule(eq, SEED, date(2027, 1, 1))
    fail_day = sched[0].fail
    p, scheds = _plant_and_scheds(fail_day)
    orders, truth = generate.order_and_truth_rows(fail_day, p, scheds, SEED)
    assert any(row[5] == eq.eq_id for row in orders)
    assert any(row[1] == eq.code for row in truth)


def test_daily_generation_matches_range(tmp_path):
    """Un día generado suelto == el mismo día dentro de un rango (cron estable)."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    day = date(2026, 5, 10)
    generate.generate_day(day, str(out_a), SEED, 60, True)
    generate.generate_range(date(2026, 5, 9), date(2026, 5, 11), str(out_b),
                            SEED, 60, True)
    fa = (out_a / "bronze" / "historian" / "2026-05-10.csv").read_text(
        encoding="utf-8")
    fb = (out_b / "bronze" / "historian" / "2026-05-10.csv").read_text(
        encoding="utf-8")
    assert fa == fb
