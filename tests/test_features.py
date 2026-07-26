"""Pruebas de la construcción del dataset, centradas en las guardas anti-fuga.

Una fuga de información (leakage) no rompe el código: produce un modelo con
métricas excelentes que en producción no sirve. Por eso cada guarda se prueba
de forma explícita.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features import build_dataset as bd
from src.features.clean import clean_historian, flag_flatlines
from src.simulator.dirty import BAD_VALUE
from src.simulator.plant import build_plant


# --------------------------------------------------------------------------- #
# Limpieza
# --------------------------------------------------------------------------- #
def test_bad_readings_are_removed():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
        "tag": ["T1", "T1"],
        "value": [10.0, BAD_VALUE],
        "quality": ["GOOD", "BAD"],
    })
    out = clean_historian(df)
    assert len(out) == 1
    assert BAD_VALUE not in out["value"].values


def test_frozen_sensor_is_detected():
    """Una racha de valores idénticos se marca como flatline y se descarta."""
    values = [10.1, 10.2, 7.7, 7.7, 7.7, 7.7, 10.3]
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=len(values), freq="h"),
        "tag": ["T1"] * len(values),
        "value": values,
        "quality": ["GOOD"] * len(values),
    })
    flagged = flag_flatlines(df)
    assert flagged["flatline"].sum() == 4
    assert len(clean_historian(df)) == 3


def test_identical_values_across_different_tags_are_not_flatline():
    """La detección es por tag: dos sensores distintos pueden coincidir."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01 00:00"] * 3),
        "tag": ["T1", "T2", "T3"],
        "value": [5.0, 5.0, 5.0],
        "quality": ["GOOD"] * 3,
    })
    assert not flag_flatlines(df)["flatline"].any()


# --------------------------------------------------------------------------- #
# Etiquetas
# --------------------------------------------------------------------------- #
def _orders(rows):
    return pd.DataFrame(rows, columns=["equipo_code", "fecha_orden", "tipo_orden"])


def test_label_is_positive_within_horizon():
    base = pd.DataFrame({"equipo_code": ["BV1"], "fecha": [date(2026, 1, 1)]})
    orders = _orders([("BV1", date(2026, 1, 10), "PM01")])
    assert bd.build_labels(base, orders, horizon=14).iloc[0] == 1


def test_label_is_negative_beyond_horizon():
    base = pd.DataFrame({"equipo_code": ["BV1"], "fecha": [date(2026, 1, 1)]})
    orders = _orders([("BV1", date(2026, 2, 20), "PM01")])
    assert bd.build_labels(base, orders, horizon=14).iloc[0] == 0


def test_label_ignores_past_orders():
    """Una falla ya ocurrida no puede etiquetar el presente como positivo."""
    base = pd.DataFrame({"equipo_code": ["BV1"], "fecha": [date(2026, 1, 20)]})
    orders = _orders([("BV1", date(2026, 1, 5), "PM01")])
    assert bd.build_labels(base, orders, horizon=14).iloc[0] == 0


def test_label_only_uses_corrective_orders():
    """Las preventivas y predictivas no son averías, así que no etiquetan."""
    base = pd.DataFrame({"equipo_code": ["BV1"], "fecha": [date(2026, 1, 1)]})
    for tipo in ("PM02", "PM03"):
        orders = _orders([("BV1", date(2026, 1, 5), tipo)])
        assert bd.build_labels(base, orders, horizon=14).iloc[0] == 0


def test_label_does_not_cross_equipment():
    base = pd.DataFrame({"equipo_code": ["BV1"], "fecha": [date(2026, 1, 1)]})
    orders = _orders([("FP1", date(2026, 1, 5), "PM01")])
    assert bd.build_labels(base, orders, horizon=14).iloc[0] == 0


# --------------------------------------------------------------------------- #
# Contexto de mantenimiento, sin mirar al futuro
# --------------------------------------------------------------------------- #
def test_days_since_order_ignores_future_orders():
    base = pd.DataFrame({"equipo_code": ["BV1"], "fecha": [date(2026, 1, 10)]})
    orders = _orders([
        ("BV1", date(2026, 1, 4), "PM01"),
        ("BV1", date(2026, 1, 25), "PM01"),   # futura: debe ignorarse
    ])
    assert bd.days_since_last_order(base, orders).iloc[0] == 6


def test_route_uses_last_past_measurement_only():
    base = pd.DataFrame({"equipo_code": ["BV1"], "fecha": [date(2026, 3, 10)]})
    routes = pd.DataFrame({
        "fecha": [date(2026, 2, 3), date(2026, 3, 3), date(2026, 4, 3)],
        "equipo_code": ["BV1"] * 3,
        "zona_iso_ord": [0, 2, 3],
        "rms_ruta": [2.4, 5.1, 8.0],
    })
    out = bd.last_known_route(base, routes)
    assert out["zona_iso_ord"].iloc[0] == 2      # la de marzo, no la de abril
    assert out["dias_desde_ruta"].iloc[0] == 7


# --------------------------------------------------------------------------- #
# Ventanas móviles
# --------------------------------------------------------------------------- #
def test_rolling_window_does_not_use_future_values():
    """Alterar el futuro no puede cambiar la feature de un día anterior."""
    fechas = [date(2026, 1, 1) + timedelta(days=i) for i in range(10)]
    daily = pd.DataFrame({
        "equipo_code": ["BV1"] * 10,
        "kind": ["vibration"] * 10,
        "fecha": fechas,
        "valor_medio": [2.0] * 10,
        "n_lecturas": [24] * 10,
    })
    original = bd.rolling_features(daily)

    alterado = daily.copy()
    alterado.loc[alterado["fecha"] == fechas[9], "valor_medio"] = 99.0
    nuevo = bd.rolling_features(alterado)

    dia7 = fechas[6]
    col = "vibration_d7_mean"
    antes = original.loc[original["fecha"] == dia7, col].iloc[0]
    despues = nuevo.loc[nuevo["fecha"] == dia7, col].iloc[0]
    assert antes == pytest.approx(despues)


def test_slope_detects_rising_trend():
    valores = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    assert bd._slope(valores) == pytest.approx(1.0)
    assert bd._slope(valores[::-1]) == pytest.approx(-1.0)


def test_incomplete_window_yields_nan():
    """Sin siete días de historia no puede haber tendencia calculada."""
    fechas = [date(2026, 1, 1) + timedelta(days=i) for i in range(3)]
    daily = pd.DataFrame({
        "equipo_code": ["BV1"] * 3,
        "kind": ["vibration"] * 3,
        "fecha": fechas,
        "valor_medio": [2.0, 2.1, 2.2],
        "n_lecturas": [24] * 3,
    })
    out = bd.rolling_features(daily)
    assert out["vibration_d7_mean"].isna().all()


# --------------------------------------------------------------------------- #
# Exclusión de equipos en reparación
# --------------------------------------------------------------------------- #
def test_low_coverage_days_are_identified():
    """Un equipo en reparación no emite lecturas y su cobertura cae."""
    equipos = pd.DataFrame({"equipo_code": ["BV1"], "n_signals": [4]})
    daily = pd.DataFrame({
        "equipo_code": ["BV1", "BV1"],
        "kind": ["vibration", "current"],
        "fecha": [date(2026, 1, 1), date(2026, 1, 1)],
        "valor_medio": [2.0, 170.0],
        "n_lecturas": [24, 24],          # 48 de 96 esperadas
    })
    cov = bd.coverage_frame(daily, equipos)
    assert cov["cobertura"].iloc[0] == pytest.approx(0.5)


def test_tag_index_covers_every_signal():
    """El mapa tag-equipo no puede dejar ninguna señal sin asociar."""
    idx = bd.tag_index()
    total = sum(len(eq.signals) for eq in build_plant())
    assert len(idx) == total
