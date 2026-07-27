"""Pruebas de la partición temporal.

Un solapamiento entre conjuntos no produce ningún error visible: simplemente
infla las métricas de validación. Por eso conviene vigilarlo con pruebas.
"""
from datetime import date, timedelta

import pandas as pd

from src.features.split import partir


def _tabla(n_dias=100, inicio=date(2024, 1, 1)):
    fechas = [(inicio + timedelta(days=i)).isoformat() for i in range(n_dias)]
    return pd.DataFrame({
        "fecha": fechas * 2,
        "equipo_code": ["BV1"] * n_dias + ["BV2"] * n_dias,
        "falla_14d": [0] * (2 * n_dias),
    })


def test_los_conjuntos_no_se_solapan():
    partes = partir(_tabla(), "2024-02-01", "2024-03-01")
    assert partes["train"]["fecha"].max() < partes["valid"]["fecha"].min()
    assert partes["valid"]["fecha"].max() < partes["test"]["fecha"].min()


def test_no_se_pierde_ni_se_duplica_ninguna_fila():
    df = _tabla()
    partes = partir(df, "2024-02-01", "2024-03-01")
    total = sum(len(sub) for sub in partes.values())
    assert total == len(df)


def test_el_orden_temporal_se_respeta():
    """El entrenamiento es siempre pasado respecto de la validación."""
    partes = partir(_tabla(), "2024-02-01", "2024-03-01")
    assert partes["train"]["fecha"].max() == "2024-02-01"
    assert partes["valid"]["fecha"].min() == "2024-02-02"


def test_los_dos_equipos_aparecen_en_cada_conjunto():
    """La partición es por fecha, no por equipo: todos siguen presentes."""
    partes = partir(_tabla(), "2024-02-01", "2024-03-01")
    for sub in partes.values():
        assert set(sub["equipo_code"]) == {"BV1", "BV2"}
