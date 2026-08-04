"""Pruebas de la partición temporal.

Un solapamiento entre conjuntos no produce ningún error visible: simplemente
infla las métricas de validación. Por eso conviene vigilarlo con pruebas.
"""
from datetime import date, timedelta

import pandas as pd

from src.features.split import calcular_cortes, partir


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
    """Los tres conjuntos disjuntos cubren exactamente la tabla original."""
    df = _tabla()
    partes = partir(df, "2024-02-01", "2024-03-01")
    total = sum(len(partes[k]) for k in ("train", "valid", "test"))
    assert total == len(df)


def test_train_full_junta_entrenamiento_y_validacion():
    """Es el conjunto con el que se ajusta el modelo que se despliega."""
    partes = partir(_tabla(), "2024-02-01", "2024-03-01")
    assert len(partes["train_full"]) == len(partes["train"]) + len(partes["valid"])


def test_train_full_no_incluye_el_conjunto_de_prueba():
    """La regla que no se puede romper: la prueba queda fuera de todo ajuste."""
    partes = partir(_tabla(), "2024-02-01", "2024-03-01")
    assert partes["train_full"]["fecha"].max() < partes["test"]["fecha"].min()


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


# --------------------------------------------------------------------------- #
# Cortes relativos al último día con datos
# --------------------------------------------------------------------------- #
def test_los_cortes_se_miden_desde_el_ultimo_dia_con_datos():
    """Con cortes fijos, el conjunto de prueba envejecería en cada reentrenamiento."""
    df = _tabla(n_dias=400, inicio=date(2024, 1, 1))
    corte_train, corte_valid = calcular_cortes(df, dias_valid=100, dias_test=50)
    ultimo = df["fecha"].max()
    assert corte_valid == (date.fromisoformat(ultimo) - timedelta(days=50)).isoformat()
    assert corte_train == (date.fromisoformat(ultimo) - timedelta(days=150)).isoformat()


def test_sin_cortes_explicitos_se_usan_los_relativos():
    df = _tabla(n_dias=400)
    partes = partir(df)
    assert not partes["test"].empty
    assert partes["train"]["fecha"].max() < partes["valid"]["fecha"].min()


def test_la_prueba_siempre_es_el_periodo_mas_reciente():
    df = _tabla(n_dias=400)
    partes = partir(df)
    assert partes["test"]["fecha"].max() == df["fecha"].max()


def test_los_cortes_explicitos_anulan_las_ventanas_relativas():
    """Sirve para reproducir una partición histórica exacta."""
    df = _tabla(n_dias=400)
    partes = partir(df, "2024-02-01", "2024-03-01")
    assert partes["train"]["fecha"].max() == "2024-02-01"
