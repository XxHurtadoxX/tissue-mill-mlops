"""Pruebas de las métricas de evaluación.

La parte delicada es agrupar días positivos en eventos. Si esa agrupación se
rompe, el recall por evento sale mal y con él la comparación entre modelos, sin
que nada falle de forma visible.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.model import evaluate as ev


def _tabla(codigos, etiquetas, inicio=date(2026, 1, 1)):
    """Arma una tabla mínima con un día por fila, por equipo."""
    filas = []
    for i in range(len(etiquetas)):
        filas.append({
            "equipo_code": codigos[i],
            "fecha": (inicio + timedelta(days=i)).isoformat(),
            "falla_14d": etiquetas[i],
        })
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# Agrupación en eventos
# --------------------------------------------------------------------------- #
def test_dias_positivos_contiguos_son_un_solo_evento():
    df = _tabla(["BV1"] * 6, [0, 1, 1, 1, 0, 0])
    eventos = ev.identificar_eventos(df)
    assert eventos.dropna().nunique() == 1


def test_dos_rachas_separadas_son_dos_eventos():
    df = _tabla(["BV1"] * 7, [1, 1, 0, 0, 1, 1, 0])
    eventos = ev.identificar_eventos(df)
    assert eventos.dropna().nunique() == 2


def test_los_eventos_no_se_mezclan_entre_equipos():
    """Dos equipos con positivos el mismo día son dos eventos distintos."""
    df = pd.DataFrame({
        "equipo_code": ["BV1", "BV2", "BV1", "BV2"],
        "fecha": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
        "falla_14d": [1, 1, 1, 1],
    })
    assert ev.identificar_eventos(df).dropna().nunique() == 2


def test_las_filas_negativas_no_pertenecen_a_ningun_evento():
    df = _tabla(["BV1"] * 4, [0, 1, 1, 0])
    eventos = ev.identificar_eventos(df)
    assert eventos.isna().sum() == 2


# --------------------------------------------------------------------------- #
# Recall por evento contra recall por fila
# --------------------------------------------------------------------------- #
def test_una_sola_alerta_basta_para_detectar_el_evento():
    """Es la diferencia central entre la métrica por fila y la de negocio."""
    df = _tabla(["BV1"] * 5, [0, 1, 1, 1, 0])
    scores = np.array([0.0, 0.0, 0.9, 0.0, 0.0])   # acierta un día de tres
    res = ev.evaluar(df, scores, umbral=0.5, nombre="prueba")
    assert res.recall == pytest.approx(1 / 3)      # por fila, mediocre
    assert res.recall_eventos == 1.0               # por evento, perfecto


def test_evento_sin_ninguna_alerta_cuenta_como_perdido():
    df = _tabla(["BV1"] * 4, [0, 1, 1, 0])
    res = ev.evaluar(df, np.zeros(4), umbral=0.5, nombre="prueba")
    assert res.eventos_detectados == 0
    assert res.recall_eventos == 0.0


def test_anticipacion_se_mide_desde_la_primera_alerta():
    """La anticipación es la distancia entre el primer aviso y el fin de la ventana."""
    df = _tabla(["BV1"] * 6, [0, 1, 1, 1, 1, 0])
    scores = np.array([0.0, 0.9, 0.0, 0.0, 0.9, 0.0])
    res = ev.evaluar(df, scores, umbral=0.5, nombre="prueba")
    # Primera alerta el día 2 (índice 1), la ventana termina el día 5 (índice 4).
    assert res.anticipacion_mediana == 3.0


# --------------------------------------------------------------------------- #
# Conteos y costo
# --------------------------------------------------------------------------- #
def test_matriz_de_confusion():
    df = _tabla(["BV1"] * 4, [1, 0, 1, 0])
    scores = np.array([0.9, 0.9, 0.1, 0.1])
    res = ev.evaluar(df, scores, umbral=0.5, nombre="prueba")
    assert (res.verdaderos_positivos, res.falsos_positivos,
            res.falsos_negativos, res.verdaderos_negativos) == (1, 1, 1, 1)


def test_auc_no_se_calcula_si_el_score_es_constante():
    """Una regla que siempre dice lo mismo no tiene poder de ordenamiento."""
    df = _tabla(["BV1"] * 4, [1, 0, 1, 0])
    res = ev.evaluar(df, np.zeros(4), umbral=0.5, nombre="constante")
    assert np.isnan(res.auc)


def test_perder_un_evento_pesa_mucho_mas_que_una_falsa_alarma():
    """La asimetría de costos es la razón de ser del enfoque."""
    perdido = ev.Resultado(
        nombre="a", umbral=0.5, auc=0.5, average_precision=0.5,
        verdaderos_positivos=0, falsos_positivos=0, falsos_negativos=1,
        verdaderos_negativos=0, precision=0.0, recall=0.0,
        eventos_totales=1, eventos_detectados=0, recall_eventos=0.0)
    con_alarmas = ev.Resultado(
        nombre="b", umbral=0.5, auc=0.5, average_precision=0.5,
        verdaderos_positivos=1, falsos_positivos=100, falsos_negativos=0,
        verdaderos_negativos=0, precision=0.5, recall=1.0,
        eventos_totales=1, eventos_detectados=1, recall_eventos=1.0)
    assert ev.costo_en_dolares(perdido) > ev.costo_en_dolares(con_alarmas)


# --------------------------------------------------------------------------- #
# Recall contra el techo de eventos detectables
# --------------------------------------------------------------------------- #
def _verdad(filas):
    return pd.DataFrame(filas, columns=["code", "fecha_falla", "silenciosa", "fuente"])


def test_las_fallas_silenciosas_no_cuentan_como_fracaso():
    """Es la diferencia entre reportar un modelo mediocre y uno en su techo."""
    df = pd.concat([
        _tabla(["BV1"] * 4, [0, 1, 1, 0]),
        _tabla(["BV2"] * 4, [0, 1, 1, 0]),
    ], ignore_index=True)
    df.loc[df.equipo_code == "BV2", "fecha"] = [
        "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]

    # Detecta el evento de BV1 y no el de BV2, que es silencioso.
    scores = np.array([0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
    verdad = _verdad([
        ("BV1", "2026-01-03", 0, "historian"),
        ("BV2", "2026-01-03", 1, "historian"),
    ])
    r = ev.recall_contra_techo(df, scores, 0.5, verdad)

    assert r["eventos_detectables"] == 1
    assert r["detectables_anticipados"] == 1
    assert r["recall_sobre_techo"] == 1.0        # perfecto sobre lo posible
    assert r["eventos_silenciosos"] == 1
    assert r["silenciosos_anticipados"] == 0


def test_no_detectar_un_evento_con_precursor_si_penaliza():
    df = _tabla(["BV1"] * 4, [0, 1, 1, 0])
    verdad = _verdad([("BV1", "2026-01-03", 0, "historian")])
    r = ev.recall_contra_techo(df, np.zeros(4), 0.5, verdad)
    assert r["eventos_detectables"] == 1
    assert r["recall_sobre_techo"] == 0.0
