"""Pruebas de la elección del punto de operación."""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.model import threshold as th


def _tabla(etiquetas, inicio=date(2026, 1, 1)):
    return pd.DataFrame({
        "equipo_code": ["BV1"] * len(etiquetas),
        "fecha": [(inicio + timedelta(days=i)).isoformat() for i in range(len(etiquetas))],
        "falla_14d": etiquetas,
    })


def test_el_umbral_produce_el_numero_de_alertas_pedido():
    scores = np.linspace(0, 1, 100)
    umbral = th._umbral_para_presupuesto(scores, 10)
    assert (scores >= umbral).sum() == 10


def test_mas_presupuesto_nunca_reduce_los_eventos_detectados():
    """Bajar el umbral solo puede añadir alertas, nunca quitarlas."""
    etiquetas = [0] * 60 + [1] * 10 + [0] * 60
    df = _tabla(etiquetas)
    rng = np.random.default_rng(0)
    scores = rng.random(len(df)) + np.array(etiquetas) * 0.4
    curva = th.curva_operacion(df, scores, (2, 8, 20))
    assert curva["recall_eventos"].is_monotonic_increasing


def test_la_saturacion_marca_el_menor_presupuesto_con_el_maximo_recall():
    curva = pd.DataFrame({
        "alertas_mes_objetivo": [1, 4, 8, 16],
        "recall_eventos": [0.2, 0.8, 0.8, 0.8],
        "anticipacion_dias": [1.0, 5.0, 6.0, 7.0],
        "alertas_mes_real": [1.0, 4.0, 8.0, 16.0],
    })
    assert th.punto_de_saturacion(curva) == 4


def test_entre_empates_se_prefiere_la_mayor_anticipacion():
    """Avisar dos días antes no sirve; el desempate operativo es el tiempo."""
    curva = pd.DataFrame({
        "alertas_mes_objetivo": [1, 8],
        "alertas_mes_real": [1.0, 8.0],
        "recall_eventos": [0.44, 0.44],
        "anticipacion_dias": [2.0, 6.0],
    })
    assert th.recomendar(curva, capacidad_mensual=8)["anticipacion_dias"] == 6.0


def test_la_recomendacion_respeta_la_capacidad():
    curva = pd.DataFrame({
        "alertas_mes_objetivo": [4, 40],
        "alertas_mes_real": [4.0, 40.0],
        "recall_eventos": [0.4, 0.9],
        "anticipacion_dias": [5.0, 12.0],
    })
    elegido = th.recomendar(curva, capacidad_mensual=8)
    assert elegido["alertas_mes_real"] == 4.0
