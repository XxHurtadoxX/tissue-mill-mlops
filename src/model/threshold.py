"""Elección del umbral de operación a partir de la capacidad de mantenimiento.

Un modelo entrega un puntaje continuo; la planta necesita una decisión binaria.
Convertir uno en otro es la parte del trabajo que decide si el sistema sirve, y
no la resuelve el algoritmo.

**Por qué el umbral por omisión de 0.5 no aplica.** Con menos del 5% de días
positivos, un modelo bien ajustado rara vez asigna probabilidad mayor a 0.5.
Reportar la matriz de confusión en ese punto muestra un recall muy bajo y hace
parecer inútil un modelo que ordena bien.

**Por qué minimizar costo tampoco basta.** No anticipar una falla cuesta unas
quinientas veces más que una inspección en vano. Con esa asimetría, el óptimo
matemático se acerca a "alertar siempre", que es una respuesta degenerada:
ciento diez alertas mensuales para doce equipos son tres o cuatro diarias, y
mantenimiento dejaría de creerle al sistema en dos semanas. A partir de ahí su
valor real es cero, algo que ninguna función de costo con un precio fijo por
falsa alarma logra capturar.

**El planteamiento correcto.** Mantenimiento puede atender cierto número de
inspecciones al mes sin desatender su trabajo. Ese presupuesto es la
restricción, y la pregunta pasa a ser cuántos eventos se alcanzan a anticipar
dentro de él. Es una optimización restringida, no una minimización libre, y es
la conversación que de verdad se tiene en una planta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import costo_en_dolares, evaluar

DIAS_POR_MES = 30.4

# Presupuestos de inspección a evaluar, en alertas por mes para el conjunto de
# equipos vigilados. El rango arranca en una alerta mensual, que es lo que una
# planta acepta sin discusión, y llega hasta un nivel que ya resulta inmanejable
# y sirve para mostrar dónde deja de aportar añadir alertas.
PRESUPUESTOS = (1, 2, 4, 6, 8, 12, 16, 24, 40)


def _meses_cubiertos(df: pd.DataFrame) -> float:
    fechas = pd.to_datetime(df["fecha"])
    dias = (fechas.max() - fechas.min()).days + 1
    return max(dias / DIAS_POR_MES, 1e-9)


def _umbral_para_presupuesto(scores: np.ndarray, n_alertas: int) -> float:
    """Umbral que produce aproximadamente ``n_alertas`` alertas en el periodo."""
    n_alertas = int(min(max(n_alertas, 1), len(scores)))
    ordenados = np.sort(scores)[::-1]
    return float(ordenados[n_alertas - 1])


def curva_operacion(df: pd.DataFrame, scores: np.ndarray,
                    presupuestos: tuple[int, ...] = PRESUPUESTOS) -> pd.DataFrame:
    """Desempeño alcanzable para cada presupuesto de inspecciones mensuales."""
    scores = np.asarray(scores, dtype=float)
    meses = _meses_cubiertos(df)

    filas = []
    for por_mes in presupuestos:
        umbral = _umbral_para_presupuesto(scores, int(round(por_mes * meses)))
        res = evaluar(df, scores, umbral, nombre=f"{por_mes}/mes")
        alertas = res.verdaderos_positivos + res.falsos_positivos
        filas.append({
            "alertas_mes_objetivo": por_mes,
            "alertas_mes_real": round(alertas / meses, 1),
            "umbral": round(umbral, 4),
            "eventos": f"{res.eventos_detectados}/{res.eventos_totales}",
            "recall_eventos": round(res.recall_eventos, 3),
            "anticipacion_dias": res.anticipacion_mediana,
            "alertas_utiles_%": (round(100 * res.verdaderos_positivos / alertas, 1)
                                 if alertas else 0.0),
            "costo_usd": round(costo_en_dolares(res)),
        })
    return pd.DataFrame(filas)


def punto_de_saturacion(curva: pd.DataFrame) -> int | None:
    """Presupuesto a partir del cual añadir alertas ya no anticipa más eventos.

    Es el dato que evita pedirle a mantenimiento más inspecciones de las que
    aportan algo.
    """
    mejor = curva["recall_eventos"].max()
    alcanzan = curva[curva["recall_eventos"] >= mejor]
    if alcanzan.empty:
        return None
    return int(alcanzan["alertas_mes_objetivo"].min())


def recomendar(curva: pd.DataFrame, capacidad_mensual: int) -> pd.Series:
    """Mejor punto de operación que cabe dentro de la capacidad declarada.

    Entre puntos que anticipan la misma cantidad de eventos se prefiere el que
    avisa con más antelación. Es una diferencia práctica importante: detectar
    una falla dos días antes no alcanza para conseguir el repuesto ni para
    meterla en la parada programada, que es justamente el objetivo.
    """
    viables = curva[curva["alertas_mes_real"] <= capacidad_mensual]
    if viables.empty:
        return curva.iloc[0]
    ordenadas = viables.sort_values(
        ["recall_eventos", "anticipacion_dias"], ascending=[False, False])
    return ordenadas.iloc[0]
