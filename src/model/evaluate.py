"""Métricas de evaluación, compartidas por las líneas base y por los modelos.

Se miden dos cosas distintas y conviene no confundirlas.

**Por fila.** Cada fila es un equipo en un día. Sirve para comparar modelos entre
sí y es lo que optimizan los algoritmos, pero no responde la pregunta del
negocio.

**Por evento.** Un evento de falla abarca los catorce días previos a la avería,
así que produce catorce filas positivas. A mantenimiento no le importa acertar
las catorce: le importa que el sistema haya avisado **al menos una vez** con
tiempo suficiente. Un modelo que detecta un solo día de cada evento, pero los
detecta todos, es mejor operativamente que uno que acierta muchas filas
concentradas en la mitad de los eventos.

La distancia entre ambas métricas es grande y reportar solo la primera es la
forma más común de sobrevender un modelo de mantenimiento predictivo.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class Resultado:
    """Resumen de desempeño de un score sobre un conjunto de datos."""

    nombre: str
    umbral: float
    auc: float
    average_precision: float
    # Por fila
    verdaderos_positivos: int
    falsos_positivos: int
    falsos_negativos: int
    verdaderos_negativos: int
    precision: float
    recall: float
    # Por evento
    eventos_totales: int
    eventos_detectados: int
    recall_eventos: float
    anticipacion_mediana: float = np.nan
    extra: dict = field(default_factory=dict)

    def como_fila(self) -> dict:
        d = {
            "modelo": self.nombre,
            "umbral": round(self.umbral, 4),
            "AUC": round(self.auc, 3),
            "AP": round(self.average_precision, 3),
            "precision": round(self.precision, 3),
            "recall_filas": round(self.recall, 3),
            "recall_eventos": round(self.recall_eventos, 3),
            "eventos": f"{self.eventos_detectados}/{self.eventos_totales}",
            "falsas_alarmas": self.falsos_positivos,
            "anticipacion_mediana_dias": self.anticipacion_mediana,
        }
        d.update(self.extra)
        return d


def identificar_eventos(df: pd.DataFrame) -> pd.Series:
    """Asigna un identificador a cada ventana positiva contigua por equipo.

    Los días positivos consecutivos de un mismo equipo pertenecen al mismo
    evento de falla. Las filas negativas quedan sin identificador.
    """
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for code, grp in df.groupby("equipo_code", sort=False):
        grp = grp.sort_values("fecha")
        positivo = grp["falla_14d"] == 1
        # Cada transición de negativo a positivo abre un evento nuevo.
        nuevo_bloque = positivo & ~positivo.shift(fill_value=False)
        bloque = nuevo_bloque.cumsum()
        out.loc[grp.index[positivo]] = [
            f"{code}-{n}" for n in bloque[positivo]
        ]
    return out


def evaluar(df: pd.DataFrame, scores: np.ndarray, umbral: float,
            nombre: str) -> Resultado:
    """Calcula todas las métricas de un score sobre el conjunto dado.

    ``df`` debe traer las columnas ``equipo_code``, ``fecha`` y ``falla_14d``.
    """
    y = df["falla_14d"].to_numpy()
    scores = np.asarray(scores, dtype=float)
    pred = scores >= umbral

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    # El AUC necesita variabilidad en el score; una regla constante no la tiene.
    finitos = np.isfinite(scores)
    if finitos.sum() > 0 and len(np.unique(scores[finitos])) > 1:
        auc = roc_auc_score(y[finitos], scores[finitos])
        ap = average_precision_score(y[finitos], scores[finitos])
    else:
        auc = np.nan
        ap = np.nan

    trabajo = df[["equipo_code", "fecha", "falla_14d"]].copy()
    trabajo["evento"] = identificar_eventos(trabajo)
    trabajo["alerta"] = pred

    eventos = trabajo.dropna(subset=["evento"])
    total_eventos = eventos["evento"].nunique()
    detectados = eventos.groupby("evento")["alerta"].any()
    n_detectados = int(detectados.sum())

    # Anticipación: días entre la primera alerta y el fin de la ventana, que es
    # el día en que se abrió la orden correctiva.
    anticipaciones = []
    for evento, grp in eventos[eventos["alerta"]].groupby("evento"):
        fin = eventos.loc[eventos["evento"] == evento, "fecha"].max()
        primera = grp["fecha"].min()
        anticipaciones.append((pd.Timestamp(fin) - pd.Timestamp(primera)).days)

    return Resultado(
        nombre=nombre,
        umbral=umbral,
        auc=auc,
        average_precision=ap,
        verdaderos_positivos=tp,
        falsos_positivos=fp,
        falsos_negativos=fn,
        verdaderos_negativos=tn,
        precision=precision,
        recall=recall,
        eventos_totales=total_eventos,
        eventos_detectados=n_detectados,
        recall_eventos=n_detectados / total_eventos if total_eventos else 0.0,
        anticipacion_mediana=float(np.median(anticipaciones)) if anticipaciones else np.nan,
    )


def auc_intra_equipo(df: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    """AUC calculado dentro de cada equipo, promediando después.

    Distingue dos capacidades que el AUC global mezcla y que valen cosas muy
    distintas en planta.

    La primera es ordenar equipos entre sí. Es fácil de conseguir y de poco uso:
    basta con saber que las bombas de vacío fallan cuatro veces más que los
    agitadores para obtener un AUC global respetable, aunque el modelo no note
    ninguna degradación. Medido aparte, un modelo que solo conoce la identidad
    del equipo alcanza 0.69 de AUC global.

    La segunda es reconocer, en una misma máquina, que hoy está peor que la
    semana pasada. Es la que de verdad decide a quién se manda a inspeccionar, y
    es la que mide esta función. Ese mismo modelo de solo identidad obtiene aquí
    exactamente 0.50, es decir nada.

    Reportar solo el AUC global permite presentar como logro lo que apenas es un
    censo de tasas históricas.
    """
    trabajo = df[["equipo_code", "falla_14d"]].copy()
    trabajo["score"] = np.asarray(scores, dtype=float)

    por_equipo: dict[str, float] = {}
    for code, grupo in trabajo.groupby("equipo_code"):
        # Un equipo sin fallas en el periodo no aporta información aquí.
        if grupo["falla_14d"].nunique() < 2:
            continue
        por_equipo[code] = float(
            roc_auc_score(grupo["falla_14d"], grupo["score"]))
    return por_equipo


def auc_intra_promedio(df: pd.DataFrame, scores: np.ndarray) -> float:
    """Promedio simple del AUC dentro de cada equipo evaluable."""
    por_equipo = auc_intra_equipo(df, scores)
    return float(np.mean(list(por_equipo.values()))) if por_equipo else np.nan


# Costos de referencia, en dólares, tomados del caso de negocio.
#
# Es tentador expresarlo todo en horas para no depender de precios, y es un
# error: una hora de máquina parada y una hora de técnico inspeccionando no
# valen lo mismo ni de lejos. Mezclarlas hace parecer que atender falsas alarmas
# cuesta tanto como perder producción, y lleva a elegir umbrales demasiado
# conservadores.
USD_HORA_MAQUINA = 1500.0     # punto medio del rango del caso de negocio
USD_HORA_TECNICO = 30.0

HORAS_PARADA = 18.0           # duración típica de una falla no anticipada
USD_FALLA_NO_ANTICIPADA = HORAS_PARADA * USD_HORA_MAQUINA

# Una intervención anticipada ocurre dentro de la parada programada, así que no
# suma tiempo de máquina detenida: solo cuesta mano de obra y repuestos.
USD_INTERVENCION_PROGRAMADA = 500.0

# Una falsa alarma cuesta la hora del técnico que va a revisar y no encuentra
# nada. Es unas novecientas veces más barata que perder la producción.
USD_FALSA_ALARMA = USD_HORA_TECNICO


def costo_en_dolares(res: Resultado) -> float:
    """Traduce un resultado al costo esperado del periodo evaluado.

    La comparación entre modelos se hace sobre este número, no sobre el AUC,
    porque es el que responde la pregunta que hace la gerencia.

    La asimetría es enorme: no anticipar una falla cuesta unas quinientas veces
    más que una inspección en vano. Eso significa que en este problema conviene
    tolerar muchas falsas alarmas con tal de no perder eventos, y explica por
    qué el umbral óptimo queda mucho más bajo de lo que sugeriría maximizar F1.
    """
    no_detectados = res.eventos_totales - res.eventos_detectados
    return (
        res.eventos_detectados * USD_INTERVENCION_PROGRAMADA
        + no_detectados * USD_FALLA_NO_ANTICIPADA
        + res.falsos_positivos * USD_FALSA_ALARMA
    )
