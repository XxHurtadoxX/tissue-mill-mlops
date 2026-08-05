"""Decide si el modelo recién entrenado debe reemplazar al de producción.

Es el paso que separa un pipeline útil de uno peligroso. Un pipeline que
registra el modelo nuevo cada vez que corre acaba, tarde o temprano,
sustituyendo uno bueno por uno peor: basta un mes con datos raros, un cambio de
proceso o una racha de fallas silenciosas.

**Por qué el criterio no puede ser el AUC.** Ya se comprobó en este proyecto que
un modelo puede ganar varios puntos de AUC sin anticipar un solo evento más. El
AUC mide ordenamiento, y buena parte de ese ordenamiento se consigue sabiendo
qué equipos fallan más, no notando que uno se está degradando.

El criterio, en orden de prioridad:

1. ¿Anticipa al menos tantos **eventos con precursor** como el que está en
   producción? Es lo único que mantenimiento nota.
2. A igual número de eventos, ¿avisa con más antelación? Detectar una falla dos
   días antes no alcanza para conseguir el repuesto.
3. ¿Se mantiene dentro del presupuesto de inspecciones acordado?

Si no supera al actual, no se registra y el pipeline **termina bien**. No
registrar es un desenlace legítimo, no un fallo.

Uso::

    python -m src.model.gate --modelo outputs/model --test workdir/splits/test \\
        --ground-truth workdir/ground_truth/eventos.csv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from .evaluate import auc_intra_promedio, evaluar, recall_contra_techo
from .features import OBJETIVO, matriz
from .threshold import curva_operacion, recomendar

# Piso absoluto. Por debajo de esto no se registra aunque supere al modelo
# actual: un desplome así casi siempre significa que los datos cambiaron, no que
# el modelo mejoró o empeoró, y eso merece que lo mire una persona.
AUC_MINIMO = 0.70


def medir(modelo, datos: pd.DataFrame, capacidad: int,
          ground_truth: pd.DataFrame | None) -> dict:
    """Evalúa un modelo sobre el conjunto dado, en las métricas que deciden."""
    scores = modelo.predict_proba(matriz(datos))[:, 1]
    punto = recomendar(curva_operacion(datos, scores), capacidad)
    resultado = evaluar(datos, scores, float(punto["umbral"]), nombre="candidato")

    from sklearn.metrics import roc_auc_score
    metricas = {
        "auc": float(roc_auc_score(datos[OBJETIVO], scores)),
        "auc_intra": auc_intra_promedio(datos, scores),
        "eventos_detectados": resultado.eventos_detectados,
        "eventos_totales": resultado.eventos_totales,
        "anticipacion_dias": resultado.anticipacion_mediana,
        "falsas_alarmas": resultado.falsos_positivos,
        "umbral": float(punto["umbral"]),
        "alertas_mes": float(punto["alertas_mes_real"]),
    }
    if ground_truth is not None:
        techo = recall_contra_techo(
            datos, scores, float(punto["umbral"]), ground_truth)
        metricas["detectables_anticipados"] = techo["detectables_anticipados"]
        metricas["eventos_detectables"] = techo["eventos_detectables"]
    return metricas


def _anticipados(m: dict) -> int:
    """Eventos con precursor anticipados, o los totales si no hay ground truth."""
    return int(m.get("detectables_anticipados", m["eventos_detectados"]))


def decidir(candidato: dict, produccion: dict | None,
            capacidad: int) -> tuple[bool, str]:
    """Devuelve si registrar y la razón, en lenguaje que se pueda leer en un log."""
    if candidato["auc"] < AUC_MINIMO:
        return False, (
            f"AUC {candidato['auc']:.3f} por debajo del piso {AUC_MINIMO}. "
            f"Un desplome así suele indicar un cambio en los datos: revisar "
            f"antes de reentrenar de nuevo.")

    if candidato["alertas_mes"] > capacidad * 1.5:
        return False, (
            f"Generaría {candidato['alertas_mes']:.0f} alertas al mes, muy por "
            f"encima de las {capacidad} que mantenimiento puede atender.")

    if produccion is None:
        return True, "No hay modelo en producción: se registra el primero."

    nuevos, actuales = _anticipados(candidato), _anticipados(produccion)
    if nuevos > actuales:
        return True, f"Anticipa {nuevos} eventos con precursor frente a {actuales}."
    if nuevos < actuales:
        return False, (
            f"Anticipa {nuevos} eventos con precursor frente a {actuales} del "
            f"modelo actual. Se conserva el que está en producción.")

    # A igual número de eventos, decide la antelación.
    dias_nuevo = candidato.get("anticipacion_dias", np.nan)
    dias_actual = produccion.get("anticipacion_dias", np.nan)
    if pd.notna(dias_nuevo) and pd.notna(dias_actual):
        if dias_nuevo > dias_actual + 0.5:
            return True, (
                f"Anticipa los mismos {nuevos} eventos pero con "
                f"{dias_nuevo:.1f} días de antelación frente a {dias_actual:.1f}.")
        if dias_nuevo < dias_actual - 0.5:
            return False, (
                f"Anticipa los mismos {nuevos} eventos con menos antelación: "
                f"{dias_nuevo:.1f} días frente a {dias_actual:.1f}.")

    return False, (
        f"Empata con el modelo actual en {nuevos} eventos y en antelación. "
        f"Sin mejora demostrable, se conserva el que ya está validado en planta.")


def _leer(ruta: str) -> pd.DataFrame:
    if os.path.isdir(ruta):
        ruta = os.path.join(ruta, "training_table.csv")
    return pd.read_csv(ruta)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decide si el modelo nuevo reemplaza al de producción.")
    parser.add_argument("--modelo", required=True, help="Carpeta del modelo candidato.")
    parser.add_argument("--test", required=True, help="Conjunto de evaluación.")
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--modelo-produccion", default=None,
                        help="Nombre registrado del modelo actual. Sin él se "
                             "asume que no hay ninguno.")
    parser.add_argument("--capacidad", type=int, default=8)
    parser.add_argument("--salida", default=None,
                        help="Archivo donde escribir el veredicto en JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Ver la nota sobre esta importación diferida en src/model/train.py.
    import mlflow
    import mlflow.sklearn

    args = build_parser().parse_args(argv)

    datos = _leer(args.test)
    verdad = pd.read_csv(args.ground_truth) if args.ground_truth else None

    candidato = mlflow.sklearn.load_model(args.modelo)
    metricas_candidato = medir(candidato, datos, args.capacidad, verdad)

    metricas_produccion = None
    if args.modelo_produccion:
        try:
            actual = mlflow.sklearn.load_model(f"models:/{args.modelo_produccion}/latest")
            metricas_produccion = medir(actual, datos, args.capacidad, verdad)
        except Exception as exc:  # noqa: BLE001
            print(f"Aviso: no se pudo cargar el modelo en producción ({exc}). "
                  f"Se decide como si no hubiera ninguno.")

    registrar, razon = decidir(metricas_candidato, metricas_produccion, args.capacidad)

    print("CANDIDATO")
    print(pd.Series(metricas_candidato).to_string())
    if metricas_produccion:
        print("\nEN PRODUCCIÓN")
        print(pd.Series(metricas_produccion).to_string())
    print(f"\nVEREDICTO: {'REGISTRAR' if registrar else 'CONSERVAR EL ACTUAL'}")
    print(razon)

    mlflow.log_metrics({f"candidato_{k}": v for k, v in metricas_candidato.items()
                        if isinstance(v, (int, float)) and pd.notna(v)})
    mlflow.log_param("registrar", registrar)
    mlflow.set_tag("veredicto", razon)

    if args.salida:
        os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
        with open(args.salida, "w", encoding="utf-8") as fh:
            json.dump({"registrar": registrar, "razon": razon,
                       "candidato": metricas_candidato,
                       "produccion": metricas_produccion}, fh,
                      indent=2, ensure_ascii=False, default=float)

    # Se termina siempre en cero. No registrar es un desenlace legítimo, y hacer
    # fallar el pipeline por eso llenaría de alertas falsas a quien lo vigile.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
