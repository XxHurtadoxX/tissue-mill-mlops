"""Traduce el puntaje del modelo en una decisión operativa.

Responde la pregunta que hace mantenimiento: si el equipo puede atender N
inspecciones al mes, ¿cuántas fallas alcanza a anticipar y con cuánto tiempo?

Uso::

    python -m src.model.operating_point --capacidad 8
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
from sklearn.metrics import roc_auc_score

from . import features, reference
from .baseline import calcular_todas, tabla_comparativa
from .threshold import curva_operacion, punto_de_saturacion, recomendar


def comparacion_justa(valid: pd.DataFrame, scores_modelo,
                      presupuestos: tuple[int, ...] = (4, 8, 12, 16, 24)) -> pd.DataFrame:
    """Compara el modelo con las reglas simples al mismo volumen de alertas.

    Sin igualar el presupuesto la comparación no dice nada, porque cualquier
    regla anticipa más eventos si se le permite alertar más seguido. Lo que
    interesa es cuántos eventos alcanza cada método cuando mantenimiento puede
    atender el mismo número de inspecciones.
    """
    fuentes = {
        "modelo": scores_modelo,
        "vibracion_relativa": valid["vibration_ratio_1_7"].fillna(0).to_numpy(),
        "vibracion_absoluta": valid["vibration_d7_mean"].fillna(0).to_numpy(),
    }
    filas = []
    for presupuesto in presupuestos:
        fila = {"alertas_mes": presupuesto}
        for nombre, score in fuentes.items():
            punto = curva_operacion(valid, score, (presupuesto,)).iloc[0]
            fila[nombre] = punto["eventos"]
            fila[f"{nombre}_dias"] = punto["anticipacion_dias"]
        filas.append(fila)
    return pd.DataFrame(filas)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Elige el umbral de operación según la capacidad de inspección.")
    parser.add_argument("--splits", default="workdir/splits",
                        help="Carpeta con los conjuntos train, valid y test.")
    parser.add_argument("--capacidad", type=int, default=8,
                        help="Inspecciones al mes que mantenimiento puede atender.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ruta_train = os.path.join(args.splits, "train", "training_table.csv")
    ruta_valid = os.path.join(args.splits, "valid", "training_table.csv")
    for ruta in (ruta_train, ruta_valid):
        if not os.path.exists(ruta):
            raise SystemExit(f"No existe {ruta}. Genera antes la partición.")

    train = pd.read_csv(ruta_train)
    valid = pd.read_csv(ruta_valid)

    modelo = reference.entrenar(train)
    scores = reference.puntuar(modelo, valid)
    auc = roc_auc_score(valid[features.OBJETIVO], scores)

    print(f"Validación: {len(valid)} filas, {int(valid[features.OBJETIVO].sum())} "
          f"positivos, {valid['fecha'].min()} a {valid['fecha'].max()}")
    print(f"AUC del modelo de referencia: {auc:.3f}  "
          f"(AutoML alcanzó 0.843 con un ensamble de seis modelos)\n")

    print("LÍNEAS BASE SOBRE EL MISMO PERIODO")
    print(tabla_comparativa(calcular_todas(valid)).to_string(index=False))

    curva = curva_operacion(valid, scores)
    print("\nCURVA DE OPERACIÓN DEL MODELO")
    print(curva.to_string(index=False))

    print("\nMODELO CONTRA REGLAS, AL MISMO PRESUPUESTO DE ALERTAS")
    print(comparacion_justa(valid, scores).to_string(index=False))
    print("Comparar a igual volumen de alertas es la única forma honesta de "
          "saber si el modelo aporta: cualquier regla detecta más eventos si se "
          "le permite alertar más seguido.")

    saturacion = punto_de_saturacion(curva)
    elegido = recomendar(curva, args.capacidad)

    print(f"\nCon capacidad para {args.capacidad} inspecciones al mes, el mejor "
          f"punto anticipa {elegido['eventos']} eventos")
    print(f"con {elegido['anticipacion_dias']:.0f} días de antelación mediana, "
          f"usando el umbral {elegido['umbral']}.")
    if saturacion is not None:
        print(f"A partir de {saturacion} alertas mensuales, añadir más deja de "
              f"anticipar eventos nuevos.")
    print("\nEl umbral final lo decide mantenimiento sobre esta tabla, no el "
          "científico de datos sobre una métrica.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
