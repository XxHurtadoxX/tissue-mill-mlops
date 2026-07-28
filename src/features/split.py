"""Parte la tabla de entrenamiento en tres conjuntos por fecha.

Se parte por tiempo y no al azar. Las filas de un mismo equipo en días
consecutivos comparten casi toda su ventana de siete días, y la etiqueta abarca
catorce, así que días vecinos comparten además el mismo evento futuro. Una
partición aleatoria pondría filas casi idénticas a ambos lados y devolvería
métricas infladas que no se sostienen en producción.

La partición temporal, en cambio, reproduce la situación real: entrenar con lo
que ya pasó y evaluar sobre lo que viene después.

Los tres conjuntos y su función::

    train   ajusta los modelos
    valid   compara modelos entre sí y elige el candidato
    test    se abre una sola vez, al final, para la cifra que se reporta

El conjunto de prueba se separa desde ahora aunque no se use hasta más adelante.
Si se dejara mezclado, cada decisión tomada mirando esos datos los contaminaría
y la cifra final dejaría de ser una estimación honesta.

Uso::

    python -m src.features.split --tabla workdir/gold/training_table.csv
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from ..model.evaluate import identificar_eventos
from .build_dataset import COLUMNAS_NO_PREDICTORAS, write_mltable

# Los cortes se eligieron mirando cuántos eventos de falla quedaba en cada lado,
# no cuántas filas. Con 37 eventos en total, repartir por porcentaje de filas
# habría dejado la validación con dos o tres eventos, y el recall por evento se
# movería a saltos de treinta puntos ante un solo acierto.
CORTE_TRAIN = "2025-09-30"
CORTE_VALID = "2026-02-28"


def partir(df: pd.DataFrame, corte_train: str = CORTE_TRAIN,
           corte_valid: str = CORTE_VALID) -> dict[str, pd.DataFrame]:
    """Devuelve los conjuntos, sin solapamiento y en orden temporal.

    Además de los tres conjuntos separados devuelve ``train_full``, que junta
    entrenamiento y validación. Es el que se usa para ajustar el modelo que se
    despliega: una vez elegidos los hiperparámetros, no hay razón para
    desaprovechar los nueve eventos de la validación, que son más de un tercio
    de los disponibles. Lo que no puede tocarse es el conjunto de prueba.
    """
    fecha = df["fecha"].astype(str)
    train = df[fecha <= corte_train].reset_index(drop=True)
    valid = df[(fecha > corte_train) & (fecha <= corte_valid)].reset_index(drop=True)
    return {
        "train": train,
        "valid": valid,
        "test": df[fecha > corte_valid].reset_index(drop=True),
        "train_full": pd.concat([train, valid], ignore_index=True),
    }


def resumen(nombre: str, sub: pd.DataFrame) -> dict:
    con_eventos = sub.assign(evento=identificar_eventos(sub))
    return {
        "conjunto": nombre,
        "filas": len(sub),
        "eventos": int(con_eventos["evento"].dropna().nunique()),
        "positivos_%": round(100 * sub["falla_14d"].mean(), 2),
        "desde": sub["fecha"].min(),
        "hasta": sub["fecha"].max(),
    }


def escribir(sub: pd.DataFrame, destino: str) -> None:
    """Escribe el conjunto con su descriptor MLTable, listo para registrarse.

    El CSV conserva todas las columnas, porque la evaluación local necesita la
    fecha y el equipo para agrupar los días en eventos. El descriptor, en
    cambio, las descarta para que el modelo no las use como predictoras.
    """
    os.makedirs(destino, exist_ok=True)
    sub.to_csv(os.path.join(destino, "training_table.csv"), index=False)
    write_mltable(destino, drop_columns=COLUMNAS_NO_PREDICTORAS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parte la tabla de entrenamiento en train, valid y test.")
    parser.add_argument("--tabla", default="workdir/gold/training_table.csv")
    parser.add_argument("--out", default="workdir/splits")
    parser.add_argument("--corte-train", default=CORTE_TRAIN)
    parser.add_argument("--corte-valid", default=CORTE_VALID)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not os.path.exists(args.tabla):
        raise SystemExit(f"No existe {args.tabla}.")

    df = pd.read_csv(args.tabla)
    partes = partir(df, args.corte_train, args.corte_valid)

    filas = []
    for nombre, sub in partes.items():
        if sub.empty:
            raise SystemExit(f"El conjunto '{nombre}' quedó vacío. Revisa los cortes.")
        escribir(sub, os.path.join(args.out, nombre))
        filas.append(resumen(nombre, sub))

    print(pd.DataFrame(filas).to_string(index=False))
    print(f"\nEscrito en: {os.path.abspath(args.out)}")
    print("El conjunto de prueba queda sellado: no debe mirarse hasta la "
          "evaluación final.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
