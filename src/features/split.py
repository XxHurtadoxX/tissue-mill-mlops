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

# Duración de las ventanas de validación y prueba, contadas hacia atrás desde el
# último día con datos.
#
# Se expresan como duración y no como fechas fijas porque el pipeline reentrena
# cada mes: con cortes fijos, el conjunto de prueba envejecería hasta volverse
# irrelevante mientras los datos nuevos se acumulan en entrenamiento.
#
# Los ciento cincuenta días salieron de contar eventos, no filas. Repartir por
# porcentaje de filas dejaba la validación con dos o tres eventos, y con tan
# pocos el recall se mueve a saltos de treinta puntos ante un solo acierto.
DIAS_VALID = 150
DIAS_TEST = 150


def calcular_cortes(df: pd.DataFrame, dias_valid: int = DIAS_VALID,
                    dias_test: int = DIAS_TEST) -> tuple[str, str]:
    """Fechas de corte, medidas hacia atrás desde el último día con datos.

    Tiene un efecto que conviene tener presente: cada reentrenamiento evalúa
    contra un conjunto de prueba distinto, así que las cifras entre corridas no
    son estrictamente comparables. Es el precio de mantener la evaluación sobre
    datos recientes, y es el compromiso correcto, porque un número comparable
    pero obsoleto no sirve para decidir nada.
    """
    ultimo = pd.to_datetime(df["fecha"]).max()
    corte_valid = ultimo - pd.Timedelta(days=dias_test)
    corte_train = corte_valid - pd.Timedelta(days=dias_valid)
    return corte_train.date().isoformat(), corte_valid.date().isoformat()


def partir(df: pd.DataFrame, corte_train: str | None = None,
           corte_valid: str | None = None) -> dict[str, pd.DataFrame]:
    """Devuelve los conjuntos, sin solapamiento y en orden temporal.

    Sin cortes explícitos los calcula relativos al último día con datos, que es
    lo que corresponde cuando el pipeline reentrena de forma periódica.

    Además de los tres conjuntos separados devuelve ``train_full``, que junta
    entrenamiento y validación. Es el que se usa para ajustar el modelo que se
    despliega: una vez elegidos los hiperparámetros, no hay razón para
    desaprovechar los eventos de la validación, que son más de un tercio de los
    disponibles. Lo que no puede tocarse es el conjunto de prueba.
    """
    if corte_train is None or corte_valid is None:
        corte_train, corte_valid = calcular_cortes(df)

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
    parser.add_argument("--tabla", default="workdir/gold/training_table.csv",
                        help="Tabla de la capa gold, o carpeta que la contiene.")
    parser.add_argument("--out", default="workdir/splits")
    parser.add_argument("--dias-valid", type=int, default=DIAS_VALID,
                        help="Duración de la ventana de validación, en días.")
    parser.add_argument("--dias-test", type=int, default=DIAS_TEST,
                        help="Duración de la ventana de prueba, en días.")
    # Los cortes explícitos anulan las ventanas relativas. Sirven para
    # reproducir una partición histórica exacta.
    parser.add_argument("--corte-train", default=None)
    parser.add_argument("--corte-valid", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ruta = args.tabla
    if os.path.isdir(ruta):
        ruta = os.path.join(ruta, "training_table.csv")
    if not os.path.exists(ruta):
        raise SystemExit(f"No existe {ruta}.")

    df = pd.read_csv(ruta)
    corte_train, corte_valid = args.corte_train, args.corte_valid
    if corte_train is None or corte_valid is None:
        corte_train, corte_valid = calcular_cortes(
            df, args.dias_valid, args.dias_test)
    print(f"Cortes: entrenamiento hasta {corte_train}, "
          f"validación hasta {corte_valid}\n")
    partes = partir(df, corte_train, corte_valid)

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
