"""Líneas base contra las que se compara cualquier modelo posterior.

Sin este paso, un AUC de 0.80 no significa nada. Puede ser un gran avance o
puede ser peor que la regla de dos líneas que la planta ya aplica. Medir la
práctica actual antes de modelar es lo que permite afirmar que el modelo aporta,
y también lo que permite reconocer con honestidad cuando no aporta.

Se miden tres referencias, de menor a mayor exigencia:

1. **Nunca avisar.** El modelo degenerado. Existe para dejar en evidencia por qué
   la exactitud es una métrica inservible aquí: acierta más del 95% de las filas
   y no evita una sola parada.

2. **Regla ISO 20816.** Lo que mantenimiento hace hoy: si la ruta mensual de
   vibración deja el equipo en zona C o peor, se programa intervención. Es el
   rival real.

3. **Umbral sobre la tendencia de vibración.** Usa el historian en continuo en
   lugar de la ruta mensual. Sirve para separar cuánto del mérito de un modelo
   viene de tener datos más frecuentes y cuánto de combinar variables.

Uso::

    python -m src.model.baseline --tabla workdir/gold/training_table.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .evaluate import Resultado, costo_en_dolares, evaluar

# Zona C de la norma ISO 20816, codificada como 2 en la tabla. Es el punto en
# que la práctica actual programa una intervención.
ZONA_ALERTA = 2.0


def linea_base_nunca_avisar(df: pd.DataFrame) -> Resultado:
    """El modelo degenerado que siempre dice que no hay falla."""
    scores = np.zeros(len(df))
    res = evaluar(df, scores, umbral=0.5, nombre="Nunca avisar")
    exactitud = (df["falla_14d"] == 0).mean()
    res.extra["exactitud"] = round(float(exactitud), 3)
    return res


def linea_base_iso(df: pd.DataFrame) -> Resultado:
    """La práctica actual: alertar cuando la ruta mensual marca zona C o peor.

    Los equipos sin medición de ruta vigente quedan sin alerta, que es lo que
    ocurre en la planta: si el técnico no pasó, no hay dato que mirar.
    """
    scores = df["zona_iso_ord"].fillna(0).to_numpy(dtype=float)
    res = evaluar(df, scores, umbral=ZONA_ALERTA, nombre="Regla ISO 20816")
    sin_ruta = int(df["zona_iso_ord"].isna().sum())
    res.extra["filas_sin_ruta"] = sin_ruta
    return res


def linea_base_vibracion(df: pd.DataFrame, percentil: float = 95.0) -> Resultado:
    """Umbral fijo sobre la vibración media de siete días.

    El umbral se fija en un percentil alto de la propia distribución en lugar de
    un valor absoluto, porque cada equipo tiene su nivel base y un número fijo
    solo alertaría sobre los equipos que de por sí vibran más.
    """
    scores = df["vibration_d7_mean"].to_numpy(dtype=float)
    validos = scores[np.isfinite(scores)]
    umbral = float(np.percentile(validos, percentil)) if len(validos) else np.inf
    res = evaluar(df, scores, umbral=umbral,
                  nombre=f"Umbral vibracion p{percentil:.0f}")
    return res


def linea_base_vibracion_relativa(df: pd.DataFrame,
                                  percentil: float = 95.0) -> Resultado:
    """Umbral sobre el cociente entre el día y su propia semana.

    Normaliza cada equipo contra sí mismo, así que no penaliza a las máquinas
    que siempre vibran más. Es la versión honesta del umbral simple.
    """
    scores = df["vibration_ratio_1_7"].to_numpy(dtype=float)
    validos = scores[np.isfinite(scores)]
    umbral = float(np.percentile(validos, percentil)) if len(validos) else np.inf
    return evaluar(df, scores, umbral=umbral,
                   nombre=f"Vibracion relativa p{percentil:.0f}")


def calcular_todas(df: pd.DataFrame) -> list[Resultado]:
    return [
        linea_base_nunca_avisar(df),
        linea_base_iso(df),
        linea_base_vibracion(df),
        linea_base_vibracion_relativa(df),
    ]


def tabla_comparativa(resultados: list[Resultado]) -> pd.DataFrame:
    filas = []
    for res in resultados:
        fila = res.como_fila()
        fila["costo_usd"] = round(costo_en_dolares(res))
        filas.append(fila)
    return pd.DataFrame(filas)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mide las líneas base contra las que se compara el modelo.")
    parser.add_argument("--tabla", default="workdir/gold/training_table.csv",
                        help="Ruta de la tabla de entrenamiento.")
    parser.add_argument("--desde", default=None,
                        help="Filtra desde esta fecha (AAAA-MM-DD), para medir "
                             "solo sobre el conjunto de validación.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not os.path.exists(args.tabla):
        raise SystemExit(
            f"No existe {args.tabla}. Genera antes la tabla de entrenamiento.")

    df = pd.read_csv(args.tabla)
    if args.desde:
        df = df[df["fecha"] >= args.desde].reset_index(drop=True)

    eventos = df.loc[df["falla_14d"] == 1]
    print(f"Filas: {len(df)}  ·  positivos: {int(df['falla_14d'].sum())} "
          f"({100 * df['falla_14d'].mean():.2f}%)  ·  "
          f"rango: {df['fecha'].min()} a {df['fecha'].max()}")
    print(f"Eventos de falla en el periodo: "
          f"{eventos.pipe(lambda d: d.groupby('equipo_code').size().count())} equipos "
          f"afectados\n")

    tabla = tabla_comparativa(calcular_todas(df))
    print(tabla.to_string(index=False))
    print("\ncosto_usd combina las paradas no anticipadas, las intervenciones "
          "hechas a tiempo y el costo de atender falsas alarmas, para todo el "
          "periodo evaluado. Menos es mejor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
