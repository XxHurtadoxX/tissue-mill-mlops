"""Comprueba que las predicciones del despliegue por lotes tengan sentido.

Que un trabajo termine en éxito y deje un archivo no prueba nada. Un modelo mal
empaquetado, con las columnas en otro orden o con el preprocesamiento perdido,
también produce un archivo con la forma correcta y el contenido equivocado.

Las comprobaciones van de menor a mayor exigencia:

1. **Forma.** Tantas predicciones como equipos se enviaron a puntuar.
2. **Rango.** Valores dentro de lo que puede tomar una probabilidad.
3. **Sentido.** El equipo que se sabe en ventana de falla debe puntuar por
   encima de los que se saben sanos. Es la única que detecta un modelo invertido
   o que perdió el preprocesamiento al empaquetarse, porque esos casos devuelven
   números perfectamente válidos con el significado contrario.

Uso::

    python -m src.model.check_endpoint --predicciones predicciones.csv
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

# Umbral operativo elegido en la fase de exploración, con el que el modelo
# anticipa los eventos detectables dentro del presupuesto de inspecciones.
UMBRAL_OPERACION = 0.4


def cargar_predicciones(ruta: str) -> pd.DataFrame:
    """Lee la salida del trabajo por lotes.

    El archivo no trae encabezado y, pese a llamarse csv, Azure separa los
    valores con espacios y no con comas al acumular las filas que devuelve el
    script de puntuación. Se deja que pandas deduzca el separador en vez de
    suponer uno, porque el comportamiento depende de la versión del entorno.

    Cada fila repite las columnas de entrada y añade la predicción al final.
    """
    if os.path.isdir(ruta):
        candidatos = [f for f in os.listdir(ruta) if f.endswith(".csv")]
        if len(candidatos) != 1:
            raise SystemExit(f"Se esperaba un solo CSV en {ruta}, hay {len(candidatos)}.")
        ruta = os.path.join(ruta, candidatos[0])

    return pd.read_csv(ruta, header=None, sep=None, engine="python")


def verificar(predicciones: pd.Series, identidad: pd.DataFrame) -> list[str]:
    """Devuelve los problemas encontrados. Lista vacía significa todo bien."""
    problemas: list[str] = []

    if len(predicciones) != len(identidad):
        problemas.append(
            f"Se enviaron {len(identidad)} equipos y volvieron "
            f"{len(predicciones)} predicciones.")
        return problemas

    if predicciones.isna().any():
        problemas.append("Hay predicciones que no son números.")
        return problemas

    fuera = predicciones[(predicciones < 0) | (predicciones > 1)]
    if len(fuera):
        problemas.append(f"{len(fuera)} predicciones fuera del rango [0, 1].")

    real = identidad["falla_14d"].to_numpy()
    sanos = predicciones[real == 0]
    en_falla = predicciones[real == 1]
    if len(sanos) and len(en_falla) and en_falla.max() <= sanos.median():
        problemas.append(
            f"El equipo en ventana de falla puntúa {en_falla.max():.3f} y la "
            f"mediana de los sanos es {sanos.median():.3f}. El modelo podría "
            f"estar invertido o haber perdido el preprocesamiento.")

    return problemas


def alinear(salida: pd.DataFrame, entrada: pd.DataFrame) -> pd.Series:
    """Empareja cada predicción con la fila de entrada que le corresponde.

    El trabajo por lotes reparte las filas en lotes que pueden procesarse en
    paralelo, así que el orden de salida no tiene por qué coincidir con el de
    entrada. Como cada fila de salida repite sus columnas de entrada, se
    emparejan por esos valores en vez de confiar en la posición.
    """
    n = entrada.shape[1]
    if salida.shape[1] < n + 1:
        raise SystemExit(
            f"La salida trae {salida.shape[1]} columnas y se esperaban al menos "
            f"{n + 1}: las {n} de entrada más la predicción.")

    clave = "__clave__"
    izq = entrada.copy()
    izq[clave] = entrada.round(6).astype(str).agg("|".join, axis=1)

    der = salida.iloc[:, :n].copy()
    der.columns = entrada.columns
    der[clave] = der.round(6).astype(str).agg("|".join, axis=1)
    der["prediccion"] = pd.to_numeric(salida.iloc[:, -1], errors="coerce").to_numpy()

    unido = izq[[clave]].merge(der[[clave, "prediccion"]], on=clave, how="left")
    return unido["prediccion"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verifica las predicciones de un despliegue por lotes.")
    parser.add_argument("--predicciones", required=True,
                        help="Archivo o carpeta con la salida del trabajo.")
    parser.add_argument("--entrada", default="workdir/scoring/a_puntuar.csv",
                        help="Tabla que se envió a puntuar.")
    parser.add_argument("--identidad", default="workdir/scoring/identidad.csv",
                        help="Equipos enviados a puntuar, con su etiqueta real.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    salida = cargar_predicciones(args.predicciones)
    entrada = pd.read_csv(args.entrada)
    identidad = pd.read_csv(args.identidad)

    predicciones = alinear(salida, entrada)
    problemas = verificar(predicciones, identidad)

    tabla = identidad.assign(
        prediccion=predicciones.to_numpy()[:len(identidad)],
    ).sort_values("prediccion", ascending=False)
    tabla["alerta"] = tabla["prediccion"] >= UMBRAL_OPERACION

    print(f"Día puntuado: {identidad['fecha'].iloc[0]}\n")
    print(tabla[["equipo_code", "nombre", "prediccion", "falla_14d", "alerta"]]
          .to_string(index=False))

    if problemas:
        print("\nPROBLEMAS ENCONTRADOS:")
        for p in problemas:
            print(f"  - {p}")
        return 1

    alertados = tabla[tabla["alerta"]]
    print(f"\nEl modelo alerta sobre {len(alertados)} de {len(tabla)} equipos.")
    print("Las predicciones son coherentes: forma correcta, valores en rango, y "
          "el equipo en ventana de falla puntúa por encima de los sanos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
