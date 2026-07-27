"""Registra las líneas base y el modelo de referencia como corridas de MLflow.

El objetivo es que todo quede en una sola tabla comparable dentro del workspace,
junto a los modelos que produjo AutoML. Si las líneas base viven en la salida de
una terminal y los modelos en el estudio, nadie compara nada: se termina citando
el AUC del modelo sin contrastarlo con la regla que la planta ya aplica.

Se registra desde la máquina local apuntando al workspace, sin instancia de
cómputo. El workspace actúa como servidor de seguimiento de MLflow, así que las
corridas aparecen en el estudio igual que si se hubieran lanzado allá, y el
seguimiento no consume cómputo ni genera costo.

Uso::

    python -m src.model.experiments --tracking-uri "$(az ml workspace show \\
        -n mlw-tissue-mlops -g rg-tissue-mlops --query mlflow_tracking_uri -o tsv)"

Sin ``--tracking-uri`` registra en una carpeta local, útil para probar el
script sin conexión.
"""
from __future__ import annotations

import argparse
import os

import mlflow
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import reference
from .baseline import calcular_todas
from .evaluate import Resultado, costo_en_dolares, evaluar
from .threshold import PRESUPUESTOS, curva_operacion, recomendar

EXPERIMENTO = "tissue-falla-14d"


def _registrar(res: Resultado, params: dict, tags: dict) -> None:
    """Escribe una corrida con sus parámetros, métricas y etiquetas."""
    with mlflow.start_run(run_name=res.nombre):
        mlflow.log_params(params)
        mlflow.set_tags(tags)
        metricas = {
            "precision": res.precision,
            "recall_filas": res.recall,
            "recall_eventos": res.recall_eventos,
            "eventos_detectados": res.eventos_detectados,
            "eventos_totales": res.eventos_totales,
            "falsas_alarmas": res.falsos_positivos,
            "costo_usd": costo_en_dolares(res),
        }
        if pd.notna(res.auc):
            metricas["AUC"] = res.auc
            metricas["average_precision"] = res.average_precision
        if pd.notna(res.anticipacion_mediana):
            metricas["anticipacion_dias"] = res.anticipacion_mediana
        mlflow.log_metrics(metricas)


def registrar_lineas_base(valid: pd.DataFrame) -> None:
    """Deja constancia de la práctica actual y de las reglas simples."""
    for res in calcular_todas(valid):
        _registrar(
            res,
            params={"tipo": "linea_base", "umbral": round(res.umbral, 4)},
            tags={"familia": "linea_base", "particion": "valid"},
        )


def registrar_modelo(train: pd.DataFrame, valid: pd.DataFrame,
                     capacidad: int = 8) -> float:
    """Entrena el modelo de referencia y registra su punto de operación.

    Se registra una corrida por presupuesto de alertas, porque el desempeño del
    mismo modelo cambia por completo según dónde se ponga el umbral y esa
    variación es la información que necesita mantenimiento para decidir.
    """
    modelo, columnas = reference.entrenar(train)
    scores = reference.puntuar(modelo, columnas, valid)
    auc = roc_auc_score(valid[reference.OBJETIVO], scores)

    curva = curva_operacion(valid, scores)
    elegido = recomendar(curva, capacidad)

    for presupuesto in PRESUPUESTOS:
        punto = curva[curva["alertas_mes_objetivo"] == presupuesto].iloc[0]
        res = evaluar(valid, scores, punto["umbral"],
                      nombre=f"ExtraTrees {presupuesto} alertas/mes")
        _registrar(
            res,
            params={
                "tipo": "modelo",
                "algoritmo": "ExtraTrees",
                "n_estimators": 500,
                "imputacion": "mediana con indicadores",
                "alertas_mes": presupuesto,
                "umbral": round(float(punto["umbral"]), 4),
            },
            tags={
                "familia": "modelo",
                "particion": "valid",
                "recomendado": str(presupuesto == elegido["alertas_mes_objetivo"]),
            },
        )
    return auc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Registra líneas base y modelo como corridas de MLflow.")
    parser.add_argument("--splits", default="workdir/splits")
    parser.add_argument("--tracking-uri", default=None,
                        help="URI del workspace. Sin él registra localmente.")
    parser.add_argument("--capacidad", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENTO)

    train = pd.read_csv(os.path.join(args.splits, "train", "training_table.csv"))
    valid = pd.read_csv(os.path.join(args.splits, "valid", "training_table.csv"))

    registrar_lineas_base(valid)
    auc = registrar_modelo(train, valid, args.capacidad)

    destino = args.tracking_uri or "carpeta local ./mlruns"
    print(f"Corridas registradas en el experimento '{EXPERIMENTO}'.")
    print(f"Destino: {destino}")
    print(f"AUC del modelo de referencia: {auc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
