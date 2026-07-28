"""Entrenamiento parametrizado, pensado para correr como trabajo de Azure ML.

Es el script que reemplaza al modelo de AutoML. La razón para escribirlo no es
mejorar el número, que ya se alcanzó, sino recuperar el control: un modelo
propio se puede inspeccionar, versionar, explicar y reentrenar, y las fases de
despliegue y monitoreo necesitan las cuatro cosas.

Uso local::

    python -m src.model.train --train workdir/splits/train --valid workdir/splits/valid

Como trabajo de Azure, los mismos argumentos llegan desde el YAML del job.

**Un detalle que rompe la búsqueda de hiperparámetros si se descuida.** El
nombre con el que aquí se registra la métrica principal tiene que coincidir
exactamente con el declarado en el archivo del sweep. Si no coinciden, la
búsqueda corre sin errores, termina, y devuelve los intentos sin ordenar porque
no encontró ningún valor que comparar. Es un fallo silencioso y de los más
comunes al empezar.
"""
from __future__ import annotations

import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline

from .evaluate import (auc_intra_promedio, costo_en_dolares, evaluar,
                       recall_contra_techo)
from .features import OBJETIVO, matriz
from .threshold import curva_operacion, recomendar

# Nombre de la métrica principal. Debe coincidir con `primary_metric` del sweep.
METRICA_PRINCIPAL = "auc_valid"


def construir_modelo(n_estimators: int, max_features: str, min_samples_leaf: int,
                     random_state: int) -> Pipeline:
    """Arma el pipeline completo, imputación incluida.

    La imputación va dentro del pipeline y no antes, para que al ajustarse use
    solo estadísticas del conjunto de entrenamiento. Calcular la mediana sobre
    todos los datos y después partir sería una fuga discreta pero real.
    """
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        ),
    )


def entrenar_y_evaluar(train: pd.DataFrame, valid: pd.DataFrame,
                       modelo: Pipeline, capacidad: int,
                       ground_truth: pd.DataFrame | None = None) -> dict:
    """Ajusta el modelo y devuelve todas las métricas que interesan."""
    modelo.fit(matriz(train), train[OBJETIVO])
    scores = modelo.predict_proba(matriz(valid))[:, 1]

    curva = curva_operacion(valid, scores)
    punto = recomendar(curva, capacidad)
    resultado = evaluar(valid, scores, float(punto["umbral"]), nombre="modelo")

    metricas = {
        METRICA_PRINCIPAL: float(roc_auc_score(valid[OBJETIVO], scores)),
        # Mide detección de condición, sin el aporte de ordenar equipos por su
        # tasa histórica de falla.
        "auc_intra_equipo": auc_intra_promedio(valid, scores),
        "recall_eventos": resultado.recall_eventos,
        "eventos_detectados": resultado.eventos_detectados,
        "eventos_totales": resultado.eventos_totales,
        "anticipacion_dias": resultado.anticipacion_mediana,
        "falsas_alarmas": resultado.falsos_positivos,
        "costo_usd": costo_en_dolares(resultado),
        "umbral_operacion": float(punto["umbral"]),
    }
    if ground_truth is not None:
        metricas.update(recall_contra_techo(
            valid, scores, float(punto["umbral"]), ground_truth))
    return metricas


def _leer(carpeta: str) -> pd.DataFrame:
    ruta = os.path.join(carpeta, "training_table.csv")
    if not os.path.exists(ruta):
        raise SystemExit(f"No existe {ruta}.")
    return pd.read_csv(ruta)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrena el clasificador de falla a catorce días.")
    parser.add_argument("--train", required=True, help="Carpeta del conjunto de entrenamiento.")
    parser.add_argument("--valid", required=True, help="Carpeta del conjunto de validación.")
    parser.add_argument("--ground-truth", default=None,
                        help="CSV de eventos reales, para separar las fallas silenciosas.")
    parser.add_argument("--output", default=None, help="Carpeta donde guardar el modelo.")

    # Hiperparámetros expuestos, que son los que recorrerá la búsqueda.
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-features", default="sqrt", choices=["sqrt", "log2", "None"])
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=0)

    parser.add_argument("--capacidad", type=int, default=8,
                        help="Inspecciones al mes que mantenimiento puede atender.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    train, valid = _leer(args.train), _leer(args.valid)
    verdad = pd.read_csv(args.ground_truth) if args.ground_truth else None

    modelo = construir_modelo(
        n_estimators=args.n_estimators,
        max_features=None if args.max_features == "None" else args.max_features,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )

    mlflow.log_params({
        "n_estimators": args.n_estimators,
        "max_features": args.max_features,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": args.random_state,
        "n_variables": matriz(train).shape[1],
        "filas_entrenamiento": len(train),
    })

    metricas = entrenar_y_evaluar(train, valid, modelo, args.capacidad, verdad)
    mlflow.log_metrics({k: v for k, v in metricas.items() if pd.notna(v)})

    # Formato MLflow: es lo que habilita el despliegue sin escribir un script de
    # puntuación en la fase de producción.
    mlflow.sklearn.log_model(modelo, name="model")
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        mlflow.sklearn.save_model(modelo, os.path.join(args.output, "model"))

    print(f"{METRICA_PRINCIPAL}: {metricas[METRICA_PRINCIPAL]:.4f}")
    print(f"AUC intra-equipo: {metricas['auc_intra_equipo']:.4f}")
    print(f"Eventos anticipados: {metricas['eventos_detectados']}"
          f"/{metricas['eventos_totales']} "
          f"con {metricas['anticipacion_dias']:.0f} días de mediana")
    if "recall_sobre_techo" in metricas:
        print(f"Sobre los detectables: {metricas['detectables_anticipados']}"
              f"/{metricas['eventos_detectables']}  "
              f"(hay {metricas['eventos_silenciosos']} fallas sin precursor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
