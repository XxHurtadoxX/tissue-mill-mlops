"""Evaluación única sobre el conjunto de prueba.

Este conjunto se apartó en la primera fase y no se ha mirado desde entonces. No
se usó para elegir variables, ni algoritmo, ni hiperparámetros, ni umbral. Por
eso su resultado es la única estimación no sesgada del proyecto, y por eso se
abre una sola vez.

La comparación que justifica correrlo ahora es concreta. La búsqueda de
hiperparámetros encontró una configuración que en validación rinde 0.876 frente
a 0.853 de la configuración por omisión, lo que parece una mejora de dos puntos.
Una medición previa en local, optimizando por validación cruzada temporal dentro
del conjunto de entrenamiento en vez de contra la validación, situaba el margen
real del ajuste en una milésima.

Si esos dos puntos fueran reales, deberían aparecer también aquí. Si no
aparecen, eran selección sobre nueve eventos: cuarenta intentos contra nueve
eventos encuentran ruido y lo llaman señal.

Uso::

    python -m src.model.final_evaluation
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
from sklearn.metrics import roc_auc_score

from .baseline import calcular_todas, tabla_comparativa
from .evaluate import (auc_intra_promedio, costo_en_dolares, evaluar,
                       recall_contra_techo)
from .features import OBJETIVO, matriz
from .threshold import curva_operacion, recomendar
from .train import construir_modelo

# Las dos configuraciones que se comparan.
POR_OMISION = {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 1}
DEL_SWEEP = {"n_estimators": 200, "max_features": "log2", "min_samples_leaf": 4}


def medir(train: pd.DataFrame, evaluacion: pd.DataFrame, config: dict,
          capacidad: int, verdad: pd.DataFrame | None = None) -> dict:
    """Entrena con la configuración dada y mide sobre el conjunto indicado."""
    modelo = construir_modelo(random_state=0, **config)
    modelo.fit(matriz(train), train[OBJETIVO])
    scores = modelo.predict_proba(matriz(evaluacion))[:, 1]

    punto = recomendar(curva_operacion(evaluacion, scores), capacidad)
    resultado = evaluar(evaluacion, scores, float(punto["umbral"]), nombre="final")

    fila = {
        "AUC": round(float(roc_auc_score(evaluacion[OBJETIVO], scores)), 4),
        "AUC_intra": round(auc_intra_promedio(evaluacion, scores), 4),
        "eventos": f"{resultado.eventos_detectados}/{resultado.eventos_totales}",
        "anticipacion_dias": resultado.anticipacion_mediana,
        "falsas_alarmas": resultado.falsos_positivos,
        "costo_usd": round(costo_en_dolares(resultado)),
    }
    if verdad is not None:
        techo = recall_contra_techo(evaluacion, scores, float(punto["umbral"]), verdad)
        fila["sobre_detectables"] = (
            f"{techo['detectables_anticipados']}/{techo['eventos_detectables']}")
        fila["silenciosas"] = techo["eventos_silenciosos"]
    return fila


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evalúa una sola vez sobre el conjunto de prueba sellado.")
    parser.add_argument("--splits", default="workdir/splits")
    parser.add_argument("--ground-truth", default="workdir/ground_truth/eventos.csv")
    parser.add_argument("--capacidad", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def leer(nombre: str) -> pd.DataFrame:
        return pd.read_csv(os.path.join(args.splits, nombre, "training_table.csv"))

    train, valid, test = leer("train"), leer("valid"), leer("test")
    verdad = pd.read_csv(args.ground_truth) if os.path.exists(args.ground_truth) else None

    print(f"Conjunto de prueba: {len(test)} filas, "
          f"{int(test[OBJETIVO].sum())} positivos, "
          f"{test['fecha'].min()} a {test['fecha'].max()}\n")

    print("¿SOBREVIVE LA MEJORA DE LA BÚSQUEDA?")
    comparacion = pd.DataFrame([
        {"configuracion": "por omisión", "conjunto": "validación",
         **medir(train, valid, POR_OMISION, args.capacidad)},
        {"configuracion": "del sweep", "conjunto": "validación",
         **medir(train, valid, DEL_SWEEP, args.capacidad)},
        {"configuracion": "por omisión", "conjunto": "PRUEBA",
         **medir(train, test, POR_OMISION, args.capacidad, verdad)},
        {"configuracion": "del sweep", "conjunto": "PRUEBA",
         **medir(train, test, DEL_SWEEP, args.capacidad, verdad)},
    ])
    print(comparacion.to_string(index=False))

    # Modelo final: se reentrena con todo lo disponible antes de la prueba, que
    # es lo que se desplegaría. Los hiperparámetros ya están decididos, así que
    # sumar la validación al entrenamiento no contamina nada.
    completo = pd.concat([train, valid], ignore_index=True)
    print("\nMODELO FINAL, reentrenado con entrenamiento y validación juntos")
    final = medir(completo, test, POR_OMISION, args.capacidad, verdad)
    print(pd.DataFrame([final]).to_string(index=False))

    print("\nLÍNEAS BASE SOBRE EL MISMO PERIODO DE PRUEBA")
    print(tabla_comparativa(calcular_todas(test)).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
