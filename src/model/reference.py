"""Modelo de referencia local, provisional hasta la fase de entrenamiento formal.

Sirve para poder analizar el umbral de operación sin depender del modelo que
produjo AutoML, cuyo empaquetado exige el entorno de ejecución de Azure y no se
carga fuera de él.

Su desempeño queda por debajo del que alcanzó AutoML (0.843 de AUC con un
ensamble de seis modelos, 0.818 el mejor individual). La brecha es el objetivo
a cerrar cuando se escriba el script de entrenamiento definitivo, y tenerla
medida es más útil que arrastrar una caja negra que no se puede inspeccionar.

Una advertencia sobre la imputación. Añadir indicadores de valor faltante sube
el AUC unos cuatro puntos, y conviene entender por qué antes de celebrarlo: los
nulos de este dataset son estructurales, porque no todos los equipos llevan los
mismos sensores, y el patrón de ausencias distingue cinco grupos entre los doce
equipos. Esos indicadores reconstruyen buena parte de la identidad del equipo,
que es exactamente lo que se quiso evitar al excluir su código. No es una fuga
temporal, porque en producción esos equipos tendrán los mismos sensores, pero
significa que el modelo aprende tasas base por grupo y no solo física. Un modelo
así no serviría en otra planta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline

from ..features.build_dataset import COLUMNAS_NO_PREDICTORAS

OBJETIVO = "falla_14d"


def columnas_predictoras(df: pd.DataFrame) -> list[str]:
    excluir = set(COLUMNAS_NO_PREDICTORAS) | {OBJETIVO, "criticidad"}
    return [c for c in df.columns if c not in excluir]


def preparar(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    X = df[columnas].copy()
    X["criticidad_A"] = (df["criticidad"] == "A").astype(int)
    return X


def entrenar(train: pd.DataFrame, add_indicator: bool = True) -> tuple[Pipeline, list[str]]:
    """Ajusta el modelo de referencia sobre el conjunto de entrenamiento."""
    columnas = columnas_predictoras(train)
    modelo = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=add_indicator),
        ExtraTreesClassifier(n_estimators=500, random_state=0, n_jobs=-1),
    )
    modelo.fit(preparar(train, columnas), train[OBJETIVO])
    return modelo, columnas


def puntuar(modelo: Pipeline, columnas: list[str], df: pd.DataFrame) -> np.ndarray:
    return modelo.predict_proba(preparar(df, columnas))[:, 1]
