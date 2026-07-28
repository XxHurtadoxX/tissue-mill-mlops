"""Modelo de referencia: ExtraTrees sobre el conjunto reducido de variables.

La selección de variables y el criterio detrás de cada decisión están en
``features.py``. Aquí solo queda el ajuste, que resultó ser la parte menos
interesante del problema: probar familias de algoritmos y buscar
hiperparámetros movió el resultado mucho menos que decidir qué columnas entran.

Sobre la imputación. Se conserva la mediana con indicadores de valor faltante,
que fue lo que mejor funcionó, aunque conviene entender qué está haciendo. Los
nulos de esta tabla son estructurales, porque no todos los equipos llevan los
mismos sensores, así que los indicadores describen la instrumentación de la
máquina y no la calidad del dato. Al declarar ``n_sensores`` de forma explícita
esa información ya entra por la puerta principal, y los indicadores pasan a ser
redundantes en su mayor parte. Se dejan porque quitarlos no mejoró nada y el
código queda más simple con una sola estrategia de imputación.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline

from .features import OBJETIVO, matriz

N_ARBOLES = 500


def entrenar(train: pd.DataFrame, n_estimators: int = N_ARBOLES,
             random_state: int = 0) -> Pipeline:
    """Ajusta el modelo sobre el conjunto de entrenamiento."""
    modelo = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        ExtraTreesClassifier(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1),
    )
    modelo.fit(matriz(train), train[OBJETIVO])
    return modelo


def puntuar(modelo: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Probabilidad de falla en los próximos catorce días, por fila."""
    return modelo.predict_proba(matriz(df))[:, 1]
