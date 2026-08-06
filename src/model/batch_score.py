"""Script de puntuación del despliegue por lotes.

**Por qué existe, si el modelo permite despliegue sin código.** Se intentó
primero esa vía, que es la recomendada para modelos en formato MLflow, y falló:
Azure construye el entorno a partir del modelo y le añade paquetes propios que
arrastran una versión antigua de matplotlib, la cual ya no compila en Python
moderno porque usa una función del módulo ``platform`` eliminada hace varias
versiones.

El problema no está en el modelo ni en sus dependencias, sino en lo que la
generación automática añade encima. Declarar el entorno propio evita todo eso, y
a cambio obliga a escribir estas dos funciones.

**El contrato de un script de puntuación por lotes** se parece al de uno en
línea pero no es igual:

- ``init()`` se ejecuta una vez cuando arranca el proceso. Aquí se carga el
  modelo, que es lo caro, para no repetirlo en cada archivo.
- ``run(mini_batch)`` recibe una **lista de rutas de archivo**, no los datos.
  Esa es la diferencia con el caso en línea, donde llega el contenido de la
  petición. Devuelve las filas que se acumularán en el archivo de salida.
"""
from __future__ import annotations

import os

import mlflow.sklearn
import pandas as pd

modelo = None


def _ruta_del_modelo() -> str:
    """Localiza el modelo dentro de la carpeta que monta Azure.

    ``AZUREML_MODEL_DIR`` apunta a la raíz de lo registrado. Según cómo se haya
    registrado, el descriptor MLmodel puede estar ahí mismo o dentro de una
    subcarpeta, así que se comprueban ambos casos en vez de suponer uno.
    """
    raiz = os.environ.get("AZUREML_MODEL_DIR", ".")
    if os.path.exists(os.path.join(raiz, "MLmodel")):
        return raiz
    for nombre in os.listdir(raiz):
        candidato = os.path.join(raiz, nombre)
        if os.path.isdir(candidato) and os.path.exists(os.path.join(candidato, "MLmodel")):
            return candidato
    raise RuntimeError(f"No se encontró un modelo MLflow bajo {raiz}")


def init():
    """Carga el modelo una sola vez, al arrancar el proceso."""
    global modelo
    ruta = _ruta_del_modelo()
    modelo = mlflow.sklearn.load_model(ruta)
    print(f"Modelo cargado desde {ruta}")


def run(mini_batch):
    """Puntúa cada archivo del lote y devuelve las filas del resultado.

    Se devuelve la probabilidad y no la clase. La decisión de alertar depende
    del umbral que elija mantenimiento según cuántas inspecciones pueda
    atender, y ese umbral cambia sin que el modelo cambie. Un servicio que
    devolviera solo sí o no obligaría a redesplegar cada vez que se ajusta.
    """
    salida = []
    for ruta in mini_batch:
        datos = pd.read_csv(ruta)

        # La carpeta de entrada puede traer archivos auxiliares que no son
        # tabla de puntuación. Se reconocen porque no traen las columnas que el
        # modelo declara, y se ignoran en vez de hacer fallar el lote entero.
        esperadas = list(modelo.feature_names_in_) if hasattr(
            modelo, "feature_names_in_") else None
        if esperadas and not set(esperadas).issubset(datos.columns):
            print(f"Se ignora {os.path.basename(ruta)}: no trae las columnas del modelo.")
            continue

        entradas = datos[esperadas] if esperadas else datos
        probabilidades = modelo.predict_proba(entradas)[:, 1]

        resultado = datos.copy()
        resultado["probabilidad_falla_14d"] = probabilidades
        salida.append(resultado)

    if not salida:
        return pd.DataFrame()
    return pd.concat(salida, ignore_index=True)
