"""Selección de variables para el modelo, con el criterio de cada decisión.

La tabla de la capa gold trae treinta variables. El modelo usa once. Reducir no
fue una preferencia estética: con veinticuatro eventos de falla en el conjunto
de entrenamiento, cada columna añadida diluye la señal, y medirlo costó pasar de
0.75 a 0.85 de AUC quitando en vez de agregando.

## Por qué sobran columnas

Las cinco señales de la planta no son variables intercambiables. Vistas como lo
que son, indicadores de falla con distinto tiempo de anticipación, la jerarquía
física es clara:

| Señal | Qué mide | Cuándo avisa | Cobertura |
|---|---|---|---|
| Vibración | Defecto mecánico: rodamiento, desbalanceo | Primero | 12 de 12 |
| Corriente | Torque, es decir fricción creciente | Temprano, si se normaliza | 10 de 12 |
| Temperatura | Calor por fricción ya establecida | Tarde, aunque específico | 5 de 12 |
| Vacío y presión | Que la máquina dejó de hacer su trabajo | Último: falla funcional | 5 de 12 |

Las columnas de presión existen para un solo equipo, la fan pump. Once de cada
doce filas las traen vacías, e imputarles la mediana equivale a afirmar que un
agitador tiene 340 kPa de descarga cuando no tiene siquiera línea de presión.
Quitarlas mejoró el resultado.

## Por qué la identidad del equipo sí entra

`n_sensores` parece un dato administrativo y resulta ser una variable física
útil. Los equipos se agrupan por instrumentación en tres familias, y esas
familias tienen tasas de falla muy distintas: 2.9% para agitadores y pulpers,
4.7% para fan pump y prensa, 12.7% para las bombas de vacío.

Podría objetarse que eso es aprender tasas base en vez de reconocer degradación,
y es una objeción correcta a medias. Medido aparte, la identidad sola alcanza
0.69 de AUC global pero exactamente 0.50 dentro de cada equipo: por sí sola no
distingue un día bueno de uno malo en la misma máquina.

Su aporte es otro. Permite que el modelo aprenda **umbrales distintos por
familia**, que es precisamente lo que hace la norma ISO 20816 al definir límites
de vibración según la clase de máquina. Lo que alarma en una bomba de vacío no
alarma en un agitador. Con la identidad presente, la discriminación dentro de
cada equipo sube de 0.79 a 0.83.

Se declara de forma explícita y no se deja que los indicadores de valor faltante
la reconstruyan por la puerta de atrás, que es lo que ocurría antes.

## Lo que se probó y no funcionó

- **Variables normalizadas por equipo** (percentil histórico, desviación
  robusta, pendiente de treinta días): costaron entre 1.5 y 7 puntos de AUC.
- **Manejo nativo de nulos** con LightGBM en vez de imputar: 0.66 frente a 0.75.
- **Rebalanceo de clases** en todas sus formas, pesos, submuestreo y SMOTE:
  diferencias dentro del ruido. Era previsible, porque el rebalanceo cambia la
  calibración de las probabilidades y no su ordenamiento, que es lo único que
  mide el AUC. SMOTE además interpola entre días consecutivos del mismo evento,
  así que sintetiza casi-duplicados sin añadir información.
"""
from __future__ import annotations

import pandas as pd

OBJETIVO = "falla_14d"

# Tipos de señal presentes en la planta, usados para contar la instrumentación
# de cada equipo.
TIPOS_SENAL = ("vibration", "temp", "current", "vacuum", "pressure")

# Vibración: el indicador universal y el más temprano.
PREFIJO_PRIMARIO = "vibration"

# Contexto de mantenimiento y de operación, siempre definido.
CONTEXTO = ("dias_desde_orden", "zona_iso_ord", "cobertura")


def anadir_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega las dos variables que describen al equipo, no a su estado."""
    out = df.copy()
    columnas_ratio = [f"{k}_ratio_1_7" for k in TIPOS_SENAL if f"{k}_ratio_1_7" in out]
    out["n_sensores"] = out[columnas_ratio].notna().sum(axis=1)
    out["criticidad_A"] = (out["criticidad"] == "A").astype(int)
    return out


def columnas_modelo(df: pd.DataFrame) -> list[str]:
    """Las once variables que entran al modelo."""
    primarias = [c for c in df.columns if c.startswith(PREFIJO_PRIMARIO)]
    return primarias + list(CONTEXTO) + ["n_sensores", "criticidad_A"]


def matriz(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara la matriz de entrada a partir de la tabla de la capa gold."""
    completo = anadir_derivadas(df)
    return completo[columnas_modelo(completo)]
