"""Inyección de suciedad realista sobre las series limpias del historian.

Ningún dato de planta llega limpio. Este módulo reproduce las tres patologías
más comunes de un historian real, de forma determinista (sembrada), para que el
pipeline de la fase de limpieza (bronze -> silver) tenga algo de verdad que
resolver:

* **BAD / -9999**: cuando falla la comunicación, el historian escribe un valor
  imposible y marca la calidad como BAD.
* **Flatline (sensor congelado)**: un sensor se traba y repite el último valor
  durante un tramo, con calidad GOOD (la trampa: parece dato bueno).
"""
from __future__ import annotations

from .plant import rng_for

BAD_VALUE = -9999.0
P_BAD = 0.0018          # probabilidad de lectura BAD por muestra
P_FLATLINE = 0.02       # probabilidad de que una serie-día se congele un tramo


def corrupt_series(values: list[float], seed: int, tag: str, day_iso: str,
                   dirty: bool) -> list[tuple[float, str]]:
    """Devuelve [(valor, calidad)] tras aplicar la suciedad a una serie-día.

    Si ``dirty`` es False, devuelve todo GOOD sin tocar (útil para tests y para
    generar un dataset de referencia perfecto).
    """
    out: list[tuple[float, str]] = [(v, "GOOD") for v in values]
    if not dirty or not values:
        return out

    # 1) Sensor congelado: un tramo repite el valor de arranque (calidad GOOD).
    r = rng_for(seed, "flatline", tag, day_iso)
    if r.random() < P_FLATLINE:
        n = len(values)
        start = r.randrange(n)
        length = r.randint(max(2, n // 12), max(3, n // 4))
        frozen = out[start][0]
        for i in range(start, min(n, start + length)):
            out[i] = (frozen, "GOOD")

    # 2) Lecturas BAD sueltas (-9999): pérdida de comunicación.
    rb = rng_for(seed, "bad", tag, day_iso)
    for i in range(len(out)):
        if rb.random() < P_BAD:
            out[i] = (BAD_VALUE, "BAD")

    return out
