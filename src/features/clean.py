"""Limpieza de las lecturas del historian (capa bronze a capa silver).

El historian entrega tres patologías que hay que resolver antes de calcular
cualquier estadística, porque todas envenenan una media o una pendiente:

1. Lecturas con ``quality = BAD``, que traen el valor centinela -9999. Si entran
   a un promedio lo desploman.
2. Sensores congelados (flatline). Son el caso traicionero: la calidad dice GOOD
   pero el instrumento se trabó y repite el mismo valor durante horas. Se
   detectan por repetición exacta, porque con ruido gaussiano la probabilidad de
   que dos lecturas consecutivas coincidan hasta el último decimal es
   despreciable.
3. Huecos, por parada de máquina o por equipo en reparación. No se rellenan; se
   propagan como ausencia de dato y más adelante se usan para decidir si el día
   de ese equipo es utilizable.
"""
from __future__ import annotations

import pandas as pd

# Longitud mínima de una racha de valores idénticos para considerarla flatline.
# Con tres lecturas consecutivas iguales ya es prácticamente imposible que sea
# ruido legítimo, y es corto como para atrapar congelamientos breves.
MIN_FLATLINE_RUN = 3


def drop_bad_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina las lecturas marcadas como BAD por el historian."""
    return df[df["quality"] == "GOOD"].copy()


def flag_flatlines(df: pd.DataFrame, min_run: int = MIN_FLATLINE_RUN) -> pd.DataFrame:
    """Marca en la columna ``flatline`` las lecturas de un sensor congelado.

    Trabaja por tag y en orden temporal: agrupa las rachas de valores idénticos
    consecutivos y marca aquellas cuya longitud alcanza ``min_run``.
    """
    out = df.sort_values(["tag", "timestamp"]).copy()
    same_as_previous = out["value"].eq(out["value"].shift()) & out["tag"].eq(out["tag"].shift())
    run_id = (~same_as_previous).cumsum()
    run_length = out.groupby(run_id)["value"].transform("size")
    out["flatline"] = run_length >= min_run
    return out


def clean_historian(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la limpieza completa y devuelve solo lecturas confiables."""
    out = drop_bad_quality(df)
    out = flag_flatlines(out)
    return out[~out["flatline"]].drop(columns=["flatline"]).reset_index(drop=True)
