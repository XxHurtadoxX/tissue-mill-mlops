"""Construye la tabla de entrenamiento (capa gold) a partir de los datos bronze.

Produce una fila por equipo y por día, con features calculadas sobre ventanas
móviles y una etiqueta binaria que responde si ese equipo falla dentro de los
siguientes 14 días.

Uso::

    python -m src.features.build_dataset --data workdir --out workdir/gold

Dos decisiones de diseño merecen explicación, porque son las que sostienen la
validez del dataset.

**De dónde salen las etiquetas.** Se derivan de las órdenes correctivas de SAP
(tipo PM01), no del archivo ``ground_truth/eventos.csv``. En una planta real
nadie conoce la fecha en que empezó la degradación; lo único registrado es
cuándo alguien reportó la avería, con su retraso y su ruido. Entrenar contra el
oráculo daría un modelo que en producción no se puede replicar. El ground truth
queda reservado para medir con cuánta anticipación avisó el modelo.

**Cómo se evita la fuga de información.** Todas las ventanas miran hacia atrás,
los días con el equipo en reparación se descartan (si se dejaran, el modelo
aprendería a reconocer paradas que ya ocurrieron en vez de anticiparlas) y las
lecturas sucias se eliminan antes de promediar. Cada una de estas guardas tiene
su prueba en ``tests/test_features.py``.
"""
from __future__ import annotations

import argparse
import glob
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from ..simulator import plant as plant_mod
from .clean import clean_historian

# Horizonte de predicción: la pregunta es si el equipo falla en los próximos N días.
# Catorce días le dan a mantenimiento margen para meter la intervención en la
# parada programada de la semana en vez de atenderla de urgencia.
HORIZON_DAYS = 14

# Ventana larga para tendencias. Una semana suaviza el ciclo de carga diario.
LONG_WINDOW_DAYS = 7

# Cobertura mínima de datos para considerar utilizable el día de un equipo.
# Por debajo de esto se asume equipo en reparación o instrumentación caída.
MIN_COVERAGE = 0.5

SIGNAL_KINDS = ("current", "vibration", "temp", "vacuum", "pressure")
ISO_ZONE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}


# --------------------------------------------------------------------------- #
# Carga de las fuentes bronze
# --------------------------------------------------------------------------- #
def tag_index() -> dict[str, tuple[str, str]]:
    """Mapea cada tag del historian a su equipo y tipo de señal.

    Es el equivalente en miniatura del mapa tag-equipo que en una planta real
    hay que levantar a mano con el instrumentista, porque ningún sistema tiene
    ese cruce.
    """
    index: dict[str, tuple[str, str]] = {}
    for eq in plant_mod.build_plant():
        for sig in eq.signals:
            index[eq.tag(sig)] = (eq.code, sig.kind)
    return index


def equipment_frame() -> pd.DataFrame:
    """Catálogo de equipos con su identidad de SAP y su criticidad."""
    rows = [
        {
            "equipo_code": eq.code,
            "eq_id": eq.eq_id,
            "nombre": eq.name,
            "criticidad": eq.criticality,
            "n_signals": len(eq.signals),
        }
        for eq in plant_mod.build_plant()
    ]
    return pd.DataFrame(rows)


def load_historian(data_dir: str) -> pd.DataFrame:
    """Lee todos los extractos diarios del historian y los limpia."""
    pattern = os.path.join(data_dir, "bronze", "historian", "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No se encontraron extractos en {pattern}")
    raw = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    return clean_historian(raw)


def load_production(data_dir: str) -> pd.DataFrame:
    """Carga diaria de la planta, en kilogramos procesados por los pulpers."""
    pattern = os.path.join(data_dir, "bronze", "produccion", "batches_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame(columns=["fecha", "produccion_kg"])
    raw = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    raw["fecha"] = pd.to_datetime(raw["fecha"]).dt.date
    return raw.groupby("fecha", as_index=False)["kg_cargados"].sum().rename(
        columns={"kg_cargados": "produccion_kg"})


def load_routes(data_dir: str) -> pd.DataFrame:
    """Mediciones de la ruta de vibración mensual, por equipo."""
    pattern = os.path.join(data_dir, "bronze", "rutas", "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame(columns=["fecha", "equipo_code", "zona_iso_ord", "rms_ruta"])
    raw = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    raw["fecha"] = pd.to_datetime(raw["fecha"]).dt.date

    point_to_code = {
        eq.route_point: eq.code for eq in plant_mod.build_plant() if eq.route_point
    }
    raw["equipo_code"] = raw["punto"].map(point_to_code)
    raw = raw.dropna(subset=["equipo_code"])
    raw["zona_iso_ord"] = raw["zona_iso"].map(ISO_ZONE_ORDER)
    return raw[["fecha", "equipo_code", "zona_iso_ord", "rms_mm_s"]].rename(
        columns={"rms_mm_s": "rms_ruta"})


def load_orders(data_dir: str) -> pd.DataFrame:
    """Órdenes de trabajo de SAP, que son la fuente de las etiquetas."""
    path = os.path.join(data_dir, "bronze", "sap", "ordenes.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["equipo_code", "fecha_orden", "tipo_orden"])
    raw = pd.read_csv(path)
    raw["fecha_orden"] = pd.to_datetime(raw["fecha_aviso"]).dt.date

    id_to_code = {eq.eq_id: eq.code for eq in plant_mod.build_plant()}
    raw["equipo_code"] = raw["equipo"].astype(str).map(id_to_code)
    raw = raw.dropna(subset=["equipo_code"])
    return raw[["equipo_code", "fecha_orden", "tipo_orden"]]


# --------------------------------------------------------------------------- #
# Agregación diaria y ventanas móviles
# --------------------------------------------------------------------------- #
def daily_aggregates(hist: pd.DataFrame) -> pd.DataFrame:
    """Resume el historian a una fila por equipo, tipo de señal y día."""
    idx = tag_index()
    hist = hist[hist["tag"].isin(idx)].copy()
    hist["equipo_code"] = hist["tag"].map(lambda t: idx[t][0])
    hist["kind"] = hist["tag"].map(lambda t: idx[t][1])
    hist["fecha"] = hist["timestamp"].dt.date

    return hist.groupby(["equipo_code", "kind", "fecha"], as_index=False).agg(
        valor_medio=("value", "mean"),
        n_lecturas=("value", "size"),
    )


def _slope(values: np.ndarray) -> float:
    """Pendiente por día de una ventana, vía ajuste lineal por mínimos cuadrados."""
    if len(values) < 2 or np.isnan(values).any():
        return np.nan
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def rolling_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Calcula ventanas móviles por equipo y señal, mirando solo hacia atrás.

    ``rolling`` de pandas usa por defecto ventanas que terminan en la fila
    actual, así que ninguna feature incorpora información futura.
    """
    frames = []
    for (code, kind), grp in daily.groupby(["equipo_code", "kind"], sort=False):
        grp = grp.sort_values("fecha")
        serie = grp["valor_medio"]
        window = serie.rolling(LONG_WINDOW_DAYS, min_periods=LONG_WINDOW_DAYS)

        out = pd.DataFrame({
            "equipo_code": code,
            "kind": kind,
            "fecha": grp["fecha"].to_numpy(),
            "d1_mean": serie.to_numpy(),
            "d7_mean": window.mean().to_numpy(),
            "d7_std": window.std().to_numpy(),
            "d7_slope": window.apply(_slope, raw=True).to_numpy(),
        })
        out["ratio_1_7"] = out["d1_mean"] / out["d7_mean"]
        frames.append(out)

    # Se arma en formato largo y se pivota al final. Nombrar las columnas por
    # tipo de señal durante el bucle provocaría choques, porque varios equipos
    # comparten el mismo tipo (todos tienen corriente, por ejemplo).
    largo = pd.concat(frames, ignore_index=True)
    stats = ["d1_mean", "d7_mean", "d7_std", "d7_slope", "ratio_1_7"]
    ancho = largo.pivot(index=["equipo_code", "fecha"], columns="kind", values=stats)
    ancho.columns = [f"{kind}_{stat}" for stat, kind in ancho.columns]
    return ancho.reset_index()


def coverage_frame(daily: pd.DataFrame, equipos: pd.DataFrame) -> pd.DataFrame:
    """Fracción de lecturas esperadas que realmente llegaron, por equipo y día.

    Sirve para detectar los días en que un equipo estuvo en reparación: sin
    equipo girando no hay lecturas, y ese día no debe entrar al entrenamiento.
    """
    total = daily.groupby(["equipo_code", "fecha"], as_index=False)["n_lecturas"].sum()
    total = total.merge(equipos[["equipo_code", "n_signals"]], on="equipo_code")
    total["cobertura"] = total["n_lecturas"] / (total["n_signals"] * 24.0)
    return total[["equipo_code", "fecha", "cobertura"]]


# --------------------------------------------------------------------------- #
# Contexto de mantenimiento y etiquetas
# --------------------------------------------------------------------------- #
def days_since_last_order(base: pd.DataFrame, orders: pd.DataFrame) -> pd.Series:
    """Días transcurridos desde la última orden de trabajo de ese equipo.

    Solo considera órdenes anteriores o iguales a la fecha evaluada.
    """
    if orders.empty:
        return pd.Series(np.nan, index=base.index)

    by_equipment: dict[str, list[date]] = {}
    for code, grp in orders.groupby("equipo_code"):
        by_equipment[code] = sorted(grp["fecha_orden"].tolist())

    result = []
    for code, fecha in zip(base["equipo_code"], base["fecha"]):
        past = [d for d in by_equipment.get(code, []) if d <= fecha]
        result.append((fecha - past[-1]).days if past else np.nan)
    return pd.Series(result, index=base.index, dtype="float64")


def last_known_route(base: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    """Última medición de ruta disponible para ese equipo a esa fecha.

    La ruta es mensual, así que el valor vigente es el de la última visita del
    técnico, nunca uno futuro.
    """
    empty = pd.DataFrame({
        "zona_iso_ord": np.nan, "rms_ruta": np.nan, "dias_desde_ruta": np.nan,
    }, index=base.index)
    if routes.empty:
        return empty

    by_equipment: dict[str, list[tuple]] = {}
    for code, grp in routes.groupby("equipo_code"):
        rows = sorted(grp.itertuples(index=False), key=lambda r: r.fecha)
        by_equipment[code] = [(r.fecha, r.zona_iso_ord, r.rms_ruta) for r in rows]

    zonas, rms_vals, dias = [], [], []
    for code, fecha in zip(base["equipo_code"], base["fecha"]):
        past = [r for r in by_equipment.get(code, []) if r[0] <= fecha]
        if past:
            f_ruta, zona, rms = past[-1]
            zonas.append(zona)
            rms_vals.append(rms)
            dias.append((fecha - f_ruta).days)
        else:
            zonas.append(np.nan)
            rms_vals.append(np.nan)
            dias.append(np.nan)
    return pd.DataFrame(
        {"zona_iso_ord": zonas, "rms_ruta": rms_vals, "dias_desde_ruta": dias},
        index=base.index,
    )


def build_labels(base: pd.DataFrame, orders: pd.DataFrame,
                 horizon: int = HORIZON_DAYS) -> pd.Series:
    """Etiqueta binaria: hay una orden correctiva en los próximos ``horizon`` días.

    Usa exclusivamente órdenes PM01 (correctivas, es decir el equipo ya falló).
    Las PM02 preventivas y PM03 predictivas se excluyen porque responden a un
    plan o a una alerta, no a una avería.
    """
    correctivas = orders[orders["tipo_orden"] == "PM01"]
    by_equipment: dict[str, list[date]] = {}
    for code, grp in correctivas.groupby("equipo_code"):
        by_equipment[code] = sorted(grp["fecha_orden"].tolist())

    labels = []
    for code, fecha in zip(base["equipo_code"], base["fecha"]):
        limite = fecha + timedelta(days=horizon)
        futuras = by_equipment.get(code, [])
        labels.append(int(any(fecha < d <= limite for d in futuras)))
    return pd.Series(labels, index=base.index, dtype="int64")


# --------------------------------------------------------------------------- #
# Orquestación
# --------------------------------------------------------------------------- #
def build_dataset(data_dir: str) -> pd.DataFrame:
    """Construye la tabla de entrenamiento completa desde los datos bronze."""
    equipos = equipment_frame()
    hist = load_historian(data_dir)
    daily = daily_aggregates(hist)

    features = rolling_features(daily)
    coverage = coverage_frame(daily, equipos)
    base = features.merge(coverage, on=["equipo_code", "fecha"], how="left")

    # Guarda anti-fuga: fuera los días sin operación normal del equipo.
    base = base[base["cobertura"] >= MIN_COVERAGE].reset_index(drop=True)

    orders = load_orders(data_dir)
    routes = load_routes(data_dir)
    produccion = load_production(data_dir)

    base["dias_desde_orden"] = days_since_last_order(base, orders)
    base = pd.concat([base, last_known_route(base, routes)], axis=1)
    base = base.merge(produccion, on="fecha", how="left")
    base = base.merge(
        equipos[["equipo_code", "criticidad", "nombre"]], on="equipo_code", how="left")
    base["falla_14d"] = build_labels(base, orders)

    # Las filas sin ventana completa de 7 días no son utilizables.
    trend_cols = [c for c in base.columns if c.endswith("_d7_mean")]
    base = base.dropna(subset=trend_cols, how="all").reset_index(drop=True)

    ordered = ["fecha", "equipo_code", "nombre", "criticidad", "falla_14d"]
    rest = [c for c in base.columns if c not in ordered]
    return base[ordered + rest].sort_values(["fecha", "equipo_code"]).reset_index(drop=True)


MLTABLE_TEMPLATE = """\
$schema: https://azuremlschemas.azureedge.net/latest/MLTable.schema.json
type: mltable

# Descriptor que acompaña al CSV. Azure Machine Learning solo acepta datos
# tabulares en formato MLTable para los trabajos de AutoML, porque necesita el
# esquema y las instrucciones de lectura junto a los datos.
paths:
  - file: ./training_table.csv

transformations:
  - read_delimited:
      delimiter: ','
      encoding: utf8
      empty_as_string: false
      header: all_files_same_headers
"""


def write_mltable(out_dir: str) -> str:
    """Escribe el descriptor MLTable junto a la tabla de entrenamiento."""
    destino = os.path.join(out_dir, "MLTable")
    with open(destino, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(MLTABLE_TEMPLATE)
    return destino


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construye la tabla de entrenamiento a partir de los datos bronze.")
    parser.add_argument("--data", default="workdir",
                        help="Carpeta con la estructura bronze/ (por defecto workdir).")
    parser.add_argument("--out", default="workdir/gold",
                        help="Carpeta de salida para la tabla de entrenamiento.")
    parser.add_argument("--horizon", type=int, default=HORIZON_DAYS,
                        help="Horizonte de predicción en días.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    df = build_dataset(args.data)

    os.makedirs(args.out, exist_ok=True)
    destino = os.path.join(args.out, "training_table.csv")
    df.to_csv(destino, index=False)
    write_mltable(args.out)

    positivos = int(df["falla_14d"].sum())
    total = len(df)
    print(f"Filas: {total}  ·  equipos: {df['equipo_code'].nunique()}  "
          f"·  rango: {df['fecha'].min()} a {df['fecha'].max()}")
    print(f"Positivos: {positivos} ({100.0 * positivos / total:.2f}%)  ·  "
          f"columnas: {len(df.columns)}")
    print(f"Escrito en: {os.path.abspath(destino)}")
    print("Descriptor MLTable generado junto al CSV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
