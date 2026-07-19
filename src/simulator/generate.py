"""Generador de datos de la planta tissue (capa bronze).

Uso típico::

    # Un día concreto (lo que corre el cron a diario):
    python -m src.simulator.generate --date 2026-07-19

    # Backfill de los últimos 60 días (para tener muestra en el repo):
    python -m src.simulator.generate --days 60

    # Un rango explícito:
    python -m src.simulator.generate --from 2026-01-01 --to 2026-03-31

Salidas (bajo ``--out``, por defecto ``data/``)::

    bronze/historian/<fecha>.csv        lecturas de sensores (tag, valor, calidad)
    bronze/produccion/batches_<fecha>.csv  lotes de pulper (duracion, kwh, ...)
    bronze/rutas/ruta_<aaaa-mm>.csv     ruta de vibracion mensual (dia 3)
    bronze/sap/ordenes.csv              ordenes de trabajo (texto libre sucio)
    ground_truth/eventos.csv            ORACULO del simulador (no usar en features)

Todo es determinista dado ``--seed``: volver a generar la misma fecha produce
exactamente el mismo dato. Por eso la generación diaria del cron es idempotente
y no necesita estado previo.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from datetime import date, datetime, timedelta, timezone

from . import dirty as dirty_mod
from . import plant as plant_mod

DEFAULT_SEED = 42
DEFAULT_SAMPLE_MIN = 60
ROUTE_DAY = 3  # día del mes en que se toma la ruta de vibración

# Energía específica de desfibrado por material (kWh/kg) y factor de duración.
MATERIALS = {
    "archivo_blanco": (0.068, 1.00),
    "periodico": (0.075, 1.12),
    "recorte_mixto": (0.072, 1.06),
}
PULPERS = ("P1", "P2")


# --------------------------------------------------------------------------- #
# Utilidades de escritura
# --------------------------------------------------------------------------- #
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_csv(path: str, header: list[str], rows: list[list]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _append_csv(path: str, header: list[str], rows: list[list],
                key_idx: int) -> int:
    """Añade filas nuevas a un CSV-ledger, deduplicando por la columna key_idx.

    Devuelve cuántas filas nuevas se escribieron. Mantiene el archivo como una
    tabla que crece (como SAP), pero idempotente: reejecutar un día no duplica.
    """
    _ensure_dir(os.path.dirname(path))
    seen: set[str] = set()
    exists = os.path.exists(path)
    if exists:
        with open(path, newline="", encoding="utf-8") as fh:
            r = csv.reader(fh)
            next(r, None)
            for row in r:
                if row:
                    seen.add(row[key_idx])
    new = [row for row in rows if str(row[key_idx]) not in seen]
    if not new:
        return 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if not exists:
            w.writerow(header)
        w.writerows(new)
    return len(new)


# --------------------------------------------------------------------------- #
# Modelo de valor de señal
# --------------------------------------------------------------------------- #
def signal_value(eq: plant_mod.Equipment, sig: plant_mod.Signal,
                 ts: datetime, fail: plant_mod.Failure | None,
                 seed: int) -> float:
    """Valor de una señal en un instante: base + ciclo diario + ruido + falla."""
    hour = ts.hour + ts.minute / 60.0
    day_cycle = math.sin((hour - 6.0) / 24.0 * 2.0 * math.pi)
    value = sig.base + sig.day_amp * day_cycle
    r = plant_mod.rng_for(seed, "noise", eq.code, sig.loop, ts.isoformat())
    value += r.gauss(0.0, sig.noise)
    if fail is not None:
        p = plant_mod.progress(fail, ts.date())
        if p is not None:
            value *= (1.0 + sig.gain * p * p)
    return value


def _planned_stop_hours(day: date, seed: int) -> set[int]:
    """Bloque de parada programada de toda la máquina (~1 vez al mes)."""
    r = plant_mod.rng_for(seed, "stop", day.year, day.month)
    stop_day = r.randint(8, 24)
    if day.day != stop_day:
        return set()
    start = r.randint(1, 16)
    length = r.randint(4, 7)
    return set(range(start, start + length))


# --------------------------------------------------------------------------- #
# Historian
# --------------------------------------------------------------------------- #
def historian_rows(day: date, plant, scheds: dict, seed: int,
                   sample_min: int, dirty: bool) -> list[list]:
    n = 24 * 60 // sample_min
    slots = [datetime(day.year, day.month, day.day) + timedelta(minutes=i * sample_min)
             for i in range(n)]
    stop_hours = _planned_stop_hours(day, seed)
    rows: list[list] = []
    for eq in plant:
        sched = scheds[eq.code]
        if plant_mod.is_under_repair(eq, sched, day):
            continue  # equipo en reparación -> hueco en el historian
        fail = plant_mod.active_failure(sched, day)
        for sig in eq.signals:
            series, keep = [], []
            for ts in slots:
                if ts.hour in stop_hours:
                    continue  # parada programada de la máquina -> hueco
                series.append(signal_value(eq, sig, ts, fail, seed))
                keep.append(ts)
            day_iso = day.isoformat()
            corrupted = dirty_mod.corrupt_series(series, seed, eq.tag(sig),
                                                 day_iso, dirty)
            for ts, (val, qual) in zip(keep, corrupted):
                stamp = ts.strftime("%Y-%m-%d %H:%M:%S")
                out_val = round(val, 2) if qual != "BAD" else int(val)
                rows.append([stamp, eq.tag(sig), out_val, qual])
    return rows


# --------------------------------------------------------------------------- #
# Lotes de pulper (firma del desgaste del rotor -> proyecto P2)
# --------------------------------------------------------------------------- #
def _rotor_cycles(code: str, seed: int, horizon: date):
    """Ciclos de vida del rotor: (instalado, enviado_a_taller). Deterministas."""
    rng = plant_mod.rng_for(seed, "rotor", code)
    cycles, t = [], plant_mod.PLANT_EPOCH
    while t <= horizon + timedelta(days=30):
        life = max(120.0, rng.gauss(205.0, 25.0))
        shop = t + timedelta(days=life)
        cycles.append((t, shop))
        t = shop + timedelta(days=2)  # 2 días de cambio de rotor
    return cycles


def _rotor_wear(code: str, seed: int, day: date) -> float:
    """Desgaste del rotor en [0, 1]; se acelera en el último tercio del ciclo."""
    for install, shop in _rotor_cycles(code, seed, day):
        if install <= day <= shop:
            frac = (day - install).days / max(1, (shop - install).days)
            return round(min(1.0, frac ** 1.6), 4)
    return 0.0


def batch_rows(day: date, seed: int) -> list[list]:
    rows: list[list] = []
    for code in PULPERS:
        wear = _rotor_wear(code, seed, day)
        rng = plant_mod.rng_for(seed, "batch", code, day.isoformat())
        n_batches = rng.randint(16, 20)
        clock = datetime(day.year, day.month, day.day, 6, 0)
        for i in range(n_batches):
            material = rng.choice(list(MATERIALS))
            spec, dur_factor = MATERIALS[material]
            kg = round(rng.gauss(5800, 180), 0)
            dur = rng.gauss(32.0, 1.3) * dur_factor * (1.0 + 0.5 * wear)
            dur = round(dur, 1)
            kwh = round(kg * spec * (1.0 + 0.18 * wear) * rng.gauss(1.0, 0.02), 1)
            lote = f"L-{day.strftime('%Y%m%d')}-{code}-{i + 1:02d}"
            rows.append([lote, code, day.isoformat(),
                         clock.strftime("%H:%M"), dur, int(kg), kwh, material])
            clock += timedelta(minutes=dur + rng.randint(6, 12))
    return rows


# --------------------------------------------------------------------------- #
# Ruta de vibración mensual + zona ISO 20816 (simplificada)
# --------------------------------------------------------------------------- #
def _iso_zone(rms: float) -> str:
    if rms < 2.8:
        return "A"
    if rms < 4.5:
        return "B"
    if rms < 7.1:
        return "C"
    return "D"


def route_rows(day: date, plant, scheds: dict, seed: int) -> list[list]:
    rows: list[list] = []
    for eq in plant:
        if not eq.has_kind("vibration") or not eq.route_point:
            continue
        vib = next(s for s in eq.signals if s.kind == "vibration")
        fail = plant_mod.active_failure(scheds[eq.code], day)
        rms = vib.base
        if fail is not None:
            p = plant_mod.progress(fail, day) or 0.0
            rms *= (1.0 + vib.gain * p * p)
        r = plant_mod.rng_for(seed, "route", eq.code, day.isoformat())
        rms = round(rms + r.gauss(0.0, 0.15), 2)
        ge = round(0.6 + max(0.0, (rms - vib.base)) * 0.9 + r.gauss(0.0, 0.1), 2)
        rows.append([day.isoformat(), eq.route_point, rms, ge, _iso_zone(rms)])
    return rows


# --------------------------------------------------------------------------- #
# Órdenes de trabajo (SAP PM) + oráculo de ground truth
# --------------------------------------------------------------------------- #
_AVISO_TEXT = {
    "Desgaste de rodamiento": [
        "ruido y vibracion en {name}, revisar urgente",
        "{code} vibra raro se calienta el rodamiento",
        "se escucha golpeteo en {name} lado acople",
    ],
    "Sobrecalentamiento por lubricacion": [
        "{name} se calienta demaciado revisar lubricacion",
        "temperatura alta en {code}, bota aceite",
    ],
    "Perdida de vacio por sellos": [
        "cae el vacio en {name}, revisar sellos",
        "{code} no levanta vacio, urge revision",
    ],
    "Falla mecanica": [
        "para de emergencia por falla mecanica en {name}",
        "{code} se detuvo, revisar",
    ],
}
_REPUESTO = {
    "Desgaste de rodamiento": "rodamiento SKF 22320",
    "Sobrecalentamiento por lubricacion": "kit sellos + aceite ISO VG220",
    "Perdida de vacio por sellos": "sello mecanico bomba vacio",
    "Falla mecanica": "varios",
}


def _events_on(day: date, plant, scheds: dict, seed: int):
    """Fallas de historian cuyo día de falla dura cae en ``day``."""
    for eq in plant:
        for f in scheds[eq.code]:
            if f.fail == day:
                yield eq, f


def order_and_truth_rows(day: date, plant, scheds: dict, seed: int):
    """Devuelve (filas_ordenes, filas_ground_truth) para el día."""
    orders, truth = [], []
    for eq, f in _events_on(day, plant, scheds, seed):
        r = plant_mod.rng_for(seed, "order", eq.code, f.fail.isoformat())
        aviso = 10_000_000 + _seed_small(seed, eq.code, f.fail)
        orden = 40_000_000 + _seed_small(seed, eq.code, f.fail, "wo")
        texto = r.choice(_AVISO_TEXT[f.kind]).format(name=eq.name, code=eq.code)
        # El aviso llega horas DESPUÉS del inicio real: fecha_aviso != fecha real.
        fecha_aviso = (datetime(f.fail.year, f.fail.month, f.fail.day)
                       + timedelta(hours=r.randint(2, 20)))
        horas = round(r.uniform(3.0, 14.0), 1)
        orders.append([aviso, "M2", orden, "PM01", eq.iflot, eq.eq_id,
                       fecha_aviso.strftime("%Y-%m-%d %H:%M"), texto,
                       horas, _REPUESTO[f.kind]])
        truth.append([f"{eq.code}-{f.fail.isoformat()}", eq.code, eq.eq_id,
                      eq.name, f.kind, f.onset.isoformat(), f.fail.isoformat(),
                      "historian"])
    # Eventos de taller de rotor (proyecto P2)
    for code in PULPERS:
        for _, shop in _rotor_cycles(code, seed, day):
            if shop == day:
                eq = next(e for e in plant if e.code == code)
                aviso = 10_000_000 + _seed_small(seed, code, shop, "rotor")
                orden = 40_000_000 + _seed_small(seed, code, shop, "rotorwo")
                orders.append([
                    aviso, "M1", orden, "PM03", eq.iflot, eq.eq_id,
                    shop.strftime("%Y-%m-%d %H:%M"),
                    f"envio de rotor {code} a taller por desgaste, lotes lentos",
                    round(24.0, 1), "recuperacion de alabes + balanceo"])
                truth.append([f"{code}-rotor-{shop.isoformat()}", code, eq.eq_id,
                              eq.name, "Desgaste de rotor de pulper",
                              (shop - timedelta(days=60)).isoformat(),
                              shop.isoformat(), "batches"])
    return orders, truth


def _seed_small(seed: int, *parts: object) -> int:
    return plant_mod._seed_int(seed, *parts) % 9_000_000


# --------------------------------------------------------------------------- #
# Orquestación por día / rango
# --------------------------------------------------------------------------- #
HIST_HEADER = ["timestamp", "tag", "value", "quality"]
BATCH_HEADER = ["lote", "pulper", "fecha", "inicio", "duracion_min",
                "kg_cargados", "kwh", "material"]
ROUTE_HEADER = ["fecha", "punto", "rms_mm_s", "gE", "zona_iso"]
ORDER_HEADER = ["aviso", "tipo_aviso", "orden", "tipo_orden", "ubicacion_tecnica",
                "equipo", "fecha_aviso", "texto", "horas", "repuesto"]
TRUTH_HEADER = ["evento_id", "code", "equipo", "nombre", "modo_falla",
                "fecha_onset", "fecha_falla", "fuente"]


def generate_day(day: date, out: str, seed: int, sample_min: int,
                 dirty: bool) -> dict:
    """Genera todos los artefactos bronze de un día. Devuelve un resumen."""
    plant = plant_mod.build_plant()
    scheds = {eq.code: plant_mod.failure_schedule(eq, seed, day) for eq in plant}

    hist = historian_rows(day, plant, scheds, seed, sample_min, dirty)
    _write_csv(os.path.join(out, "bronze", "historian", f"{day.isoformat()}.csv"),
               HIST_HEADER, hist)

    batches = batch_rows(day, seed)
    _write_csv(os.path.join(out, "bronze", "produccion",
                            f"batches_{day.isoformat()}.csv"),
               BATCH_HEADER, batches)

    n_route = 0
    if day.day == ROUTE_DAY:
        route = route_rows(day, plant, scheds, seed)
        _write_csv(os.path.join(out, "bronze", "rutas",
                                f"ruta_{day.strftime('%Y-%m')}.csv"),
                   ROUTE_HEADER, route)
        n_route = len(route)

    orders, truth = order_and_truth_rows(day, plant, scheds, seed)
    n_ord = _append_csv(os.path.join(out, "bronze", "sap", "ordenes.csv"),
                        ORDER_HEADER, orders, key_idx=0)
    n_tru = _append_csv(os.path.join(out, "ground_truth", "eventos.csv"),
                        TRUTH_HEADER, truth, key_idx=0)
    return {"date": day.isoformat(), "historian": len(hist),
            "batches": len(batches), "route": n_route,
            "orders": n_ord, "truth": n_tru}


def generate_range(start: date, end: date, out: str, seed: int,
                   sample_min: int, dirty: bool) -> list[dict]:
    summaries, day = [], start
    while day <= end:
        summaries.append(generate_day(day, out, seed, sample_min, dirty))
        day += timedelta(days=1)
    return summaries


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Simulador de datos de planta tissue (capa bronze).")
    p.add_argument("--date", type=_parse_date,
                   help="Genera un solo día (por defecto: hoy en UTC).")
    p.add_argument("--days", type=int,
                   help="Genera los últimos N días terminando en --date/hoy.")
    p.add_argument("--from", dest="dfrom", type=_parse_date,
                   help="Inicio de rango (con --to).")
    p.add_argument("--to", dest="dto", type=_parse_date,
                   help="Fin de rango (con --from).")
    p.add_argument("--out", default="data", help="Carpeta de salida.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--sample-min", type=int, default=DEFAULT_SAMPLE_MIN,
                   help="Resolución del historian en minutos.")
    p.add_argument("--clean", action="store_true",
                   help="Genera datos SIN suciedad (para referencia/tests).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dirty = not args.clean
    today = datetime.now(timezone.utc).date()

    if args.dfrom and args.dto:
        summaries = generate_range(args.dfrom, args.dto, args.out, args.seed,
                                   args.sample_min, dirty)
    elif args.days:
        end = args.date or today
        start = end - timedelta(days=args.days - 1)
        summaries = generate_range(start, end, args.out, args.seed,
                                   args.sample_min, dirty)
    else:
        day = args.date or today
        summaries = [generate_day(day, args.out, args.seed, args.sample_min,
                                  dirty)]

    tot_h = sum(s["historian"] for s in summaries)
    tot_o = sum(s["orders"] for s in summaries)
    tot_t = sum(s["truth"] for s in summaries)
    print(f"Generados {len(summaries)} día(s) -> {tot_h} lecturas historian, "
          f"{tot_o} orden(es) SAP, {tot_t} evento(s) de ground truth.")
    print(f"Salida en: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
