"""Modelo físico de la planta tissue: equipos, tags y firmas de falla.

Todo es DETERMINISTA dado un ``seed``: el mismo seed reproduce exactamente la
misma historia de la planta (degradaciones, fallas, órdenes de trabajo). Gracias
a eso la generación diaria incremental (el cron de GitHub Actions) y el backfill
histórico comparten una única línea de tiempo física y coherente, sin necesidad
de guardar estado entre corridas.

Convención de tags (ISA-5.1), heredada del caso papelera::

    TIS1.20II0205.PV
     |    | | |   |
     |    | | |   +-- .PV = Process Value (valor medido)
     |    | | +------ 0205 = número de lazo (pulper 02, instrumento 05)
     |    | +-------- II   = tipo de señal (I de corriente)
     |    +---------- 20   = área (20 = preparación de pasta)
     +--------------- TIS1 = máquina tissue 1
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import date, timedelta

# Época de nacimiento de la planta: ancla de todas las tendencias temporales.
PLANT_EPOCH = date(2024, 1, 1)
MACHINE = "TIS1"

# ISA-5.1: tipo de señal -> prefijo de instrumento en el tag.
SIGNAL_CODE = {
    "current": "II",     # corriente de motor (A)
    "temp": "TI",        # temperatura de soporte/rodamiento (°C)
    "vibration": "VI",   # vibración RMS (mm/s)
    "vacuum": "PI",      # vacío entregado (kPa)
    "pressure": "PI",    # presión de descarga (kPa)
}

# Cuánto mueve una falla activa a cada tipo de señal (ganancia relativa en p=1).
# Positiva = sube (vibración, temperatura, corriente); negativa = cae (el vacío
# de una bomba desgastada baja). Hace la firma de falla MULTIVARIADA y realista.
FAIL_GAIN = {
    "vibration": 2.6,
    "temp": 0.55,
    "current": 0.12,
    "vacuum": -0.18,
    "pressure": -0.12,
}


@dataclass(frozen=True)
class Signal:
    """Una señal (tag) que emite un equipo."""

    kind: str          # clave de SIGNAL_CODE
    loop: str          # número de lazo de 4 dígitos, p.ej. "0205"
    base: float        # valor nominal en estado sano
    noise: float       # desviación estándar del ruido gaussiano
    day_amp: float = 0.0  # amplitud del ciclo diario de carga (día vs noche)

    @property
    def gain(self) -> float:
        return FAIL_GAIN.get(self.kind, 0.0)


@dataclass(frozen=True)
class Equipment:
    """Un equipo rotativo con su identidad de planta y de SAP."""

    code: str          # código corto de planta, p.ej. "BV2"
    eq_id: str         # id de SAP (tabla EQUI), p.ej. "10004512"
    name: str          # nombre legible
    area: str          # área de 2 dígitos para el tag
    iflot: str         # ubicación técnica de SAP (tabla IFLOT)
    signals: tuple[Signal, ...]
    mtbf_days: float   # tiempo medio entre fallas
    onset_days: int    # días de degradación antes de la falla dura
    repair_days: int   # días de parada por reparación tras la falla
    criticality: str   # "A" = detiene la máquina · "B" = solo degrada
    route_point: str = ""  # nombre del punto en la ruta de vibración mensual

    def tag(self, sig: Signal) -> str:
        return f"{MACHINE}.{self.area}{SIGNAL_CODE[sig.kind]}{sig.loop}.PV"

    def has_kind(self, kind: str) -> bool:
        return any(s.kind == kind for s in self.signals)


@dataclass(frozen=True)
class Failure:
    """Un evento de falla: ventana de degradación + fecha de falla dura."""

    code: str          # código del equipo afectado
    onset: date        # inicio real de la degradación
    fail: date         # fecha de la falla dura (para/orden de trabajo)
    kind: str          # descripción legible del modo de falla


def _seed_int(seed: int, *parts: object) -> int:
    """Deriva un entero estable (multiplataforma) de un seed y unas partes."""
    raw = "|".join([str(seed)] + [str(p) for p in parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def rng_for(seed: int, *parts: object) -> random.Random:
    """random.Random sembrado de forma determinista y estable."""
    return random.Random(_seed_int(seed, *parts))


# --------------------------------------------------------------------------- #
# Roster de equipos de la planta tissue (máquina TIS1)
# --------------------------------------------------------------------------- #
def _agitator(n: int, eq_id: str, cur_base: float) -> Equipment:
    loop_c = f"03{n}0"
    loop_v = f"03{n}1"
    return Equipment(
        code=f"AG{n}",
        eq_id=eq_id,
        name=f"Agitador tina {n}",
        area="20",
        iflot=f"PLBQ-TIS1-PREP-AGIT0{n}",
        signals=(
            Signal("current", loop_c, cur_base, 1.1, day_amp=2.0),
            Signal("vibration", loop_v, 2.3 + 0.1 * n, 0.18),
        ),
        mtbf_days=400, onset_days=30, repair_days=1, criticality="B",
        route_point=f"Agitador tina {n} · lado transmision · horizontal",
    )


def build_plant() -> tuple[Equipment, ...]:
    """Devuelve el roster completo de equipos de la planta."""
    return (
        # --- Preparación de pasta (área 20) ---
        Equipment(
            code="P1", eq_id="10004511", name="Motor pulper 1", area="20",
            iflot="PLBQ-TIS1-PREP-PULP01",
            signals=(
                Signal("current", "0105", 280.0, 6.0, day_amp=10.0),
                Signal("vibration", "0106", 2.4, 0.2),
            ),
            mtbf_days=520, onset_days=40, repair_days=2, criticality="B",
            route_point="Motor pulper 1 · lado libre · axial",
        ),
        Equipment(
            code="P2", eq_id="10004512", name="Motor pulper 2", area="20",
            iflot="PLBQ-TIS1-PREP-PULP02",
            signals=(
                Signal("current", "0205", 285.0, 6.0, day_amp=10.0),
                Signal("vibration", "0206", 2.5, 0.2),
            ),
            mtbf_days=500, onset_days=40, repair_days=2, criticality="B",
            route_point="Motor pulper 2 · lado libre · axial",
        ),
        _agitator(1, "10004521", 45.0),
        _agitator(2, "10004522", 46.0),
        _agitator(3, "10004523", 44.0),
        _agitator(4, "10004524", 47.0),
        # --- Vacío y prensa (área 45) ---
        Equipment(
            code="BV1", eq_id="10004531", name="Bomba de vacio 1", area="45",
            iflot="PLBQ-TIS1-VACIO-BOMB01",
            signals=(
                Signal("current", "4510", 175.0, 4.0, day_amp=6.0),
                Signal("vacuum", "4511", 60.0, 1.2),
                Signal("temp", "4512", 62.0, 1.5),
                Signal("vibration", "4513", 2.5, 0.2),
            ),
            mtbf_days=170, onset_days=28, repair_days=3, criticality="A",
            route_point="Bomba de vacio 1 · lado acople · horizontal",
        ),
        Equipment(
            code="BV2", eq_id="10004532", name="Bomba de vacio 2", area="45",
            iflot="PLBQ-TIS1-VACIO-BOMB02",
            signals=(
                Signal("current", "4520", 178.0, 4.0, day_amp=6.0),
                Signal("vacuum", "4521", 61.0, 1.2),
                Signal("temp", "4522", 63.0, 1.5),
                Signal("vibration", "4523", 2.6, 0.2),
            ),
            mtbf_days=185, onset_days=28, repair_days=3, criticality="A",
            route_point="Bomba de vacio 2 · lado acople · horizontal",
        ),
        Equipment(
            code="BV3", eq_id="10004533", name="Bomba de vacio 3", area="45",
            iflot="PLBQ-TIS1-VACIO-BOMB03",
            signals=(
                Signal("current", "4530", 176.0, 4.0, day_amp=6.0),
                Signal("vacuum", "4531", 59.0, 1.2),
                Signal("temp", "4532", 61.0, 1.5),
                Signal("vibration", "4533", 2.4, 0.2),
            ),
            mtbf_days=200, onset_days=30, repair_days=3, criticality="A",
            route_point="Bomba de vacio 3 · lado acople · horizontal",
        ),
        Equipment(
            code="FP1", eq_id="10004540", name="Fan pump", area="33",
            iflot="PLBQ-TIS1-APROX-FANPUMP",
            signals=(
                Signal("current", "3310", 520.0, 9.0, day_amp=14.0),
                Signal("pressure", "3312", 340.0, 5.0),
                Signal("vibration", "3311", 2.8, 0.22),
            ),
            mtbf_days=210, onset_days=35, repair_days=3, criticality="A",
            route_point="Fan pump · lado acople · horizontal",
        ),
        Equipment(
            code="PRE", eq_id="10004545", name="Prensa de succion", area="45",
            iflot="PLBQ-TIS1-PRENSA-SUCC",
            signals=(
                Signal("vibration", "4560", 3.0, 0.25),
                Signal("temp", "4561", 58.0, 1.6),
                Signal("vacuum", "4562", 55.0, 1.3),
            ),
            mtbf_days=240, onset_days=35, repair_days=3, criticality="A",
            route_point="Prensa de succion · muñon transmision · horizontal",
        ),
        # --- Conversión (área 60) ---
        Equipment(
            code="RED1", eq_id="10004560", name="Reductor bobinadora",
            area="60", iflot="PLBQ-TIS1-CONV-BOBIN-RED",
            signals=(
                Signal("vibration", "6010", 2.2, 0.2),
                Signal("temp", "6011", 65.0, 1.8),
            ),
            mtbf_days=300, onset_days=45, repair_days=4, criticality="A",
            route_point="Reductor bobinadora · engrane salida · horizontal",
        ),
    )


# --------------------------------------------------------------------------- #
# Cronograma de fallas (determinista por equipo + seed)
# --------------------------------------------------------------------------- #
_FAIL_KINDS = {
    "vibration": "Desgaste de rodamiento",
    "temp": "Sobrecalentamiento por lubricacion",
    "vacuum": "Perdida de vacio por sellos",
}


def _fail_kind(eq: Equipment) -> str:
    if eq.has_kind("vibration"):
        return _FAIL_KINDS["vibration"]
    if eq.has_kind("temp"):
        return _FAIL_KINDS["temp"]
    return "Falla mecanica"


# Forma (k) de la Weibull para el tiempo entre fallas. k>1 modela desgaste
# (wear-out): las fallas se agrupan alrededor de la vida característica, a
# diferencia de la exponencial (k=1), que es para fallas aleatorias. El factor
# corrige la escala para que la media de la Weibull sea ~mtbf_days.
_WEIBULL_K = 2.2
_WEIBULL_MEAN_FACTOR = 0.886  # Gamma(1 + 1/k) para k=2.2


def failure_schedule(eq: Equipment, seed: int, horizon: date) -> list[Failure]:
    """Fallas del equipo desde la época de la planta hasta ``horizon``.

    Los intervalos entre fallas siguen una Weibull con media ~``mtbf_days`` y
    forma de desgaste; el resultado es estable dado ``seed`` y el código del
    equipo, así el backfill y el cron diario comparten la misma línea de tiempo.
    """
    rng = rng_for(seed, "failsched", eq.code)
    scale = eq.mtbf_days / _WEIBULL_MEAN_FACTOR
    out: list[Failure] = []
    t = PLANT_EPOCH
    limit = horizon + timedelta(days=30)
    while True:
        gap = max(eq.onset_days + 10.0, rng.weibullvariate(scale, _WEIBULL_K))
        t = t + timedelta(days=gap)
        if t > limit:
            break
        onset = t - timedelta(days=eq.onset_days)
        out.append(Failure(eq.code, onset, t, _fail_kind(eq)))
    return out


def progress(f: Failure, day: date) -> float | None:
    """Avance de la degradación en [0, 1] si el día cae en la ventana, si no None."""
    if day < f.onset or day > f.fail:
        return None
    span = (f.fail - f.onset).days or 1
    return (day - f.onset).days / span


def active_failure(sched: list[Failure], day: date) -> Failure | None:
    for f in sched:
        if f.onset <= day <= f.fail:
            return f
    return None


def is_under_repair(eq: Equipment, sched: list[Failure], day: date) -> bool:
    """True si el equipo está en reparación (parada) ese día."""
    for f in sched:
        if f.fail < day <= f.fail + timedelta(days=eq.repair_days):
            return True
    return False
