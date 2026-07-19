# tissue-mill-mlops

**Mantenimiento predictivo de una planta de papel tissue, con el ciclo MLOps completo sobre Azure Machine Learning.**

[![CI](https://github.com/XxHurtadoxX/tissue-mill-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/XxHurtadoxX/tissue-mill-mlops/actions/workflows/ci.yml)
[![Datos diarios](https://github.com/XxHurtadoxX/tissue-mill-mlops/actions/workflows/daily-data.yml/badge.svg)](https://github.com/XxHurtadoxX/tissue-mill-mlops/actions/workflows/daily-data.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> Proyecto de portafolio de **Daniel Hurtado**. Implementa el ciclo completo de MLOps sobre Azure Machine Learning aplicado a un caso industrial real: anticipar la falla de los equipos rotativos críticos de una máquina de papel tissue, desde la generación de datos hasta el despliegue en producción.

---

## ¿Qué problema resuelve?

En una máquina de papel tissue la producción es una línea continua: si un equipo se detiene, se detiene toda la máquina. Una bomba de vacío que falla un domingo a las 2 a.m. puede costar 18 horas de parada y más de 40 toneladas de producción perdida. Las señales suelen estar disponibles antes del evento (la corriente del motor llevaba dos semanas subiendo, la ruta de vibración del mes ya marcaba la bomba en zona C), pero nadie las cruza porque viven en sistemas distintos y cada turno atiende lo suyo.

Este proyecto predice esas fallas con 7–14 días de anticipación para que la intervención caiga en la parada programada, no en la madrugada.

## El simulador de datos

Antes de llegar al modelo, el proyecto resuelve un problema previo: generar datos de planta que se comporten como datos de planta reales, con su suciedad incluida. No encontré un generador open-source en español que simulara historian, órdenes de mantenimiento y rutas de vibración con ese nivel de detalle, así que lo construí para este proyecto. Cubre cinco fuentes:

| Fuente | Qué simula | Suciedad realista incluida |
|---|---|---|
| **Historian** (DCS) | Lecturas de sensores por hora, con tags ISA-5.1 (`TIS1.45VI4523.PV`) | Lecturas `-9999`/`BAD` por fallo de comunicación · **sensores congelados** (flatline) · huecos por parada |
| **Lotes de pulper** | Duración, kg y kWh de cada lote | Firma del **desgaste del rotor** (kWh/kg sube ciclo a ciclo) |
| **Rutas de vibración** | Medición mensual portátil, RMS + gE + zona ISO 20816 | — |
| **Órdenes SAP PM** | Avisos y órdenes de trabajo | **Texto libre con errores** ("ruido y vibracion en Prensa de succion, revisar urgente"), sin tildes, `fecha_aviso ≠ fecha real de falla` |
| **Ground truth** | El oráculo del simulador: onset y fecha real de cada falla | Separado de bronze a propósito; no se usa para construir features, solo para evaluar |

La generación es determinista: dado el mismo `--seed`, se reproduce exactamente la misma historia de planta. El tiempo entre fallas sigue una distribución Weibull, apropiada para modelar desgaste mecánico, a diferencia de la exponencial (que asume fallas puramente aleatorias y es el supuesto por defecto en simuladores más simples).

## Datos que llegan a diario

Una planta real genera datos todos los días, así que el repositorio también lo hace. El workflow [`daily-data.yml`](.github/workflows/daily-data.yml) corre cada día a las 06:00 UTC, genera el extracto del historian correspondiente a esa fecha y lo commitea automáticamente, igual que correría un job de ingesta nocturno en producción. El dataset queda actualizado sin intervención manual.

## Instalación y uso

```bash
# 1. Clonar e instalar herramientas de desarrollo
git clone https://github.com/XxHurtadoxX/tissue-mill-mlops.git
cd tissue-mill-mlops
pip install -r requirements-dev.txt      # el simulador no requiere dependencias externas

# 2. Generar datos
python -m src.simulator.generate --date 2026-07-19      # un solo día
python -m src.simulator.generate --days 60              # los últimos 60 días
python -m src.simulator.generate --from 2024-01-01 --to 2025-12-31   # backfill histórico

# 3. Correr calidad (lo mismo que la CI)
flake8 src tests
pytest
```

Los datos caen bajo `data/` en el patrón **bronze** (crudo, tal cual llega). Las capas silver (limpio) y gold (features) llegan en la Fase 1.

## Estructura del repositorio

```
tissue-mill-mlops/
├── src/simulator/
│   ├── plant.py        # modelo físico: equipos, tags, cronograma de fallas (Weibull)
│   ├── dirty.py        # inyección de suciedad (-9999, sensores congelados)
│   └── generate.py     # generador + CLI (por día / rango / backfill)
├── tests/              # determinismo, esquema, gramática de tags, deriva de falla
├── data/               # datos versionados (bronze) + ground_truth
├── docs/
│   ├── caso-negocio.md      # el "antes/después", el valor en pesos
│   └── diccionario-datos.md # gramática de tags, esquemas, roster de equipos
├── .github/workflows/
│   ├── ci.yml          # lint + tests en cada PR
│   └── daily-data.yml  # cron: genera y commitea el extracto diario
└── requirements-dev.txt
```

## Roadmap

El simulador (Fase 0) es la base sobre la que se construye el resto del ciclo de MLOps, en fases incrementales:

- [x] **Fase 0 (datos)**: simulador determinista, CI y generación diaria automática *(estás aquí)*
- [ ] **Fase 1**: workspace de Azure ML, cómputo gestionado por CLI, datos registrados como assets versionados
- [ ] **Fase 2**: AutoML + experimentación en notebooks con seguimiento de MLflow
- [ ] **Fase 3**: entrenamiento como job parametrizado + búsqueda de hiperparámetros
- [ ] **Fase 4**: pipeline de componentes reutilizables (preparación → features → entrenamiento → evaluación)
- [ ] **Fase 5**: integración continua con GitHub Actions (service principal, trunk-based development)
- [ ] **Fase 6**: entornos dev/prod con aprobación manual y despliegue blue/green
- [ ] **Fase 7**: monitoreo de data drift en producción

## Nota sobre los datos

Son **100% sintéticos**, generados por el simulador. No provienen de ninguna planta real ni contienen información confidencial. El caso está inspirado en la operación típica de una máquina tissue de fibra reciclada.

## Licencia

Distribuido bajo licencia MIT. Ver [LICENSE](LICENSE).
