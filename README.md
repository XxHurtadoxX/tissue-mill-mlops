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

El simulador también está calibrado para que el problema conserve su dificultad. Una de cada cuatro fallas ocurre sin dejar rastro en los sensores, el punto de operación de cada equipo deriva de un día a otro por causas legítimas, y algunos instrumentos se descalibran e imitan una degradación que no existe. Con estos mecanismos la mejor variable individual alcanza AUC de 0.85, un rango creíble para mantenimiento predictivo. El razonamiento completo está en el [diccionario de datos](docs/diccionario-datos.md).

## Datos que llegan a diario

Una planta real genera datos todos los días, así que el repositorio también lo hace. El workflow [`daily-data.yml`](.github/workflows/daily-data.yml) corre cada día a las 06:00 UTC, genera el extracto del historian correspondiente a esa fecha y lo commitea automáticamente, igual que correría un job de ingesta nocturno en producción. El dataset queda actualizado sin intervención manual.

## Instalación y uso

```bash
git clone https://github.com/XxHurtadoxX/tissue-mill-mlops.git
cd tissue-mill-mlops
pip install -r requirements-dev.txt
```

El repositorio ya trae una muestra de datos en `data/`, así que se puede explorar sin generar nada. Para reconstruir el histórico completo y armar la tabla de entrenamiento:

```bash
# Histórico de dos años y medio (unos 45 MB, no versionado)
python -m src.simulator.generate --from 2024-01-01 --to 2026-07-26 --out workdir

# Tabla de entrenamiento: una fila por equipo y por día
python -m src.features.build_dataset --data workdir --out workdir/gold
```

Para comprobar el estado del código, lo mismo que corre la integración continua:

```bash
flake8 src tests
pytest
```

### Dónde viven los datos

| Carpeta | Versionada | Qué contiene |
|---|---|---|
| `data/` | Sí | Muestra viva. El workflow programado escribe ahí el extracto de cada día y conserva una ventana móvil de 120 días |
| `workdir/` | No | Espacio de trabajo local: histórico completo y tabla de entrenamiento. Se regenera con los dos comandos de arriba |

Ambas siguen el patrón medallón: los datos crudos entran como **bronze**, la limpieza produce **silver** y la tabla lista para entrenar es **gold**.

## Estructura del repositorio

```
tissue-mill-mlops/
├── src/
│   ├── simulator/           # genera los datos (solo librería estándar)
│   │   ├── plant.py           equipos, tags ISA-5.1, cronograma de fallas Weibull
│   │   ├── dirty.py           lecturas BAD y sensores congelados
│   │   └── generate.py        CLI: por día, por rango o histórico completo
│   └── features/            # convierte los datos en tabla de entrenamiento
│       ├── clean.py           bronze a silver: descarta lo que no es confiable
│       └── build_dataset.py   gold: ventanas móviles, contexto y etiquetas
├── aml/                     # infraestructura de Azure ML declarada en YAML
│   ├── setup/                 workspace y clúster de cómputo
│   └── data/                  los tres data assets registrados
├── tests/
│   ├── pipeline/              guardas anti-fuga, etiquetas y limpieza
│   └── simulator/             determinismo y firmas físicas del generador
├── docs/
│   ├── caso-negocio.md        el problema en planta y el valor en pesos
│   └── diccionario-datos.md   esquemas, tags y por qué el problema es difícil
├── data/                    # muestra viva versionada (ver tabla de arriba)
└── .github/workflows/
    ├── ci.yml                 lint y pruebas en cada cambio
    └── daily-data.yml         extracto diario programado
```

El simulador no depende de librerías externas, de modo que el workflow diario corre sin instalar nada. Las dependencias de `requirements.txt` (pandas y numpy) solo hacen falta para construir la tabla de entrenamiento.

## Roadmap

El simulador (Fase 0) es la base sobre la que se construye el resto del ciclo de MLOps, en fases incrementales:

- [x] **Fase 0 (datos)**: simulador determinista, CI y generación diaria automática
- [x] **Fase 1 (infraestructura)**: workspace y cómputo declarados en YAML, tabla de entrenamiento y tres data assets versionados *(estás aquí)*
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
