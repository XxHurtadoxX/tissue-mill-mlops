# Diccionario de datos

Referencia técnica de todos los esquemas que produce el simulador y de la gramática de los tags. Base para la Fase 1 (capas silver/gold).

## Gramática de tags (ISA-5.1)

```
TIS1.20II0205.PV
 |    | | |   |
 |    | | |   +-- .PV = Process Value (valor medido)
 |    | | +------ 0205 = número de lazo (pulper 02, instrumento 05)
 |    | +-------- II   = tipo de señal
 |    +---------- 20   = área
 +--------------- TIS1 = máquina tissue 1
```

**Tipos de señal (prefijo de instrumento):**

| Prefijo | Señal | Unidad |
|---|---|---|
| `II` | Corriente de motor | A |
| `TI` | Temperatura de soporte/rodamiento | °C |
| `VI` | Vibración RMS | mm/s |
| `PI` | Vacío o presión | kPa |

**Áreas:**

| Área | Zona de planta |
|---|---|
| `20` | Preparación de pasta (pulpers, agitadores) |
| `33` | Aproximación (fan pump) |
| `45` | Vacío y prensa |
| `60` | Conversión (bobinadora) |

## Esquemas

### `bronze/historian/<fecha>.csv`
Lecturas de sensores, una fila por (tag, timestamp).

| Columna | Tipo | Notas |
|---|---|---|
| `timestamp` | datetime | resolución por defecto: 60 min |
| `tag` | str | ver gramática ISA-5.1 |
| `value` | float | `-9999` cuando `quality = BAD` |
| `quality` | str | `GOOD` o `BAD` |

### `bronze/produccion/batches_<fecha>.csv`
Lotes de pulper. La firma del desgaste del rotor vive aquí (kWh/kg sube con el desgaste).

| Columna | Tipo | Notas |
|---|---|---|
| `lote` | str | id `L-AAAAMMDD-<pulper>-NN` |
| `pulper` | str | `P1` / `P2` |
| `fecha` | date | |
| `inicio` | time | hora de inicio del lote |
| `duracion_min` | float | sube con el desgaste del rotor |
| `kg_cargados` | int | |
| `kwh` | float | energía del lote |
| `material` | str | `archivo_blanco` / `periodico` / `recorte_mixto` |

### `bronze/rutas/ruta_<aaaa-mm>.csv`
Ruta de vibración portátil mensual (día 3 del mes).

| Columna | Tipo | Notas |
|---|---|---|
| `fecha` | date | |
| `punto` | str | punto de medición descrito |
| `rms_mm_s` | float | vibración global |
| `gE` | float | envolvente de aceleración (rodamientos) |
| `zona_iso` | str | A / B / C / D (ISO 20816 simplificada) |

Umbrales de zona (simplificados): A < 2.8 · B < 4.5 · C < 7.1 · D ≥ 7.1 mm/s.

### `bronze/sap/ordenes.csv`
Órdenes de trabajo (SAP PM). **Ledger que crece**: el generador deduplica por `aviso`.

| Columna | Tipo | Notas |
|---|---|---|
| `aviso` | int | número de aviso |
| `tipo_aviso` | str | `M1` solicitud / `M2` avería |
| `orden` | int | número de orden |
| `tipo_orden` | str | `PM01` correctiva / `PM02` preventiva / `PM03` predictiva |
| `ubicacion_tecnica` | str | jerarquía SAP (tabla IFLOT) |
| `equipo` | str | id de equipo (tabla EQUI) |
| `fecha_aviso` | datetime | **≠ fecha real de la falla** (llega horas/días después) |
| `texto` | str | **texto libre con errores de ortografía** |
| `horas` | float | horas de intervención |
| `repuesto` | str | repuesto consumido |

### `ground_truth/eventos.csv` ⚠️
El oráculo del simulador. Contiene la verdad que en una planta real no se conoce con precisión: el `onset` real de cada degradación. No debe usarse para construir features; sirve únicamente para evaluar qué tan bien el modelo anticipó las fallas.

| Columna | Tipo | Notas |
|---|---|---|
| `evento_id` | str | id único del evento |
| `code` | str | código corto del equipo |
| `equipo` | str | id de SAP |
| `nombre` | str | nombre legible |
| `modo_falla` | str | modo de falla |
| `fecha_onset` | date | inicio REAL de la degradación |
| `fecha_falla` | date | fecha de la falla dura |
| `fuente` | str | `historian` o `batches` |

## Roster de equipos

12 equipos de la máquina TIS1. Criticidad **A** = su falla detiene la máquina; **B** = solo degrada.

| Código | Nombre | Área | Criticidad | MTBF (días) |
|---|---|---|---|---|
| P1 | Motor pulper 1 | 20 | B | 520 |
| P2 | Motor pulper 2 | 20 | B | 500 |
| AG1–AG4 | Agitadores de tina | 20 | B | 400 |
| BV1 | Bomba de vacío 1 | 45 | A | 170 |
| BV2 | Bomba de vacío 2 | 45 | A | 185 |
| BV3 | Bomba de vacío 3 | 45 | A | 200 |
| FP1 | Fan pump | 33 | A | 210 |
| PRE | Prensa de succión | 45 | A | 240 |
| RED1 | Reductor bobinadora | 60 | A | 300 |

El rotor de cada pulper (P1, P2) tiene además su propio ciclo de vida de desgaste (~205 días), cuya firma aparece en los lotes, no en el historian.
