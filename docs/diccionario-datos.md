# Diccionario de datos

Referencia técnica de los esquemas que produce el simulador y de la gramática de los tags. Es la base sobre la que se construyen las capas silver y gold.

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
| `silenciosa` | int | 1 si la falla no dejó rastro en los sensores |

La columna `silenciosa` marca las fallas que ocurrieron sin precursor medible. Ningún modelo puede anticiparlas, así que definen el techo de desempeño alcanzable y conviene reportarlas por separado al evaluar.

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

## Qué hace difícil el problema

Un simulador de mantenimiento predictivo es fácil de escribir mal. Si la degradación mueve las señales con fuerza y el resto de la operación es estable, el problema se resuelve con un umbral y el ejercicio pierde sentido. Una versión previa de este simulador tenía justamente ese defecto: una sola variable separaba las clases con AUC de 0.997, cuando en la práctica un modelo bueno de mantenimiento predictivo se mueve entre 0.75 y 0.90.

Cuatro mecanismos mantienen el problema en un rango realista:

| Mecanismo | Qué hace | Por qué importa |
|---|---|---|
| Ganancias de falla moderadas | La degradación mueve la vibración un 50% en el peor momento, no un 260% | Una degradación incipiente se parece a un día cargado |
| Deriva diaria del punto de operación | Cada día el equipo arranca en un punto algo distinto (16% de dispersión en vibración) | El ruido dentro del día se promedia al agregar; la deriva entre días no, así que es la que de verdad limita la detección |
| Fallas silenciosas | Una de cada cuatro fallas no deja rastro en los sensores | Pone un techo natural al recall, como ocurre en una planta real |
| Sensores descalibrados | Episodios en que un instrumento deriva sin que el equipo esté enfermo | Genera falsos positivos legítimos, el caso que narra el documento de caso de negocio |

La calibración se ajustó además para que la degradación avanzada cruce el umbral de zona C de la norma ISO 20816. No es un detalle cosmético: si un equipo a punto de fallar nunca alcanza el nivel que la norma considera alerta, la regla que mantenimiento aplica hoy no se dispara nunca y el proyecto se queda sin línea base con la cual compararse.

Con esta calibración, la mejor variable individual alcanza AUC de 0.85. Un umbral simple sobre esa variable anticipa 31 de los 37 eventos del periodo, mientras que la ruta mensual de vibración, que es la práctica actual, solo alcanza a detectar uno. Esa brecha es el punto de partida del modelado.

## Tabla de entrenamiento (capa gold)

La produce `src/features/build_dataset.py`. Una fila por equipo y por día.

| Grupo de columnas | Ejemplo | Descripción |
|---|---|---|
| Identificación | `fecha`, `equipo_code`, `nombre`, `criticidad` | No se usan como features; sirven para agrupar y particionar |
| Etiqueta | `falla_14d` | 1 si hay una orden correctiva PM01 en los próximos 14 días |
| Nivel diario | `vibration_d1_mean` | Media del día para cada tipo de señal |
| Tendencia semanal | `vibration_d7_mean`, `_d7_std`, `_d7_slope` | Ventana de 7 días que solo mira hacia atrás |
| Cambio relativo | `vibration_ratio_1_7` | Día actual contra su propia semana, detecta aceleraciones |
| Contexto de mantenimiento | `dias_desde_orden` | Días desde la última orden de ese equipo |
| Ruta de vibración | `zona_iso_ord`, `rms_ruta`, `dias_desde_ruta` | Última medición mensual vigente a esa fecha |
| Contexto de operación | `produccion_kg`, `cobertura` | Carga de la planta y fracción de lecturas recibidas |

Los tipos de señal presentes son `current`, `vibration`, `temp`, `vacuum` y `pressure`. Cada equipo tiene solo algunos, así que las columnas de los tipos que no aplican quedan vacías. Es un patrón de nulos estructural, no un problema de calidad, y AutoML lo maneja con imputación automática.

