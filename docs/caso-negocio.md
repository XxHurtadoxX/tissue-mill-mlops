# Caso de negocio: falla de rotativos críticos

## Cómo se vive el problema hoy (antes del proyecto)

Un domingo a las 2 de la mañana la bomba de vacío 2 empieza a sonar distinto. El operador lo nota pero no es su especialidad; anota en la bitácora *"ruido raro BV2"* y sigue, porque la máquina está corriendo. El martes la bomba se frena en seco: el rodamiento se soltó y rayó el eje. La máquina para. Se descubre que el repuesto del eje no está en bodega, se improvisa. Dieciocho horas de máquina parada después, la planta perdió más de 40 toneladas.

En la reunión del miércoles alguien pregunta lo de siempre: ¿nadie vio venir esto? La respuesta es incómoda, porque las señales sí estaban: la corriente venía subiendo desde hacía dos semanas y la ruta de vibración del mes anterior ya tenía la bomba en zona C. Nadie cruzó esos dos datos, porque viven en sistemas distintos y cada turno atiende lo suyo.

## Objetivo

Anticipar con **7 a 14 días** las fallas de los equipos rotativos cuya parada detiene la máquina (bombas de vacío, fan pump, prensa de succión, reductores) para que la intervención caiga en la parada programada de la semana, no en la madrugada del domingo.

## Por qué el modelo es un clasificador de ventana (y no una red neuronal)

El simulador inyecta, como ocurriría en una planta real, decenas de eventos de falla al año, no miles. Esa escasez de etiquetas determina la elección de modelo:

- Un clasificador binario ("¿este equipo falla en los próximos 14 días?") es entrenable con los datos disponibles y produce resultados interpretables.
- Una red neuronal profunda sería sobreingeniería dado el tamaño del dataset.
- La detección de anomalías por equipo, que no requiere etiquetas, cubre los casos donde los eventos históricos no alcanzan para entrenar un clasificador.

## La métrica que manda: curva de costos

La evaluación se basa en una curva de costos, no en accuracy ni F1: para cada nivel de sensibilidad del modelo se cuenta cuántas inspecciones de una hora (falsas alarmas) hacen falta por cada parada de 18 horas evitada. El umbral de decisión lo elige mantenimiento sobre esa curva, no el data scientist sobre el valor de F1. Cada alerta se acompaña de las señales que más contribuyeron a ella ("corriente subiendo 8% en dos semanas, RMS en zona C"), porque una inspección sin justificación no genera confianza en el equipo de planta.

## El análisis financiero

Todo lo que sigue sale de contar los eventos del histórico, no de estimarlos. La ventana observada son 2.55 años.

### Lo que se pierde hoy

Los seis equipos de criticidad A son los que detienen la máquina completa al fallar. En la ventana observada acumulan 24 eventos, que anualizados dan **9.4 paradas al año**. De esas, 7.5 dejan rastro en los sensores y 2.0 ocurren sin previo aviso.

La práctica actual, que es la ruta mensual de vibración, anticipa 1 de 37 eventos. A efectos prácticos, las 9.4 llegan como parada de emergencia.

| Concepto | Valor | Origen |
|---|---|---|
| Paradas de emergencia al año | 9.4 | Medido sobre el histórico |
| Duración de una parada no programada | 12 h | Punto medio del rango 8–18 h |
| Costo de la hora de máquina | USD 1.500 | Punto medio del rango 1.000–2.000 |
| **Pérdida anual** | **USD 169.000** | 9.4 × 12 h × 1.500 |

### Lo que cambia con el modelo

El modelo anticipa 4 de cada 5 fallas que dejan rastro, medido sobre datos que nunca vio. Aplicado a las 7.5 detectables al año, son **6.0 eventos que pasan de emergencia a intervención programada**. Los 3.4 restantes, entre silenciosos y no detectados, siguen llegando como hoy.

| Concepto | Valor |
|---|---|
| Eventos anticipados al año | 6.0 |
| Intervención programada, duración | 2 h |
| Costo de esos 6.0 eventos | USD 18.000 |
| Costo de los 3.4 que siguen como emergencia | USD 61.200 |
| **Pérdida anual con el modelo** | **USD 79.200** |
| **Ahorro anual** | **USD 90.000** |

### Lo que cuesta el sistema

| Concepto | USD/año | Nota |
|---|---|---|
| Cómputo en Azure | 20 | Puntuación diaria de cinco minutos y reentrenamiento mensual, sobre un clúster que escala a cero |
| Registro de contenedores | 60 | Cobra por existir, es el único cargo fijo |
| Almacenamiento y observabilidad | 120 | |
| Inspecciones generadas | 2.880 | 8 al mes × 1 h × USD 30 la hora de técnico |
| **Operación anual** | **3.080** | |
| Desarrollo | 7.500 (una vez) | Unas 250 horas |

**El retorno se alcanza en un mes.** Un solo evento anticipado libera USD 15.000, que es el doble de lo que cuesta construir el sistema entero.

### Qué tan sensible es esto a los supuestos

Los dos parámetros que mandan son el costo de la hora de máquina y la duración de una parada no programada. Vale la pena mirar los extremos antes de defender la cifra central:

| Escenario | Hora de máquina | Parada | Ahorro anual |
|---|---|---|---|
| Conservador | USD 1.000 | 8 h | USD 36.000 |
| Central | USD 1.500 | 12 h | USD 90.000 |
| Optimista | USD 2.000 | 18 h | USD 192.000 |

Incluso en el escenario conservador el sistema se paga en el primer trimestre. Esa robustez viene de la asimetría del problema y no de la calidad del modelo: una hora de técnico cuesta cincuenta veces menos que una hora de máquina parada, así que equivocarse alertando de más es barato y equivocarse alertando de menos es caro.

### Lo que este cálculo deja fuera a propósito

**El costo de la reparación no entra.** El rodamiento hay que cambiarlo en los dos escenarios, así que se cancela. Solo se contabiliza el diferencial de horas de máquina, que es lo único que el sistema modifica.

**Tampoco entran los agravantes de una falla no anticipada**, y por eso la cifra queda corta. Un rodamiento que se suelta raya el eje, el repuesto que no estaba en bodega llega con flete urgente, y la cuadrilla trabaja en recargo nocturno. Nada de eso está en los USD 169.000.

**No se contabiliza el valor de las inspecciones que no encuentran nada.** Una de ellas destapó un sensor de vibración descalibrado, que habría seguido mintiendo durante meses. Ese hallazgo tiene valor y aquí figura solo como costo.

## La trampa que hay que saber contar

El modelo alertó sobre la fan pump y la inspección no encontró nada; dos semanas después falló un sensor de vibración, no la bomba. La lectura correcta de este caso es que el modelo también detecta sensores en mal estado, y ese hallazgo, bien comunicado, cuenta como valor generado, no como un falso positivo sin más.

---

*Caso inspirado en la operación típica de una máquina tissue de fibra reciclada. Todos los datos del repo son sintéticos.*
