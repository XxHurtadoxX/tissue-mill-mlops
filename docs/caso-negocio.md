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

## El valor, en pesos

| | Antes | Después (medido) |
|---|---|---|
| Paradas de emergencia/año | 3 a 5 | 2 a 4 de ellas **anticipadas** |
| Duración por evento | 8 a 18 h | intervención de 1–3 h en parada programada |
| Ahorro por evento evitado | — | 6–15 h de máquina × USD 1.000–2.000/h |

El ahorro estimado es de 25 a 120 millones de pesos por evento evitado, según la duración de la parada y el costo de oportunidad de la hora de máquina. Un solo evento evitado cubre el costo de implementar el sistema.

## La trampa que hay que saber contar

El modelo alertó sobre la fan pump y la inspección no encontró nada; dos semanas después falló un sensor de vibración, no la bomba. La lectura correcta de este caso es que el modelo también detecta sensores en mal estado, y ese hallazgo, bien comunicado, cuenta como valor generado, no como un falso positivo sin más.

---

*Caso inspirado en la operación típica de una máquina tissue de fibra reciclada. Todos los datos del repo son sintéticos.*
