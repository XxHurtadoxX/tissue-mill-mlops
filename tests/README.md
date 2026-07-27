# Pruebas

Están separadas en dos grupos porque cumplen funciones distintas y tienen
distinto destino si este proyecto se llevara a una planta real.

## `pipeline/`

Prueban la transformación de datos crudos en tabla de entrenamiento: limpieza,
ventanas móviles, contexto de mantenimiento y etiquetas.

Este es el grupo que existe en cualquier proyecto de mantenimiento predictivo,
y su valor está en un tipo de error concreto: **una fuga de información no
produce ningún fallo visible**. El código corre, el pipeline termina y las
métricas salen mejores que antes. El problema aparece meses después, cuando el
modelo rinde en producción la mitad de lo prometido y ya nadie recuerda qué
cambió. Contra eso no sirve revisar el código con cuidado, porque el error
reaparece en la siguiente modificación. Solo sirve una prueba que lo vigile.

Cada guarda tiene la suya: que las ventanas no incorporen valores futuros, que
una orden ya cerrada no etiquete el presente, que los días con el equipo en
reparación queden fuera y que las lecturas inválidas no entren a los promedios.

## `simulator/`

Prueban que el generador de datos sea determinista y produzca las firmas físicas
que dice producir.

Son específicas de este proyecto. En una planta real no hay simulador, aunque el
grupo no desaparece del todo: lo que aquí genera datos, allá los ingiere desde
el historian y desde SAP, y ese código de ingesta se prueba con las mismas
preguntas sobre esquema, huecos y cobertura del mapeo entre tags y equipos.

## Lo que estas pruebas no hacen

No verifican la exactitud del modelo. Nadie escribe `assert auc > 0.85` en una
suite de pruebas, porque ese valor depende de los datos y cambia con cada
reentrenamiento. Medir la calidad del modelo es evaluación, y vive en el
pipeline de entrenamiento.

## Ejecución

```bash
pytest                    # todo
pytest tests/pipeline     # solo la transformación de datos
```
