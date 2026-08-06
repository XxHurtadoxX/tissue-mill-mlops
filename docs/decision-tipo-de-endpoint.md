# Por qué el despliegue es por lotes y no en línea

Azure Machine Learning ofrece dos formas de servir un modelo, y la elección no
es de gusto: cambia el costo en dos órdenes de magnitud.

## Lo que pide este problema

El sistema puntúa **doce equipos una vez al día**. La lista de alertas se revisa
en la reunión de la mañana, y una intervención se programa con días de
antelación. Nadie necesita una respuesta en cincuenta milisegundos.

## Lo que cuesta cada opción

| | En línea | Por lotes |
|---|---|---|
| Cómo factura | Por hora mientras el endpoint exista, lo llamen o no | Solo mientras corre el trabajo |
| Horas facturadas al día | 24 | ~0.08 (unos cinco minutos) |
| Cómputo | Instancias dedicadas, siempre encendidas | El clúster, que escala a cero al terminar |

**Son 288 veces más horas facturadas** para atender doce predicciones diarias.
La proporción no depende del precio de la máquina que se elija.

## Cuándo sí conviene el de tiempo real

No es que uno sea mejor que el otro, es que responden a necesidades distintas:

- Una aplicación que consulta durante una interacción con el usuario.
- Un sistema de control que reacciona en segundos.
- Cualquier caso donde esperar minutos no sea aceptable.

Si mañana la planta quisiera un tablero donde el planificador consulta un equipo
concreto bajo demanda, ese sería el argumento para añadir uno en línea. Hoy no
existe esa necesidad, y desplegarlo sería pagar disponibilidad permanente para
un proceso que corre una vez al día.

## Lo que se pierde al elegir lotes

Conviene decirlo. El endpoint en línea permite repartir el tráfico entre
versiones por porcentaje, lo que habilita liberar una versión nueva al 10% de
las peticiones y observar antes de darle el resto.

El de lotes no reparte tráfico, pero sí admite varios despliegues bajo el mismo
endpoint con uno marcado como predeterminado. Liberar una versión nueva es
cambiar cuál es el predeterminado, y volver atrás es cambiarlo de vuelta. Es
menos granular y conserva lo esencial: la vuelta atrás no exige redesplegar.
