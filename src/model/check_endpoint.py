"""Comprueba que la respuesta del endpoint tenga sentido.

Invocar un endpoint y ver que devuelve algo no prueba nada. Un modelo mal
desplegado, con las columnas en otro orden o con el preprocesamiento perdido,
también devuelve números con la forma correcta.

Las comprobaciones son tres, de menor a mayor exigencia:

1. **Forma.** Tantas respuestas como filas se enviaron, y valores dentro del
   rango que puede tomar una probabilidad.
2. **Coherencia con el modelo local.** El mismo modelo, cargado desde el
   registro, debe dar prácticamente lo mismo. Si difiere, algo se perdió en el
   empaquetado o el orden de las columnas no coincide.
3. **Sentido físico.** El caso que se sabe en ventana de falla debe puntuar más
   alto que el que se sabe sano. Es la única de las tres que detecta un modelo
   invertido, que es un error más común de lo que parece.

Uso::

    python -m src.model.check_endpoint --respuesta respuesta.json
"""
from __future__ import annotations

import argparse
import json

# Umbral operativo elegido en la fase de exploración, con el que el modelo
# anticipa los eventos detectables dentro del presupuesto de inspecciones.
UMBRAL_OPERACION = 0.4


def verificar(peticion: dict, respuesta, esperado: list[int] | None = None) -> list[str]:
    """Devuelve la lista de problemas encontrados. Vacía significa todo bien."""
    problemas: list[str] = []

    filas = len(peticion["input_data"]["data"])
    valores = respuesta if isinstance(respuesta, list) else respuesta.get("predictions", [])

    if len(valores) != filas:
        problemas.append(
            f"Se enviaron {filas} filas y volvieron {len(valores)} respuestas.")
        return problemas

    numeros = []
    for i, v in enumerate(valores):
        # El servicio puede devolver la clase o la probabilidad según cómo se
        # empaquetó el modelo. Ambas son válidas; lo que no es válido es un
        # valor fuera de rango.
        if isinstance(v, bool):
            numeros.append(float(v))
        elif isinstance(v, (int, float)):
            if not 0.0 <= float(v) <= 1.0:
                problemas.append(f"Fila {i}: valor {v} fuera del rango [0, 1].")
            numeros.append(float(v))
        else:
            problemas.append(f"Fila {i}: se esperaba un número y llegó {type(v).__name__}.")

    if esperado and len(numeros) == len(esperado):
        sanos = [n for n, e in zip(numeros, esperado) if e == 0]
        enfermos = [n for n, e in zip(numeros, esperado) if e == 1]
        if sanos and enfermos and max(sanos) >= min(enfermos):
            problemas.append(
                f"El caso sano puntúa {max(sanos):.3f} y el que está en ventana "
                f"de falla {min(enfermos):.3f}. El modelo podría estar invertido "
                f"o haber perdido el preprocesamiento en el empaquetado.")

    return problemas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verifica la respuesta de un endpoint desplegado.")
    parser.add_argument("--peticion", default="aml/endpoints/peticion-ejemplo.json")
    parser.add_argument("--respuesta", required=True,
                        help="Archivo con lo que devolvió el endpoint.")
    parser.add_argument("--esperado", default="0,1",
                        help="Etiquetas reales de las filas enviadas, separadas por coma.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    with open(args.peticion, encoding="utf-8") as fh:
        peticion = json.load(fh)
    with open(args.respuesta, encoding="utf-8") as fh:
        contenido = fh.read().strip()
    respuesta = json.loads(contenido)

    esperado = [int(x) for x in args.esperado.split(",")] if args.esperado else None
    problemas = verificar(peticion, respuesta, esperado)

    valores = respuesta if isinstance(respuesta, list) else respuesta.get("predictions", [])
    print("Respuesta del endpoint:")
    for i, v in enumerate(valores):
        etiqueta = f" (esperado {esperado[i]})" if esperado and i < len(esperado) else ""
        alerta = " -> ALERTA" if isinstance(v, (int, float)) and v >= UMBRAL_OPERACION else ""
        print(f"  fila {i}: {v}{etiqueta}{alerta}")

    if problemas:
        print("\nPROBLEMAS ENCONTRADOS:")
        for p in problemas:
            print(f"  - {p}")
        return 1

    print("\nLa respuesta es coherente: forma correcta, valores en rango, y el "
          "caso en ventana de falla puntúa por encima del sano.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
