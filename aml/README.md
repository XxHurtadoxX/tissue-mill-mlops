# Infraestructura en Azure Machine Learning

Todo se declara en YAML y se aplica con la CLI, sin tocar el portal. Así el entorno se reconstruye igual las veces que haga falta y los cambios quedan versionados junto al código.

## Antes de empezar

```bash
az extension add -n ml -y
az login
az account set --subscription "<tu-suscripcion>"
```

Los assets de datos apuntan al histórico completo, que no está versionado. Hay que generarlo primero:

```bash
python -m src.simulator.generate --from 2024-01-01 --to 2026-07-26 --out workdir
python -m src.features.build_dataset --data workdir --out workdir/gold
```

## Provisionar

```bash
RG=rg-tissue-mlops
WS=mlw-tissue-mlops

az group create -n $RG -l eastus
az ml workspace create -f setup/workspace.yml -g $RG
az ml compute create -f setup/compute-cluster.yml -g $RG -w $WS
```

El clúster se crea con `min_instances: 0`, así que no cuesta nada mientras no haya trabajos en cola. No se crea instancia de cómputo: para todo esto basta la CLI, y una instancia encendida y olvidada es la principal fuente de gasto imprevisto en Azure Machine Learning.

## Registrar los datos

```bash
for f in data/*.yml; do az ml data create -f "$f" -g $RG -w $WS; done
```

El tipo de cada asset no es un detalle administrativo, condiciona qué puede consumirlo:

| Tipo | Cuándo | Ejemplo |
|---|---|---|
| `uri_folder` | Un directorio que se recorre entero | `tissue-historian-bronze` |
| `uri_file` | Un archivo único que se lee completo | `tissue-ground-truth` |
| `mltable` | Lleva el esquema junto a los datos, requisito de AutoML | `tissue-training-table` |

## Entrenar y desplegar

```bash
# Pipeline completo: prepara, parte, entrena y decide si reemplazar
az ml job create -f jobs/pipeline-job.yml -g $RG -w $WS --stream

# Endpoint por lotes
az ml batch-endpoint create   -f endpoints/batch-endpoint.yml    -g $RG -w $WS
az ml batch-deployment create -f endpoints/batch-deployment.yml --set-default -g $RG -w $WS
```

El entorno de [`environments/`](environments/) se declara en vez de dejar que Azure lo genere a partir del modelo. La generación automática añade paquetes propios que arrastran dependencias incompatibles, y declararlo además garantiza que el modelo se sirva con las mismas versiones con las que se ajustó.

## Costo

El workspace, el registro de datos y el clúster en reposo no generan cargo apreciable. Se paga el tiempo de cómputo de los trabajos. El endpoint por lotes tampoco reserva nada: usa el clúster, que vuelve a cero nodos al terminar.

Lo que sí cobra por día es el registro de contenedores, que Azure crea la primera vez que se construye un entorno propio. Conviene tenerlo presente si el grupo de recursos va a quedar en pie sin uso.

```bash
az ml compute list -g $RG -w $WS -o table   # el clúster debe estar en cero nodos
```
