# Infraestructura y assets en Azure Machine Learning

Todo se declara en YAML y se aplica con la CLI, sin tocar el portal. Así el
entorno se reconstruye igual las veces que haga falta y los cambios quedan
versionados junto al código.

## Requisitos previos

```bash
az extension add -n ml -y
az login
az account set --subscription "<tu-suscripcion>"
```

Los assets apuntan al backfill histórico, que no está versionado. Hay que
generarlo antes:

```bash
python -m src.simulator.generate --from 2024-01-01 --to 2026-07-26 --out workdir
python -m src.features.build_dataset --data workdir --out workdir/gold
```

## Provisionar

```bash
RG=rg-tissue-mlops
WS=mlw-tissue-mlops

az group create -n $RG -l eastus
az ml workspace create -f aml/setup/workspace.yml -g $RG
az ml compute create -f aml/setup/compute-cluster.yml -g $RG -w $WS
```

El clúster se crea con `min_instances: 0`, así que no genera costo mientras no
haya trabajos en cola. No se crea instancia de cómputo: para registrar datos
basta la CLI, y una instancia encendida y olvidada es la principal fuente de
gasto imprevisto en Azure Machine Learning.

## Registrar los datos

```bash
az ml data create -f aml/data/bronze-historian.yml -g $RG -w $WS
az ml data create -f aml/data/training-table.yml   -g $RG -w $WS
az ml data create -f aml/data/ground-truth.yml     -g $RG -w $WS
```

| Asset | Tipo | Motivo |
|---|---|---|
| `tissue-historian-bronze` | `uri_folder` | Es un directorio completo de CSV que se recorre entero |
| `tissue-training-table` | `mltable` | Lleva el esquema junto a los datos, requisito de AutoML |
| `tissue-ground-truth` | `uri_file` | Es un archivo único que se lee completo |

## Verificar

```bash
az ml data list    -g $RG -w $WS -o table
az ml compute list -g $RG -w $WS -o table
```

El clúster debe aparecer con cero nodos. Conviene revisarlo al terminar cada
sesión de trabajo.

## Sobre el costo

El workspace, el registro de datos y el clúster en reposo no generan cargo
apreciable. Lo que se paga es el tiempo de cómputo de los trabajos y, si se
crea, la instancia de cómputo mientras esté encendida.

No hace falta borrar el grupo de recursos entre fases: hacerlo obligaría a
repetir el registro de los assets sin ahorrar nada relevante. La regla cambia
con el endpoint en línea de fases posteriores, que sí cobra por hora mientras
exista y debe eliminarse en cuanto se capture la evidencia.
