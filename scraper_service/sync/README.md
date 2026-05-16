# Sync

## Proposito

Esta carpeta define como se fusionan los datos nuevos o refrescados contra el
storage existente. Es la capa que decide si un aviso se inserta, se versiona o
se cierra.

## Archivo principal

- [sync.py](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/sync/sync.py)
  Define `Synchronizer`.

## Reglas principales

`Synchronizer.sync_entry(entry_id, entry)` aplica estas reglas:

1. si `entry is None`:
   - cierra el aviso activo si existe
2. si el id no existe:
   - inserta un aviso nuevo
3. si el id existe y cambian campos de negocio:
   - cierra la version anterior
   - inserta una nueva version
4. si no cambia nada relevante:
   - no hace nada

## Campos de negocio

Por default, la comparacion de cambios usa:

- `precio`
- `moneda`
- `ambientes`
- `expensas`
- `latitud`
- `longitud`
- `antiguedad`
- `pozo`
- `area_m2_total`

Si cambia ese conjunto, se considera que el aviso cambio materialmente.

## Inputs

- `Storage`
  Backend donde vive el estado actual e historico.
- `entry_id`
  Id canonico del aviso.
- `entry`
  `dict` normalizado por el scraper/updater.

## Outputs y side effects

- inserta o cierra filas en el storage
- dispara `storage.save()` solo cuando el caller lo invoca externamente

## Alcance

Editar esta carpeta cuando cambie:

- la definicion de "cambio relevante"
- la politica de versionado
- la forma de tratar cierres, bajas o reapariciones

No editar esta carpeta para:

- arreglar selectores o requests remotos
- cambiar el formato del CSV
- orquestar concurrencia o batch size
