# Routine Job

## Proposito

Esta carpeta contiene la orquestacion de alto nivel del scraper. No extrae
HTML, no decide reglas de negocio y no persiste directamente: coordina esas
capas.

## Archivo principal

- [routine_job.py](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/routine_job/routine_job.py)
  Define `RoutineJob`.

## Responsabilidades

- iniciar y cerrar el scraper
- leer el universo actual desde `Storage`
- actualizar avisos existentes en batches y con concurrencia limitada
- incorporar avisos nuevos detectados por el scraper
- delegar el versionado real en `Synchronizer`
- disparar `storage.save()` en los momentos de persistencia

## Puntos de entrada

### `fetch_and_sync_data(...)`

Uso:
- refrescar avisos ya presentes en el CSV
- procesar cierres o bajas

Entradas:
- `batch_size`
- `delay_s`
- `jitter_s`
- `max_entries`
- `max_concurrency`

Salida:
- `dict` con `processed`, `closed`, `failed`, `total`

Side effects:
- escribe cambios en el storage
- invoca requests concurrentes via `Updater`

### `fetch_and_sync_new_listings(...)`

Uso:
- scraping incremental de nuevos avisos

Entradas:
- `n_pages`
- `delay_s`
- `jitter_s`
- `max_existing_hits`

Salida:
- no devuelve un resumen estructurado; persiste nuevos avisos en storage

Side effects:
- consulta ids ya activos para cortar temprano
- inserta avisos nuevos via `Synchronizer`

## Alcance

Editar esta carpeta cuando cambie:

- el orden general del pipeline
- la estrategia de batching
- el nivel de concurrencia
- el momento en que se guarda el CSV

No editar esta carpeta para:

- cambiar selectores o parsing de Argenprop
- redefinir campos de negocio que disparan versionado
- alterar el formato del CSV

## Puntos de salida

- retorna estadisticas agregadas en el flujo de update
- persiste el CSV actualizado a traves de `Storage.save()`
- deja el scraper cerrado aun si la extraccion falla dentro del flujo
