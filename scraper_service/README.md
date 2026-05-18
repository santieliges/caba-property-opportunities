# Scraper Service

Documentacion de arquitectura para agentes y contributors que necesiten
entender, ejecutar o modificar el pipeline de scraping de Argenprop.

## Objetivo

`scraper_service/` concentra la ingesta y actualizacion del dataset crudo de
publicaciones inmobiliarias. Su flujo principal trabaja sobre
`data/raw/arg_venta_data.csv` a nivel repositorio y mantiene historial de
cambios por aviso.

Los casos de uso activos son tres:

1. `update_data.py`
   Actualiza avisos ya conocidos y cierra los que dejaron de estar activos.
2. `scrape_caba_bulk_listings.py`
   Hace scraping de volumen para poblar o ampliar el dataset de ventas en CABA.
3. `sync_daily_new_listings.py`
   Sincroniza avisos nuevos desde una URL ordenada por mas nuevos y corta al
   encontrar avisos ya existentes.

## Flujo general

1. Un script de entrada construye `CSVStorage`, `ArgenPropScraper`,
   `Synchronizer`, `Updater` y `RoutineJob`.
2. `RoutineJob` coordina el trabajo:
   - `fetch_and_sync_data()` para refrescar avisos existentes.
   - `fetch_and_sync_new_listings()` para incorporar avisos nuevos.
3. `ArgenPropScraper` obtiene ids, URLs y detalles:
   - desde el HTML de Argenprop
   - o, cuando se puede, desde la API de Sosiva
4. `Updater` transforma la respuesta remota en un `dict` compatible con el
   storage y resuelve casos de cierre o fallback.
5. `Synchronizer` decide si:
   - inserta un aviso nuevo
   - versiona un aviso existente
   - cierra un aviso
6. `CSVStorage` persiste el estado historizado en CSV.

## Mapa de carpetas

- `scraper/`
  Capa de acceso a Argenprop y servicios auxiliares. Vease
  [scraper/README.md](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/scraper/README.md).
- `routine_job/`
  Orquestacion de flujos de update y sync. Vease
  [routine_job/README.md](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/routine_job/README.md).
- `storage/`
  Persistencia historizada en CSV y runtime state de Playwright. Vease
  [storage/README.md](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/storage/README.md).
- `sync/`
  Reglas de sincronizacion y versionado. Vease
  [sync/README.md](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/sync/README.md).
- `updater/`
  Refresh de un aviso individual. Vease
  [updater/README.md](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/updater/README.md).
- `scripts/`
  Wrappers Docker, smoke tests y utilidades. Vease
  [scripts/README.md](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/scripts/README.md).
- `tests/`
  Pruebas automatizadas y smoke checks del flujo. Vease
  [tests/README.md](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/tests/README.md).

## Puntos de entrada

### Update de avisos existentes

```bash
python -m scraper_service.update_data
./scraper_service/scripts/run_update_data_docker.sh
```

Salida esperada:
- actualiza campos de avisos activos
- marca avisos cerrados cuando la fuente responde `404` o `410`
- escribe logs en `scraper_service/update_data_scraper.log`
- persiste cambios en `data/raw/arg_venta_data.csv`

### Scraping masivo de CABA

```bash
python -m scraper_service.scrape_caba_bulk_listings
./scraper_service/scripts/run_scrape_caba_bulk_listings_docker.sh
```

Salida esperada:
- recorre barrios y rangos de m2
- inserta avisos nuevos en el CSV historizado
- reutiliza la logica de corte por ids existentes para no reingestar de mas

### Sync diario de avisos nuevos

```bash
python -m scraper_service.sync_daily_new_listings \
  "https://www.argenprop.com/departamentos/venta/palermo?orden-masnuevos"

./scraper_service/scripts/run_sync_daily_new_listings_docker.sh

./scraper_service/scripts/run_sync_daily_new_listings_docker.sh \
  "https://www.argenprop.com/departamentos/venta/palermo?orden-masnuevos"
```

Salida esperada:
- asegura el orden `orden-masnuevos`
- toma publicaciones nuevas desde la URL objetivo
- corta cuando detecta avisos ya almacenados
- devuelve cantidad de ids nuevos agregados
- si no se pasa URL al wrapper Docker, usa por default
  `http://argenprop.com/departamentos/venta/capital-federal?orden-masnuevos`

### Automatizacion con cron

Instalar o reinstalar el bloque administrado de `cron`:

```bash
./scraper_service/scripts/install_cron_jobs.sh
```

Schedules default del bloque administrado:
- `0 0 * * *` para `update_data`
- `0 3 * * *` para `sync_daily_new_listings`

Los wrappers de `cron`:
- no usan TTY
- evitan corridas superpuestas con `flock`
- escriben logs en `scraper_service/logs/cron/`
- dejan estado legible en archivos `*.state`, `*.last_start`, `*.last_end`,
  `*.last_success` y `*.last_exit`

Ver estado operativo:

```bash
./scraper_service/scripts/show_cron_status.sh
tail -f scraper_service/logs/cron/update_data.log
tail -f scraper_service/logs/cron/sync_daily_new_listings.log
```

Si queres cambiar horarios al instalar:

```bash
UPDATE_SCHEDULE="30 0 * * *" \
SYNC_SCHEDULE="0 4 * * *" \
./scraper_service/scripts/install_cron_jobs.sh
```

## Contratos e invariantes

- El id canonico del aviso es el id de Argenprop.
- `CSVStorage` conserva historial con `valido_desde` y `valido_hasta`.
- `Synchronizer` solo versiona cuando cambian campos de negocio.
- Un aviso cerrado no se borra: se cierra historicamente.
- El scraper puede pausar la ejecucion si aparece un challenge de "comprobar
  que es humano".

## Runtime state y artefactos

- `data/raw/arg_venta_data.csv`
  Dataset crudo principal versionado fuera de este modulo.
- `storage/playwright_state_<Clase>.json`
  Estado serializado del navegador. Es runtime state, no logica.
- `playwright_user_data_<Clase>/`
  Perfil persistente del navegador para reducir friccion con challenges.
- `output/scrape_debug/`
  HTML y screenshots de paginas problematicas durante el scraping.
- `images/`
  Descarga opcional de imagenes de los avisos cuando `download_images=True`.

## Variables de entorno relevantes

- `HEADLESS`
  Controla si Playwright corre visible o no.
- `PLAYWRIGHT_USER_DATA_DIR`
  Ubicacion del perfil persistente del browser.
- `SCRAPER_STORAGE_DIR`
  Carpeta donde se guarda `storage_state`.
- `BROWSER_CHANNEL`
  Canal de Chromium/Chrome para el contexto persistente.
- `DOCKER_RUNTIME_VOLUME`
  Nombre del volumen Docker usado por los wrappers que requieren estado.

## Guia rapida para agentes

Si un cambio afecta:

- extraccion de HTML o una nueva fuente remota: empezar por `scraper/`
- reglas de versionado o cierres: empezar por `sync/`
- persistencia o esquema del CSV: empezar por `storage/`
- estrategia del pipeline o batching: empezar por `routine_job/`
- flujo de refresh de un aviso puntual: empezar por `updater/`
- ejecucion operativa o Docker: empezar por `scripts/`

## Notas y deuda visible

- Existen scripts de prueba con nombres viejos como
  `test_scrape_new_data_*.sh`; hoy funcionan como wrappers legacy.
- Existe una carpeta anidada `scraper_service/scraper_service/` con datos viejos
  que no forma parte del flujo canonico actual.
- La mayor parte del pipeline principal apunta al mercado de venta de
  departamentos en CABA sobre Argenprop.
