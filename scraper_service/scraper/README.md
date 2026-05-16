# Scraper

## Proposito

Esta carpeta encapsula el acceso remoto a Argenprop y servicios auxiliares.
Es la capa que sabe navegar paginas, leer HTML, llamar la API de Sosiva y
mapear respuestas remotas al esquema interno.

## Archivos principales

- [scraper_base.py](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/scraper/scraper_base.py)
  Infraestructura comun de Playwright/Patchright.
- [argenprop_scraper.py](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/scraper/argenprop_scraper.py)
  Scraper principal de listados y detalle de avisos.
- [SosivaApiClient.py](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/scraper/SosivaApiClient.py)
  Cliente auxiliar para leer detalle de avisos desde la API intermedia.
- [ambito_dolar_scraper.py](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/scraper/ambito_dolar_scraper.py)
  Scraper secundario para dolar. No es parte del flujo central de propiedades.

## Modelo principal

`InmuebleData` en `argenprop_scraper.py` es el contrato de salida mas
importante de esta carpeta. Normaliza tipos y produce `dict`s compatibles con
`CSVStorage` y `Synchronizer`.

## Flujo interno

1. `BaseScraper.start()` abre un contexto persistente de navegador.
2. `ArgenPropScraper.extract_all_pages(...)` recorre listados y extrae ids/urls.
3. Para cada aviso nuevo o a refrescar:
   - intenta usar la API de Sosiva cuando esta disponible
   - hace fallback a scraping de detalle HTML si hace falta
4. El resultado se empaqueta como `InmuebleData`.

## Inputs

- `url_base`
  Punto de entrada del scraper para listados.
- `headless`
  Controla visibilidad del navegador.
- `download_images`
  Si es `True`, descarga imagenes del aviso en `images/`.
- `use_api_details`
  Habilita priorizar detalles desde Sosiva.

## Outputs y side effects

- devuelve `dict`s o listas de `InmuebleData`
- puede descargar imagenes a `images/`
- puede escribir debug HTML/screenshots a `output/scrape_debug/`
- guarda estado del navegador en `storage/playwright_state_<Clase>.json`
- reusa `playwright_user_data_<Clase>/` para el perfil del browser

## Alcance

Editar esta carpeta cuando cambie:

- la estructura HTML de Argenprop
- la forma de extraer datos de detalle
- la heuristica de deteccion de `pozo`
- la integracion con Sosiva
- el manejo de challenges, popups o rate limits

No editar esta carpeta para:

- cambiar reglas de versionado historico
- modificar el CSV o su esquema
- cambiar batching o semaforos del pipeline

## Puntos de salida y errores relevantes

- `extract_all_pages(...)` puede cortar por:
  - fin de paginas
  - exceso de avisos ya conocidos
  - errores HTTP como `404` o `429`
- `_pause_for_human_check(...)` bloquea esperando input humano si aparece un
  challenge de verificacion
- `Updater` interpreta `404` y `410` como cierre del aviso
