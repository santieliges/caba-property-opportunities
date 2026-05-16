# Scripts

## Proposito

Esta carpeta contiene wrappers de ejecucion, smoke tests operativos y
utilidades de mantenimiento. No es la capa de negocio del scraper: es la capa
de operacion.

## Grupos de scripts

### Wrappers Docker activos

- [run_update_data_docker.sh](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/scripts/run_update_data_docker.sh)
- [run_scrape_caba_bulk_listings_docker.sh](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/scripts/run_scrape_caba_bulk_listings_docker.sh)
- [run_sync_daily_new_listings_docker.sh](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/scripts/run_sync_daily_new_listings_docker.sh)

Responsabilidad:
- construir la imagen Docker
- montar el repo en `/app`
- inyectar variables de entorno operativas
- opcionalmente montar un volumen persistente para el runtime del browser
- en `run_sync_daily_new_listings_docker.sh`, usar una URL default de
  Capital Federal ordenada por mas nuevos cuando no se pasa argumento

### Utilidades de datos

- [backfill_fecha_publicacion_aviso.py](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/scripts/backfill_fecha_publicacion_aviso.py)

Responsabilidad:
- backfill de campos faltantes sobre el CSV existente

### Smoke tests y diagnosticos

- [test_update_data_docker.sh](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/scripts/test_update_data_docker.sh)
- [test_update_data_local.sh](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/scripts/test_update_data_local.sh)
- [test_scrape_new_data_docker.sh](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/scripts/test_scrape_new_data_docker.sh)
- [test_scrape_new_data_local.sh](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/scripts/test_scrape_new_data_local.sh)
- [test_argenprop_isignored.py](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/scripts/test_argenprop_isignored.py)
- [test_sosiva_api.py](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/scripts/test_sosiva_api.py)

Responsabilidad:
- validar rapido que una ruta de scraping siga viva
- inspeccionar sintomas de bloqueo o rate limiting
- verificar endpoints auxiliares

## Alcance

Editar esta carpeta cuando cambie:

- la forma recomendada de ejecutar el scraper
- la configuracion Docker
- tareas operativas de backfill o smoke test

No editar esta carpeta para:

- cambiar parsing o selectores
- redefinir versionado de avisos
- tocar el contrato del storage

## Notas

- Algunos scripts de prueba aun conservan nombres viejos como
  `test_scrape_new_data_*`. Documentan flujo legacy y pueden requerir
  alineacion futura con los entrypoints nuevos.
