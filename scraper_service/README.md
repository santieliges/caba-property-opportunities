# Scraper Service

Este directorio es la fuente canónica del scraper.

## Build

```bash
docker build -f scraper_service/Dockerfile -t scraper-service .
```

## Run

```bash
docker run --rm -it scraper-service
```

## Scripts

- `python -m scraper_service.update_data`
- `python -m scraper_service.scrap_new_data`

## Pruebas rápidas

Local:

```bash
./scraper_service/scripts/test_update_data_local.sh
./scraper_service/scripts/test_scrap_new_data_local.sh
```

Docker:

```bash
./scraper_service/scripts/test_update_data_docker.sh
./scraper_service/scripts/test_scrap_new_data_docker.sh
```
