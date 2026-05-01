# Scraper Service

Este directorio es la fuente canónica del scraper.
Los CSV crudos se guardan en `data/raw/` a nivel repositorio.

## Build

```bash
docker build -f Dockerfile -t scraper-service .
```

## Run

```bash
docker run -it scraper-service bash
```

## Scripts

- `python -m scraper_service.update_data`
- `python -m scraper_service.scrap_new_data`

## Pruebas rápidas

Local:

```bash
./scripts/run_update_data_docker.sh
./scripts/scrap_new_data_local.sh
```

Docker:

```bash
./scraper_service/scripts/test_update_data_docker.sh
./scraper_service/scripts/test_scrap_new_data_docker.sh
```
