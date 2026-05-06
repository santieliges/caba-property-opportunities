# Scraper Service

Este directorio es la fuente canonica del scraper.
Los CSV crudos se guardan en `data/raw/` a nivel repositorio.


Local: (Build + Run + Script)

```bash
./scripts/run_update_data_docker.sh
./scripts/run_scrape_new_data_docker.sh
```

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
- `python -m scraper_service.scrape_new_data`

## Pruebas rápidas


Docker:

```bash
./scraper_service/scripts/test_update_data_docker.sh
./scraper_service/scripts/test_scrape_new_data_docker.sh
```
