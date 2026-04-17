# Modelo de Precio por Metro Cuadrado

Modelo espacial para estimar el precio por m² de alquiler usando
kernel geográfico y regresión.

## Estructura

- `scraper_service/`: servicio canónico de scraping, sincronizacion y actualizacion de avisos.
- `ml_core/`: limpieza, modelos, outliers y soporte para notebooks.
- `data/raw/`: datos crudos generados por scraping y fuentes auxiliares.
- `data/processed/`: datasets procesados listos para modelado y analisis.
- `notebooks/`: exploracion y analisis interactivo apoyado sobre `ml_core.*`.

## Docker

El scraping y los entornos Docker viven solo en `scraper_service/` y `ml_core/`.

Para construir el scraper:

```bash
docker build -f scraper_service/Dockerfile -t scraper-service .
```

Para construir el entorno de notebooks de `ml_core`:

```bash
docker build -f ml_core/docker/Dockerfile -t predictor-pipelines .
```

Luego podés ejecutar cada contenedor con `docker run` según el flujo que necesites.
