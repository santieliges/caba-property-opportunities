# Modelo de Precio por Metro Cuadrado

Modelo espacial para estimar el precio por m² de alquiler usando
kernel geográfico y regresión.

## Estructura

- `scraper_service/`: servicio canónico de scraping, sincronizacion y actualizacion de avisos.
- `pipelines/`: limpieza, modelos, outliers y soporte para notebooks.
- `notebooks/`: exploracion y analisis interactivo apoyado sobre `pipelines.*`.

## Docker

El scraping y los entornos Docker viven solo en `scraper_service/` y `pipelines/`.

Para construir el scraper:

```bash
docker build -f scraper_service/Dockerfile -t scraper-service .
```

Para construir el entorno de notebooks de `pipelines`:

```bash
docker build -f pipelines/docker/Dockerfile -t predictor-pipelines .
```

Luego podés ejecutar cada contenedor con `docker run` según el flujo que necesites.
