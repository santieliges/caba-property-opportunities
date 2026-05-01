# Predictor Espacial de Precio por m²

Pipeline de datos y modelado espacial para estimar precios de publicacion inmobiliaria y detectar oportunidades de mercado en CABA.

El proyecto cubre el flujo completo:
- scraping de publicaciones inmobiliarias
- actualizacion continua de avisos y precios
- construccion de un historico de publicaciones dadas de baja o modificadas
- limpieza y preprocesamiento para consumo analitico en `ml_core/`
- entrenamiento de un predictor de precio de publicacion
- deteccion de oportunidades a partir de residuos espaciales

## Overview

El sistema scrapea publicaciones inmobiliarias y conserva solo informacion de publicacion, especialmente precios y atributos del aviso. Sobre esa base:

- se registran actualizaciones periodicas de los avisos
- se conserva traza historica de cambios de precio y bajas
- se limpian y enriquecen los datos para generar datasets modelables
- se entrena un predictor espacial de precio por m²
- se usa la diferencia entre precio esperado y precio publicado para detectar oportunidades

La idea final no es solo predecir, sino encontrar propiedades publicadas por debajo de lo que el contexto espacial y los atributos del inmueble sugieren.

## Resultado Principal

El modelo con mejor rendimiento y actualmente tomado como referencia operativa es `RF + Kriging`, usado hoy en modo `Random Forest` sobre residuos espaciales OOF para deteccion de oportunidades.

Metricas de validacion del predictor en [`notebooks/12_rf_kriging.ipynb`](notebooks/12_rf_kriging.ipynb):

| Metrica | Valor |
| --- | ---: |
| RMSE | 36,510 |
| MAE | 16,734 |
| R² | 0.916 |
| Bias | 4,564 |
| Median absolute error | 7,272 |
| MAPE | 10.21% |

Estas metricas corresponden al predictor de precio de publicacion evaluado sobre el split de validacion del notebook `12_rf_kriging`.

## Mapa de Oportunidades

El buscador de oportunidades combina:
- un estimador de precio esperado
- residuos `precio observado - precio esperado`
- deteccion robusta de valores atipicamente bajos
- contexto espacial local con una estrategia combinada `ZTest + LISA`

Resultado interactivo:

- [Mapa interactivo de oportunidades](notebooks/output/12_rf_kriging/outliers_oof/combined_z_lisa_interactive_map.html)

Ese mapa destaca publicaciones inusualmente baratas y da prioridad a las que aparecen en zonas donde el contexto espacial sugiere presion de precios mas alta.

## Flujo del Proyecto

1. `scraper_service/` scrapea publicaciones y actualiza el historico de avisos.
2. `data/` almacena datos crudos, procesados y splits reutilizables.
3. `ml_core/` concentra limpieza, features, modelos, evaluacion y deteccion de outliers.
4. `notebooks/` documenta los experimentos y genera artefactos de salida.
5. `notebooks/output/12_rf_kriging/outliers_oof/combined_z_lisa_interactive_map.html` resume el caso de uso final de deteccion de oportunidades.

## Repo Guide

- `scraper_service/`: scraping, sync y actualizacion incremental de publicaciones.
- `ml_core/`: preprocessing, modelos espaciales, evaluacion y visualizacion.
- `data/`: datasets crudos, procesados y splits.
- `GeoData/`: capas geograficas auxiliares.
- `notebooks/`: experimentos y comparacion de modelos.

## Lectura Recomendada

- [`CASE_STUDY.md`](CASE_STUDY.md): explicacion tecnica del proyecto, decisiones de modelado y hallazgos.
- [`scraper_service/README.md`](scraper_service/README.md): como correr scraping y actualizaciones con Docker.
- [`notebooks/12_rf_kriging.ipynb`](notebooks/12_rf_kriging.ipynb): predictor base y deteccion de oportunidades.

