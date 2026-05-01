# Case Study

## Problema

El objetivo del proyecto es estimar precio de publicacion por m² para propiedades en CABA y, a partir de esa estimacion, detectar oportunidades de mercado: publicaciones que parecen estar por debajo de lo esperable para su contexto espacial y atributos.

## Pipeline

El trabajo esta organizado como un pipeline de punta a punta:

1. `scraper_service/` scrapea publicaciones inmobiliarias.
2. El sistema actualiza periodicamente los avisos y conserva un historico de:
   - cambios de precio
   - publicaciones dadas de baja
   - estados intermedios de los avisos
3. `ml_core/` limpia, normaliza y prepara los datos para modelado.
4. Los notebooks entrenan y comparan distintos modelos espaciales.
5. El predictor elegido alimenta un detector de oportunidades basado en residuos y contexto espacial.

## Modelos Explorados

Los modelos explorados durante el proyecto fueron:

- `SAR`
- `LGWR`
- `RF + Kriging`
- `GNN` basado en grafos espaciales

Con el crecimiento del dataset y de la dimensionalidad, los modelos clasicos basados en inversiones de matrices grandes dejaron de ser faciles de escalar. Por eso el proyecto termino priorizando modelos con mejor tradeoff entre calidad predictiva, interpretabilidad operativa y costo computacional.

## Modelo de Referencia

El modelo con mejor desempeno practico en la etapa actual es el experimento de [`notebooks/12_rf_kriging.ipynb`](notebooks/12_rf_kriging.ipynb), usando `RegressionKrigingModel` con `use_kriging=False` para la prediccion base y residuos OOF para el modulo de oportunidades.

Metricas de validacion:

| Metrica | Valor |
| --- | ---: |
| RMSE | 36,510 |
| MAE | 16,734 |
| R² | 0.916 |
| Bias | 4,564 |
| Median absolute error | 7,272 |
| MAPE | 10.21% |

## Deteccion de Oportunidades

La deteccion de oportunidades no se apoya solo en errores grandes del modelo. Busca publicaciones que sean:

- mas baratas de lo esperable segun el predictor
- atipicamente bajas de forma robusta
- y, ademas, ubicadas en contextos locales donde el patron espacial sugiere presion de precios mas alta

Para eso se usa una estrategia combinada:

- `ZTest`: identifica residuos anormalmente bajos
- `LISA`: incorpora contexto espacial local
- `Combined Z + LISA`: prioriza propiedades baratas en zonas que muestran una dinamica espacial favorable

Artefacto principal:

- [Mapa interactivo de oportunidades](notebooks/output/12_rf_kriging/outliers_oof/combined_z_lisa_interactive_map.html)

## Hallazgos

- El valor espacial agrega informacion util mas alla de las features del inmueble.
- La deteccion por residuos permite pasar de un problema de prediccion a un caso de uso mas accionable.
- Los modelos espaciales clasicos aportaron intuicion, pero no mantuvieron el mismo nivel de escalabilidad al crecer el problema.
- El flujo mas solido hoy combina pipeline de datos historicos, predictor tabular fuerte y diagnostico espacial sobre residuos.

## Donde Mirar

- [`README.md`](README.md): resumen corto del proyecto.
- [`scraper_service/README.md`](scraper_service/README.md): scraping y actualizacion incremental.
- [`notebooks/12_rf_kriging.ipynb`](notebooks/12_rf_kriging.ipynb): predictor de referencia y outliers espaciales.
- [`notebooks/13_gnn.ipynb`](notebooks/13_gnn.ipynb): linea de trabajo con grafos espaciales.
