# Predictor Espacial de Precio por m² en CABA

Proyecto end-to-end de data + modelado espacial para estimar precios de publicación inmobiliaria y detectar oportunidades de mercado en la Ciudad de Buenos Aires.

La idea central no es solo predecir precio, sino usar esa predicción para responder una pregunta más accionable:

**qué propiedades parecen publicadas por debajo de lo esperable para su ubicación y atributos**

## Qué muestra este proyecto

- scraping incremental de publicaciones inmobiliarias
- construcción de histórico de cambios de precio y bajas
- limpieza y feature engineering para modelado
- comparación de modelos espaciales y tabulares
- detección de oportunidades a partir de residuos espaciales
- mapas interactivos para explorar propiedades potencialmente subvaluadas

## Resultado principal

El flujo más sólido del proyecto hoy está en [`notebooks/12_rf_kriging.ipynb`](notebooks/12_rf_kriging.ipynb), que es el notebook principal y el que obtuvo los mejores resultados prácticos.

Ese notebook entrena el predictor de referencia, genera residuos OOF y construye el mapa final de oportunidades.

Modelo de referencia actual:
- `RF + Kriging`, usado operativamente como predictor tabular fuerte más diagnóstico espacial sobre residuos

Métricas de validación:

| Métrica | Valor |
| --- | ---: |
| RMSE | 36,510 |
| MAE | 16,734 |
| R² | 0.916 |
| Bias | 4,564 |
| Median absolute error | 7,272 |
| MAPE | 10.21% |

## Demo principal

- [Mapa interactivo de oportunidades - Combined ZTest + LISA](notebooks/output/12_rf_kriging/outliers_oof/combined_z_lisa_interactive_map.html)
- [Mapa interactivo de oportunidades - ZTest](notebooks/output/12_rf_kriging/outliers_oof/z_test_interactive_map.html)

Estos mapas destacan propiedades inusualmente baratas y permiten filtrarlas por precio, metros cuadrados, ambientes, pozo y contexto espacial.

## Cómo está pensado el problema

El pipeline parte de publicaciones inmobiliarias y conserva información de publicación, especialmente:
- precio
- atributos del inmueble
- ubicación
- cambios históricos del aviso

Sobre esa base, el proyecto:
- estima precio esperado de publicación
- calcula residuos `precio observado - precio esperado`
- detecta valores atípicamente bajos de forma robusta
- usa contexto espacial local para priorizar oportunidades más convincentes

## Dónde empezar

Si querés entender el proyecto rápido, este es el orden recomendado:

1. [`README.md`](README.md)
2. [`CASE_STUDY.md`](CASE_STUDY.md)
3. [`notebooks/12_rf_kriging.ipynb`](notebooks/12_rf_kriging.ipynb)
4. [`notebooks/output/12_rf_kriging/outliers_oof/combined_z_lisa_interactive_map.html`](notebooks/output/12_rf_kriging/outliers_oof/combined_z_lisa_interactive_map.html)

## Estructura del repo

- `scraper_service/`: scraping, actualización incremental y utilidades del crawler
- `ml_core/`: preprocessing, modelos, evaluación, outlier analysis y visualización
- `data/`: datos crudos, procesados y splits reutilizables
- `GeoData/`: capas geográficas auxiliares
- `notebooks/`: experimentos, comparación de modelos y artefactos de salida

## Otros experimentos

- [`notebooks/11_sar.ipynb`](notebooks/11_sar.ipynb): modelo SAR
- [`notebooks/13_gnn.ipynb`](notebooks/13_gnn.ipynb): línea de trabajo con `TGNN`
- [`notebooks/13_gnn_simulation_ideal_data.ipynb`](notebooks/13_gnn_simulation_ideal_data.ipynb): sanity checks sobre datos sintéticos

## Lectura complementaria

- [`CASE_STUDY.md`](CASE_STUDY.md): decisiones de modelado, motivación y hallazgos
- [`scraper_service/README.md`](scraper_service/README.md): cómo correr scraping y actualizaciones
- [`ml_core/README.md`](ml_core/README.md): organización de la lógica analítica
