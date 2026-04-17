# Pipelines

Este paquete concentra la logica analitica del proyecto:

- limpieza y preparacion de datos
- modelos espaciales
- deteccion de outliers
- soporte para notebooks y experimentacion

Los notebooks pueden seguir viviendo en la carpeta `notebooks/`, pero la logica reutilizable deberia importarse desde `ml_core.*`.

## Datos

- `data/raw/`: salida cruda del scraper y archivos auxiliares como `dolar_hoy.csv`.
- `data/processed/`: datasets generados por `ml_core.preprocessing`.
- `data/splits/`: particiones reproducibles `train/val/test` para reutilizar en notebooks.

## Scripts utiles

Generar datasets procesados:

```bash
python -m ml_core.preprocessing.build_processed_data --dataset all
```

Generar splits reproducibles:

```bash
python -m ml_core.preprocessing.build_dataset_splits --dataset venta
```

## Pruebas Docker

Build:

```bash
./ml_core/scripts/test_build_docker.sh
```

Smoke test de imports:

```bash
./ml_core/scripts/test_imports_docker.sh
```

Smoke test de preprocessing:

```bash
./ml_core/scripts/test_preprocessing_docker.sh
```

Levantar Jupyter:

```bash
./ml_core/scripts/run_jupyter_docker.sh
```
