# Pipelines

Este paquete concentra la logica analitica del proyecto:

- limpieza y preparacion de datos
- modelos espaciales
- deteccion de outliers
- soporte para notebooks y experimentacion

Los notebooks pueden seguir viviendo en la carpeta `notebooks/`, pero la logica reutilizable deberia importarse desde `pipelines.*`.

## Pruebas Docker

Build:

```bash
./pipelines/scripts/test_build_docker.sh
```

Smoke test de imports:

```bash
./pipelines/scripts/test_imports_docker.sh
```

Smoke test de preprocessing:

```bash
./pipelines/scripts/test_preprocessing_docker.sh
```

Levantar Jupyter:

```bash
./pipelines/scripts/run_jupyter_docker.sh
```
