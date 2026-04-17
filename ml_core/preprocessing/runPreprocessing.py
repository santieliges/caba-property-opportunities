from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import joblib

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_core.preprocessing.pipelineBuilder import (
    build_cleaning_pipeline
)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Limpia datasets y guarda versiones procesadas."
    )

    parser.add_argument(
        "--dataset",
        choices=["venta", "alquiler", "all"],
        default="all",
    )

    return parser.parse_args()


# 📦 Paths

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

PIPELINE_PATH = Path(
    "models/preprocessing_pipeline.joblib"
)


def process_dataset(
    dataset_name: str
):

    print(f"\nProcesando dataset: {dataset_name}")

    input_path = (
        RAW_DATA_DIR /
        f"arg_{dataset_name}_data.csv"
    )

    output_path = (
        PROCESSED_DATA_DIR /
        f"arg_{dataset_name}_data_processed.csv"
    )

    df = pd.read_csv(input_path)

    print("Filas originales:", len(df))

    pipeline = build_cleaning_pipeline()

    df_clean = pipeline.fit_transform(df)

    print("Filas luego de limpieza:", len(df_clean))

    # Guardar dataset limpio

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df_clean.to_csv(
        output_path,
        index=False
    )

    print("Guardado en:", output_path)

    return pipeline


def main():

    args = parse_args()

    pipelines = {}

    if args.dataset in {"venta", "all"}:

        pipelines["venta"] = process_dataset(
            "venta"
        )

    if args.dataset in {"alquiler", "all"}:

        pipelines["alquiler"] = process_dataset(
            "alquiler"
        )

    # Guardar último pipeline

    joblib.dump(
        pipelines,
        PIPELINE_PATH
    )

    print("\nPipeline guardado en:")
    print(PIPELINE_PATH)


if __name__ == "__main__":
    main()