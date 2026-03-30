from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add the project root to sys.path to allow importing ml_core
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_core.preprocessing.preprocessing import (
    build_alquiler_processed_dataset,
    build_venta_processed_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera CSVs procesados para notebooks de venta y alquiler.",
    )
    parser.add_argument(
        "--dataset",
        choices=["venta", "alquiler", "all"],
        default="all",
        help="Dataset a procesar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset in {"venta", "all"}:
        df_venta = build_venta_processed_dataset()
        print(
            "Generado scraper_service/storage/data/arg_venta_caba_processed.csv "
            f"con {len(df_venta)} filas."
        )

    if args.dataset in {"alquiler", "all"}:
        df_alquiler = build_alquiler_processed_dataset()
        print(
            "Generado scraper_service/storage/data/arg_alquiler_caba_processed.csv "
            f"con {len(df_alquiler)} filas."
        )


if __name__ == "__main__":
    main()
