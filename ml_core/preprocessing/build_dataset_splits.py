from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add the project root to sys.path to allow importing ml_core
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_core.preprocessing.splitting import (
    build_alquiler_splits,
    build_venta_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera splits reproducibles train/val/test desde datasets procesados.",
    )
    parser.add_argument(
        "--dataset",
        choices=["venta", "alquiler", "all"],
        default="all",
        help="Dataset a dividir.",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.7,
        help="Proporcion para train.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Proporcion para validation.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Proporcion para test.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla del split reproducible.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    common_kwargs = {
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "random_state": args.random_state,
    }

    if args.dataset in {"venta", "all"}:
        split_frames = build_venta_splits(**common_kwargs)
        print(
            "Generados data/splits/arg_venta_data_{train,val,test}.csv "
            f"con {len(split_frames['train'])}/{len(split_frames['val'])}/{len(split_frames['test'])} filas."
        )

    if args.dataset in {"alquiler", "all"}:
        split_frames = build_alquiler_splits(**common_kwargs)
        print(
            "Generados data/splits/arg_alquiler_data_{train,val,test}.csv "
            f"con {len(split_frames['train'])}/{len(split_frames['val'])}/{len(split_frames['test'])} filas."
            )


if __name__ == "__main__":
    main()
