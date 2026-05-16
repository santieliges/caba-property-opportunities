from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_core.preprocessing.pipeline_builder import (
    build_preprocessing_pipeline,
)
from ml_core.preprocessing.splitting import (
    split_dataframe,
)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Hace split train/val/test, ajusta el pipeline de preprocessing "
            "solo en train y guarda los tres datasets listos para usar."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=["venta", "alquiler", "all"],
        default="all",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    return parser.parse_args()


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
SPLITS_DATA_DIR = Path("data/splits")
PREPROCESSING_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = (
    PREPROCESSING_DIR
    / "artifacts"
    / "preprocessing_pipeline.joblib"
)


def _deduplicate_latest_by_listing_id(
    df: pd.DataFrame,
    *,
    id_col: str = "id",
    valid_from_col: str = "valido_desde",
) -> pd.DataFrame:
    """Keep only the latest snapshot for each listing id before splitting.

    The raw scraper dataset is historized, so the same publication can appear
    many times with different prices or states. Splitting by row would leak
    near-duplicate versions of the same listing across train/val/test.
    """

    if id_col not in df.columns:
        return df.copy()

    working_df = df.copy()
    working_df["__original_order__"] = range(len(working_df))

    id_series = working_df[id_col]
    rows_without_id = working_df.loc[id_series.isna()].copy()
    rows_with_id = working_df.loc[id_series.notna()].copy()

    if rows_with_id.empty:
        return (
            working_df
            .drop(columns="__original_order__")
            .reset_index(drop=True)
        )

    if valid_from_col in rows_with_id.columns:
        rows_with_id["__valid_from_dt__"] = pd.to_datetime(
            rows_with_id[valid_from_col],
            errors="coerce",
        )
        sort_columns = [id_col, "__valid_from_dt__", "__original_order__"]
    else:
        sort_columns = [id_col, "__original_order__"]

    latest_rows = (
        rows_with_id
        .sort_values(sort_columns, na_position="first")
        .drop_duplicates(subset=[id_col], keep="last")
    )

    deduplicated = pd.concat(
        [latest_rows, rows_without_id],
        axis=0,
        ignore_index=True,
    )

    helper_columns = [
        column_name
        for column_name in ["__original_order__", "__valid_from_dt__"]
        if column_name in deduplicated.columns
    ]

    return (
        deduplicated
        .sort_values("__original_order__")
        .drop(columns=helper_columns)
        .reset_index(drop=True)
    )


def _save_processed_splits(
    split_frames: dict[str, pd.DataFrame],
    *,
    dataset_stem: str,
) -> None:
    SPLITS_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split_name, split_df in split_frames.items():
        split_to_save = split_df.copy()
        split_to_save["split"] = split_name
        split_to_save.to_csv(
            SPLITS_DATA_DIR / f"{dataset_stem}_{split_name}.csv",
            index=False,
        )


def _save_processed_combined(
    split_frames: dict[str, pd.DataFrame],
    *,
    dataset_stem: str,
) -> None:
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined = []
    for split_name, split_df in split_frames.items():
        split_copy = split_df.copy()
        split_copy["split"] = split_name
        combined.append(split_copy)

    combined_df = pd.concat(
        combined,
        axis=0,
        ignore_index=True,
    )
    combined_df.to_csv(
        PROCESSED_DATA_DIR / f"{dataset_stem}_processed.csv",
        index=False,
    )


def process_dataset(
    dataset_name: str,
    *,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
):

    dataset_stem = f"arg_{dataset_name}_data"
    input_path = RAW_DATA_DIR / f"{dataset_stem}.csv"

    print(f"\nProcesando dataset: {dataset_name}")
    print("Input:", input_path)

    df_raw = pd.read_csv(input_path)
    print("Filas originales:", len(df_raw))

    df_raw = _deduplicate_latest_by_listing_id(df_raw)
    print("Filas luego de deduplicar por ultimo estado de cada id:", len(df_raw))

    raw_splits = split_dataframe(
        df_raw,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    print(
        "Split crudo:",
        f"train={len(raw_splits['train'])},",
        f"val={len(raw_splits['val'])},",
        f"test={len(raw_splits['test'])}",
    )

    pipeline = build_preprocessing_pipeline()

    processed_splits = {}
    processed_splits["train"] = pipeline.fit_transform(raw_splits["train"])
    processed_splits["val"] = pipeline.transform(raw_splits["val"])
    processed_splits["test"] = pipeline.transform(raw_splits["test"])

    print(
        "Split procesado:",
        f"train={len(processed_splits['train'])},",
        f"val={len(processed_splits['val'])},",
        f"test={len(processed_splits['test'])}",
    )

    _save_processed_splits(
        processed_splits,
        dataset_stem=dataset_stem,
    )
    _save_processed_combined(
        processed_splits,
        dataset_stem=dataset_stem,
    )

    print("Guardados en:", SPLITS_DATA_DIR)
    print("Dataset combinado guardado en:", PROCESSED_DATA_DIR / f"{dataset_stem}_processed.csv")

    return pipeline


def main() -> None:

    args = parse_args()

    pipelines = {}
    common_kwargs = {
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "random_state": args.random_state,
    }

    if args.dataset in {"venta", "all"}:
        pipelines["venta"] = process_dataset(
            "venta",
            **common_kwargs,
        )

    if args.dataset in {"alquiler", "all"}:
        pipelines["alquiler"] = process_dataset(
            "alquiler",
            **common_kwargs,
        )

    PIPELINE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    joblib.dump(
        pipelines,
        PIPELINE_PATH,
    )

    print("\nPipelines guardados en:")
    print(PIPELINE_PATH)


if __name__ == "__main__":
    main()
