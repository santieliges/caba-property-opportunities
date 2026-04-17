from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DATA_DIR = PROJECT_ROOT / "data" / "splits"


@dataclass(frozen=True)
class SplitConfig:
    name: str
    input_path: Path
    output_dir: Path
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42


def _validate_split_sizes(train_size: float, val_size: float, test_size: float) -> None:
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(
            "train_size + val_size + test_size debe sumar 1.0. "
            f"Valor actual: {total:.6f}"
        )

    for name, value in {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    }.items():
        if value <= 0 or value >= 1:
            raise ValueError(f"{name} debe estar entre 0 y 1. Valor actual: {value}")


def _split_counts(n_rows: int, train_size: float, val_size: float) -> tuple[int, int, int]:
    if n_rows < 3:
        raise ValueError("Se necesitan al menos 3 filas para generar train/val/test.")

    n_train = int(round(n_rows * train_size))
    n_val = int(round(n_rows * val_size))

    n_train = max(1, min(n_train, n_rows - 2))
    n_val = max(1, min(n_val, n_rows - n_train - 1))
    n_test = n_rows - n_train - n_val

    if n_test < 1:
        n_test = 1
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            raise ValueError("No se pudo reservar al menos 1 fila para test.")

    return n_train, n_val, n_test


def build_dataset_splits(
    *,
    config: SplitConfig,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    _validate_split_sizes(config.train_size, config.val_size, config.test_size)

    df = pd.read_csv(config.input_path)
    n_rows = len(df)
    n_train, n_val, n_test = _split_counts(
        n_rows=n_rows,
        train_size=config.train_size,
        val_size=config.val_size,
    )

    rng = np.random.default_rng(config.random_state)
    permutation = rng.permutation(n_rows)

    train_idx = permutation[:n_train]
    val_idx = permutation[n_train:n_train + n_val]
    test_idx = permutation[n_train + n_val:]

    split_frames = {
        "train": df.iloc[train_idx].copy(),
        "val": df.iloc[val_idx].copy(),
        "test": df.iloc[test_idx].copy(),
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in split_frames.items():
        split_df["split"] = split_name
        split_df.to_csv(config.output_dir / f"{config.name}_{split_name}.csv", index=False)

    split_assignment = pd.DataFrame(index=df.index)
    split_assignment["split"] = "unassigned"
    split_assignment.loc[train_idx, "split"] = "train"
    split_assignment.loc[val_idx, "split"] = "val"
    split_assignment.loc[test_idx, "split"] = "test"

    assignment_df = df.copy()
    assignment_df["split"] = split_assignment["split"].values
    assignment_df.to_csv(config.output_dir / f"{config.name}_with_split.csv", index=False)

    assignment_columns = ["split"]
    if "id" in df.columns:
        assignment_columns = ["id", "split"]
    assignment_df.loc[:, assignment_columns].to_csv(
        config.output_dir / f"{config.name}_split_assignment.csv",
        index=False,
    )

    if verbose:
        print(f"[build_dataset_splits:{config.name}] input={config.input_path}")
        print(f"  - total: {n_rows}")
        print(f"  - train: {len(split_frames['train'])}")
        print(f"  - val: {len(split_frames['val'])}")
        print(f"  - test: {len(split_frames['test'])}")
        print(f"  - output_dir: {config.output_dir}")

    return split_frames


def build_venta_splits(
    *,
    input_path: str | Path = PROCESSED_DATA_DIR / "arg_venta_data_processed.csv",
    output_dir: str | Path = SPLITS_DATA_DIR,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    return build_dataset_splits(
        config=SplitConfig(
            name="arg_venta_data",
            input_path=Path(input_path),
            output_dir=Path(output_dir),
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
        )
    )


def build_alquiler_splits(
    *,
    input_path: str | Path = PROCESSED_DATA_DIR / "arg_alquiler_data_processed.csv",
    output_dir: str | Path = SPLITS_DATA_DIR,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    return build_dataset_splits(
        config=SplitConfig(
            name="arg_alquiler_data",
            input_path=Path(input_path),
            output_dir=Path(output_dir),
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
        )
    )
