from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import Kernel
from sklearn.model_selection import KFold

from .spatialOutlierDetector import SpatialOutlierDetector
from .strategies import (
    LISAStrategy,
    NegativeResidualsStrategy,
    QuantileStrategy,
    ZTestStrategy,
    CombinedZLisaStrategy,
)


DEFAULT_STRATEGIES = {
    "negative": NegativeResidualsStrategy(),
    "ztest": ZTestStrategy(),
    "quantile": QuantileStrategy(),
    "lisa": LISAStrategy(),
    "combined_z_lisa": CombinedZLisaStrategy(),
}


def detect_model_outliers(
    *,
    model,
    residuals,
    gdf,
    coords,
    output_dir,
    methods=None,
    params_for_methods=None,
    k_neighbors=15,
    strategies=None,
    model_name=None,
    w=None,
    **legacy_params_for_methods,
):

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    params_for_methods = params_for_methods or {}
    if legacy_params_for_methods:
        params_for_methods = {
            **params_for_methods,
            **legacy_params_for_methods,
        }

    if methods is None:
        methods = [
            "negative",
            "ztest",
            "quantile",
            "lisa",
        ]

    strategies = strategies or DEFAULT_STRATEGIES
    detector = SpatialOutlierDetector()
    res = np.asarray(residuals).reshape(-1)
    coords_arr = np.asarray(coords)

    current_model_name = (
        model_name
        or model.__class__.__name__
    )

    results = {}

    # Crear w si no se pasa y alguna estrategia lo requiere
    if w is None:
        needs_weights = any(
            strategies[method].requires_weights
            for method in methods
            if method in strategies
        )
        if needs_weights:
            w = Kernel(
                coords_arr,
                k=k_neighbors,
                function="gaussian",
                fixed=False,
            )

    for method in methods:
        if method not in strategies:
            raise ValueError(
                f"Estrategia desconocida: {method}"
            )

        strategy = strategies[method]
        params_for_method = (
            params_for_methods.get(method, {})
        )

        result = strategy.run(
            res=res,
            gdf=gdf,
            coords=coords_arr,
            detector=detector,
            w=w,
            params_for_method=params_for_method,
            output_dir=output_dir,
            model_name=current_model_name,
        )
        results[method] = result

    return results


def detect_outliers_oof(
    *,
    model_factory,
    X,
    y,
    gdf,
    coords,
    output_dir,
    n_splits=5,
    methods=None,
    params_for_methods=None,
    k_neighbors=15,
    random_state=42,
    fit_kwargs_resolver=None,
    predict_kwargs_resolver=None,
    model_name=None,
):

    if not callable(model_factory):
        raise ValueError(
            "model_factory debe ser callable y devolver "
            "una instancia nueva del modelo por fold."
        )

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    current_model_name = model_name
    all_results = {}
    all_residuals = np.full(
        len(y),
        np.nan,
        dtype=float,
    )

    for fold, (train_idx, val_idx) in enumerate(
        kf.split(X),
        start=1,
    ):

        print(f"\nFold {fold}/{n_splits}")

        model = model_factory()

        if current_model_name is None:
            current_model_name = model.__class__.__name__

        X_train = _slice_like(
            X,
            train_idx,
        )
        X_val = _slice_like(
            X,
            val_idx,
        )
        y_train = _slice_like(
            y,
            train_idx,
        )
        y_val = _slice_like(
            y,
            val_idx,
        )
        gdf_train = _slice_like(
            gdf,
            train_idx,
        )
        gdf_val = _slice_like(
            gdf,
            val_idx,
        )
        coords_train = np.asarray(coords)[train_idx]
        coords_val = np.asarray(coords)[val_idx]

        context = {
            "fold": fold,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "X": X,
            "y": y,
            "gdf": gdf,
            "coords": np.asarray(coords),
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train,
            "y_val": y_val,
            "gdf_train": gdf_train,
            "gdf_val": gdf_val,
            "coords_train": coords_train,
            "coords_val": coords_val,
            "model": model,
        }

        fit_kwargs = _resolve_kwargs(
            fit_kwargs_resolver,
            context,
        )
        predict_kwargs = _resolve_kwargs(
            predict_kwargs_resolver,
            context,
        )

        model.fit(
            X_train,
            y_train,
            coords_train,
            **fit_kwargs,
        )

        y_pred = np.asarray(
            model.predict(
                X_val,
                coords_val,
                **predict_kwargs,
            )
        ).reshape(-1)

        y_val_arr = np.asarray(y_val).reshape(-1)
        residuals = y_val_arr - y_pred
        all_residuals[val_idx] = residuals

        results = detect_model_outliers(
            model=model,
            residuals=residuals,
            gdf=gdf_val,
            coords=coords_val,
            output_dir=output_dir,
            methods=methods,
            params_for_methods=params_for_methods,
            k_neighbors=k_neighbors,
            model_name=model_name,
        )

        for method_name, result_df in results.items():
            if result_df is None or result_df.empty:
                continue

            result_df = result_df.copy()
            result_df["fold"] = fold
            result_df["method"] = method_name

            base_cols = [
                column
                for column in ["idx", "url"]
                if column in result_df.columns
            ]
            other_cols = [
                column
                for column in result_df.columns
                if column not in base_cols + ["fold", "method"]
            ]
            result_df = result_df[
                base_cols
                + ["fold", "method"]
                + other_cols
            ]

            all_results.setdefault(method_name, []).append(result_df)

    aggregated_results = {}
    for method_name, result_frames in all_results.items():
        if result_frames:
            aggregated_df = pd.concat(
                result_frames,
                ignore_index=True,
            )
        else:
            aggregated_df = pd.DataFrame()

        aggregated_df.to_csv(
            output_dir / f"outliers_{method_name}_{current_model_name}.csv",
            index=False,
        )
        aggregated_results[method_name] = aggregated_df

    concat_path = output_dir / "outliers_oof_concat.csv"
    if concat_path.exists():
        concat_path.unlink()

    np.save(
        output_dir / "residuals_oof.npy",
        all_residuals,
    )

    return aggregated_results, all_residuals


def load_active_processed_geodata(
    *,
    data_path,
    feature_cols,
    target_col,
    coord_cols,
    extra_cols=None,
    valid_to_col="valido_hasta",
    crs="EPSG:4326",
):

    data_path = Path(data_path)
    extra_cols = extra_cols or []

    usecols = sorted(
        set(
            list(feature_cols)
            + [target_col]
            + list(coord_cols)
            + list(extra_cols)
            + [valid_to_col]
        )
    )

    raw = pd.read_csv(
        data_path,
        usecols=usecols,
    )

    gdf = gpd.GeoDataFrame(
        raw,
        geometry=gpd.points_from_xy(
            raw[coord_cols[0]],
            raw[coord_cols[1]],
        ),
        crs=crs,
    )

    return gdf.loc[
        lambda df: df[valid_to_col].isna()
    ].copy()


def _slice_like(data, idx):

    if hasattr(data, "iloc"):
        return data.iloc[idx]

    return np.asarray(data)[idx]


def _resolve_kwargs(
    resolver: Callable | None,
    context: dict,
):

    if resolver is None:
        return {}

    resolved = resolver(context)
    if resolved is None:
        return {}

    return deepcopy(resolved)
