from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import Kernel
from sklearn.model_selection import KFold

from ml_core.preprocessing.knhs import (
    KNHSSchema,
    KNHSWeightSpec,
    LocalWeightBuilder,
    NeighborSearchCore,
    TopKFeatureNeighborSelector,
)

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
    y_true=None,
    y_pred=None,
    feature_data=None,
    gdf,
    coords,
    output_dir,
    methods=None,
    params_for_methods=None,
    k_neighbors=15,
    strategies=None,
    model_name=None,
    w=None,
    knhs_weight_spec: KNHSWeightSpec | None = None,
    knhs_schema: KNHSSchema | None = None,
    knhs_radius_km: float = 2.0,
    knhs_feature_distance_mode: str = "euclidean",
    knhs_lambda_distance: float = 0.5,
    knhs_bandwidth_mode: str = "adaptive",
    knhs_bandwidth: float = 1.0,
    knhs_row_standardize: bool = True,
    knhs_neighbor_selector=None,
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
            w = _build_spatial_weights(
                coords=coords_arr,
                feature_data=feature_data,
                k_neighbors=k_neighbors,
                knhs_weight_spec=knhs_weight_spec,
                knhs_schema=knhs_schema,
                knhs_radius_km=knhs_radius_km,
                knhs_feature_distance_mode=knhs_feature_distance_mode,
                knhs_lambda_distance=knhs_lambda_distance,
                knhs_bandwidth_mode=knhs_bandwidth_mode,
                knhs_bandwidth=knhs_bandwidth,
                knhs_row_standardize=knhs_row_standardize,
                knhs_neighbor_selector=knhs_neighbor_selector,
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
            y_true=y_true,
            y_pred=y_pred,
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
    knhs_weight_spec: KNHSWeightSpec | None = None,
    knhs_schema: KNHSSchema | None = None,
    knhs_radius_km: float = 2.0,
    knhs_feature_distance_mode: str = "euclidean",
    knhs_lambda_distance: float = 0.5,
    knhs_bandwidth_mode: str = "adaptive",
    knhs_bandwidth: float = 1.0,
    knhs_row_standardize: bool = True,
    knhs_neighbor_selector=None,
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
    all_predictions = np.full(
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
        all_predictions[val_idx] = y_pred
        fold_knhs_weight_spec = _slice_knhs_weight_spec(
            knhs_weight_spec,
            val_idx,
        )

        results = detect_model_outliers(
            model=model,
            residuals=residuals,
            y_true=y_val_arr,
            y_pred=y_pred,
            feature_data=X_val,
            gdf=gdf_val,
            coords=coords_val,
            output_dir=output_dir,
            methods=methods,
            params_for_methods=params_for_methods,
            k_neighbors=k_neighbors,
            model_name=model_name,
            knhs_weight_spec=fold_knhs_weight_spec,
            knhs_schema=knhs_schema,
            knhs_radius_km=knhs_radius_km,
            knhs_feature_distance_mode=knhs_feature_distance_mode,
            knhs_lambda_distance=knhs_lambda_distance,
            knhs_bandwidth_mode=knhs_bandwidth_mode,
            knhs_bandwidth=knhs_bandwidth,
            knhs_row_standardize=knhs_row_standardize,
            knhs_neighbor_selector=knhs_neighbor_selector,
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
    np.save(
        output_dir / "predictions_oof.npy",
        all_predictions,
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


def _slice_knhs_weight_spec(
    knhs_weight_spec: KNHSWeightSpec | None,
    idx,
):
    if knhs_weight_spec is None or knhs_weight_spec.similarity_weights is None:
        return knhs_weight_spec

    similarity_weights = knhs_weight_spec.similarity_weights
    if len(similarity_weights) == 1:
        sliced_weights = similarity_weights.copy()
    elif hasattr(similarity_weights, "iloc"):
        sliced_weights = similarity_weights.iloc[idx].copy()
    else:
        similarity_weights_arr = np.asarray(similarity_weights)
        sliced_weights = similarity_weights_arr[idx]

    return KNHSWeightSpec(
        schema=knhs_weight_spec.schema,
        radius_km=knhs_weight_spec.radius_km,
        k=knhs_weight_spec.k,
        feature_distance_mode=knhs_weight_spec.feature_distance_mode,
        lambda_distance=knhs_weight_spec.lambda_distance,
        bandwidth_mode=knhs_weight_spec.bandwidth_mode,
        bandwidth=knhs_weight_spec.bandwidth,
        row_standardize=knhs_weight_spec.row_standardize,
        neighbor_selector=knhs_weight_spec.neighbor_selector,
        similarity_weights=sliced_weights,
        similarity_weight_cols=(
            None
            if knhs_weight_spec.similarity_weight_cols is None
            else list(knhs_weight_spec.similarity_weight_cols)
        ),
    )


def _build_spatial_weights(
    *,
    coords,
    feature_data,
    k_neighbors,
    knhs_weight_spec: KNHSWeightSpec | None = None,
    knhs_schema: KNHSSchema | None = None,
    knhs_radius_km: float = 2.0,
    knhs_feature_distance_mode: str = "euclidean",
    knhs_lambda_distance: float = 0.5,
    knhs_bandwidth_mode: str = "adaptive",
    knhs_bandwidth: float = 1.0,
    knhs_row_standardize: bool = True,
    knhs_neighbor_selector=None,
):
    def _coerce_feature_dataframe_local(
        *,
        feature_data,
        feature_cols=None,
    ):
        if isinstance(feature_data, pd.DataFrame):
            return feature_data.copy()

        feature_arr = np.asarray(feature_data)
        if feature_arr.ndim != 2:
            raise ValueError(
                "feature_data debe ser un DataFrame o un array 2D."
            )

        if feature_cols is None:
            feature_cols = [
                f"feature_{idx}"
                for idx in range(feature_arr.shape[1])
            ]

        if len(feature_cols) != feature_arr.shape[1]:
            raise ValueError(
                "feature_cols debe tener el mismo largo que las columnas de feature_data."
            )

        return pd.DataFrame(
            feature_arr,
            columns=list(feature_cols),
        )

    def _coerce_similarity_weights_frame(
        *,
        similarity_weights,
        feature_cols: list[str] | None,
        weight_cols: list[str] | None,
        n_rows: int,
    ) -> tuple[pd.DataFrame, list[str]]:
        resolved_weight_cols = (
            list(weight_cols)
            if weight_cols is not None
            else (
                [f"__sim_weight_{col}" for col in feature_cols]
                if feature_cols is not None
                else None
            )
        )
        if resolved_weight_cols is None:
            raise ValueError(
                "No se pudieron resolver los nombres de columnas para similarity_weights. "
                "Pasá similarity_weight_cols explícitamente o definí "
                "schema.similarity_feature_cols."
            )

        if isinstance(similarity_weights, pd.DataFrame):
            weights_df = similarity_weights.copy()
            if len(weights_df.columns) != len(resolved_weight_cols):
                raise ValueError(
                    "similarity_weights tiene una cantidad de columnas distinta a los pesos esperados."
                )
            weights_df.columns = resolved_weight_cols
        else:
            weights_arr = np.asarray(similarity_weights, dtype=float)
            if weights_arr.ndim != 2:
                raise ValueError(
                    "similarity_weights debe ser un DataFrame o un array 2D."
                )
            if weights_arr.shape[1] != len(resolved_weight_cols):
                raise ValueError(
                    "similarity_weights debe tener el mismo número de columnas que "
                    "similarity_feature_cols."
                )
            weights_df = pd.DataFrame(weights_arr, columns=resolved_weight_cols)

        if len(weights_df) == 1 and n_rows > 1:
            weights_df = pd.DataFrame(
                np.repeat(weights_df.to_numpy(dtype=float), n_rows, axis=0),
                columns=resolved_weight_cols,
            )

        if len(weights_df) != n_rows:
            raise ValueError(
                "similarity_weights y feature_data deben tener la misma cantidad de filas, "
                "o similarity_weights debe tener exactamente una fila para usar pesos "
                "globales por feature."
            )

        return weights_df, resolved_weight_cols

    coords_arr = np.asarray(coords)
    if knhs_weight_spec is not None:
        knhs_schema = knhs_weight_spec.schema
        k_neighbors = (
            int(knhs_weight_spec.k)
            if knhs_weight_spec.k is not None
            else int(k_neighbors)
        )
        knhs_radius_km = float(knhs_weight_spec.radius_km)
        knhs_feature_distance_mode = str(knhs_weight_spec.feature_distance_mode)
        knhs_lambda_distance = float(knhs_weight_spec.lambda_distance)
        knhs_bandwidth_mode = str(knhs_weight_spec.bandwidth_mode)
        knhs_bandwidth = float(knhs_weight_spec.bandwidth)
        knhs_row_standardize = bool(knhs_weight_spec.row_standardize)
        if knhs_weight_spec.neighbor_selector is not None:
            knhs_neighbor_selector = knhs_weight_spec.neighbor_selector

    if knhs_schema is None:
        return Kernel(
            coords_arr,
            k=int(k_neighbors),
            function="gaussian",
            fixed=False,
        )

    if feature_data is None:
        raise ValueError(
            "Para construir pesos KNHS debes proveer feature_data."
        )

    feature_cols = knhs_schema.similarity_feature_cols
    if not isinstance(feature_data, pd.DataFrame) and feature_cols is None:
        raise ValueError(
            "Si feature_data no es DataFrame, knhs_schema debe definir "
            "similarity_feature_cols explícitamente."
        )
    feature_df = _coerce_feature_dataframe_local(
        feature_data=feature_data,
        feature_cols=feature_cols,
    )
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
        raise ValueError(
            "coords debe tener shape (n, 2) para construir pesos KNHS."
        )
    if len(feature_df) != len(coords_arr):
        raise ValueError(
            "feature_data y coords deben tener la misma cantidad de filas."
        )

    lat_col = knhs_schema.lat_col
    lon_col = knhs_schema.lon_col
    knhs_df = feature_df.copy()
    knhs_df[lat_col] = coords_arr[:, 1]
    knhs_df[lon_col] = coords_arr[:, 0]

    effective_schema = knhs_schema
    if knhs_weight_spec is not None and knhs_weight_spec.similarity_weights is not None:
        weights_df, resolved_weight_cols = _coerce_similarity_weights_frame(
            similarity_weights=knhs_weight_spec.similarity_weights,
            feature_cols=feature_cols,
            weight_cols=(
                knhs_weight_spec.similarity_weight_cols
                or knhs_schema.similarity_weight_cols
            ),
            n_rows=len(feature_df),
        )
        knhs_df = pd.concat(
            [knhs_df.reset_index(drop=True), weights_df.reset_index(drop=True)],
            axis=1,
        )
        effective_schema = KNHSSchema(
            lat_col=knhs_schema.lat_col,
            lon_col=knhs_schema.lon_col,
            similarity_feature_cols=(
                list(knhs_schema.similarity_feature_cols)
                if knhs_schema.similarity_feature_cols is not None
                else None
            ),
            similarity_weight_cols=list(resolved_weight_cols),
        )

    neighbor_search = NeighborSearchCore(
        schema=effective_schema,
        radius_km=float(knhs_radius_km),
        k=int(k_neighbors),
        feature_distance_mode=knhs_feature_distance_mode,
        neighbor_selector=knhs_neighbor_selector or TopKFeatureNeighborSelector(),
    )
    prepared_data = neighbor_search.prepare(
        knhs_df,
        expected_feature_cols=list(feature_cols) if feature_cols is not None else None,
    )
    local_weight_builder = LocalWeightBuilder(
        neighbor_search=neighbor_search,
    )

    return local_weight_builder.build(
        prepared_data,
        lambda_distance=float(knhs_lambda_distance),
        kernel="gaussian",
        bandwidth_mode=knhs_bandwidth_mode,
        bandwidth=float(knhs_bandwidth),
        row_standardize=bool(knhs_row_standardize),
    )
