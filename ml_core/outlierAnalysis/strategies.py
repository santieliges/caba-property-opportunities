import numpy as np
import pandas as pd


def _maybe_add_price_columns(df, y_true=None, y_pred=None):
    if "precio" not in df.columns or y_pred is None or y_true is None:
        return

    precio_obs = pd.to_numeric(df["precio"], errors="coerce").to_numpy(dtype=float)
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)

    valid = (
        np.isfinite(precio_obs)
        & (precio_obs > 0)
        & np.isfinite(y_true_arr)
        & np.isfinite(y_pred_arr)
    )
    if valid.sum() == 0:
        return

    # Heurística simple: si y_true está mucho más cerca de log(precio) que de
    # precio crudo, asumimos que el target del modelo está en escala log.
    log_diff = np.nanmedian(np.abs(y_true_arr[valid] - np.log(precio_obs[valid])))
    raw_diff = np.nanmedian(np.abs(y_true_arr[valid] - precio_obs[valid]))

    if np.isfinite(log_diff) and np.isfinite(raw_diff) and log_diff < raw_diff:
        df["precio_estimado"] = np.exp(y_pred_arr)
        df["precio_observado_modelo"] = np.exp(y_true_arr)


class Strategy:

    def __init__(self, name, requires_weights=False):
        self.name = name
        self.requires_weights = requires_weights

    def run(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")
    
class NegativeResidualsStrategy(Strategy):

    def __init__(self):
        super().__init__("negative", requires_weights=False)

    def run(self, res, gdf, output_dir, model_name, **kwargs):

        gdf_neg = (
            gdf.loc[res < 0]
            .assign(residuo=res[res < 0])
            .sort_values("residuo", ascending=True)
        )

        return gdf_neg

class ZTestStrategy(Strategy):

    def __init__(self):
        super().__init__("ztest", requires_weights=True)

    def run(self, res, gdf, detector, w, params_for_method, output_dir, model_name, **kwargs):
        alpha = params_for_method.get("alpha", 0.05)
        tail = params_for_method.get("tail", "lower")

        def set_positional_values(df, positions, columns, values):
            pos_arr = np.asarray(positions, dtype=int).reshape(-1)
            if pos_arr.size == 0:
                return
            col_list = [columns] if isinstance(columns, str) else list(columns)
            col_idx = [df.columns.get_loc(col) for col in col_list]
            if len(col_idx) == 1:
                df.iloc[pos_arr, col_idx[0]] = values
            else:
                df.iloc[pos_arr, col_idx] = values

        z_outliers = detector.z_test_outliers(
            y=res,
            robust=params_for_method.get("robust", False),
            w=w,
            z_threshold=params_for_method.get("z_threshold", 3.0),
            z_threshold_min=params_for_method.get("z_threshold_min"),
            tail=tail,
        )

        z_scores = z_outliers["z_scores"]
        p_values = z_outliers["p_values"]
        y_true = kwargs.get("y_true")
        y_pred = kwargs.get("y_pred")

        z_df = gdf.copy()
        z_df["residuo"] = np.asarray(res).reshape(-1)
        z_df["z_score"] = z_scores
        z_df["abs_z_score"] = np.abs(z_scores)
        z_df["p_value"] = p_values
        z_df["p_value_z"] = p_values
        z_df["is_outlier"] = np.isfinite(p_values) & (p_values < alpha)
        z_df["tipo_valor_atipico"] = "NO_ATIPICO"
        z_df["severidad_valor_atipico"] = "NO_ATIPICO"

        if y_true is not None:
            z_df["valor_observado"] = np.asarray(y_true).reshape(-1)
        if y_pred is not None:
            pred_values = np.asarray(y_pred).reshape(-1)
            z_df["valor_predicho"] = pred_values
            z_df["valor_esperado"] = pred_values
        _maybe_add_price_columns(z_df, y_true=y_true, y_pred=y_pred)

        if tail in {"two-sided", "lower"}:
            set_positional_values(
                z_df,
                z_outliers["low_outliers_idx"],
                "tipo_valor_atipico",
                "BAJO",
            )
        if tail in {"two-sided", "upper"}:
            set_positional_values(
                z_df,
                z_outliers["high_outliers_idx"],
                "tipo_valor_atipico",
                "ALTO",
            )

        if z_outliers.get("borderline_low_outliers_idx") is not None:
            set_positional_values(
                z_df,
                z_outliers["borderline_low_outliers_idx"],
                ["tipo_valor_atipico", "severidad_valor_atipico"],
                ["BAJO", "MODERADO"],
            )
        if z_outliers.get("borderline_high_outliers_idx") is not None:
            set_positional_values(
                z_df,
                z_outliers["borderline_high_outliers_idx"],
                ["tipo_valor_atipico", "severidad_valor_atipico"],
                ["ALTO", "MODERADO"],
            )

        if tail in {"two-sided", "lower"}:
            set_positional_values(
                z_df,
                z_outliers["low_outliers_idx"],
                ["tipo_valor_atipico", "severidad_valor_atipico"],
                ["BAJO", "EXTREMO"],
            )
        if tail in {"two-sided", "upper"}:
            set_positional_values(
                z_df,
                z_outliers["high_outliers_idx"],
                ["tipo_valor_atipico", "severidad_valor_atipico"],
                ["ALTO", "EXTREMO"],
            )

        if tail == "lower":
            z_df.loc[z_df["is_outlier"], "tipo_valor_atipico"] = "BAJO"
        elif tail == "upper":
            z_df.loc[z_df["is_outlier"], "tipo_valor_atipico"] = "ALTO"
        else:
            negative_sig = z_df["is_outlier"] & (z_df["z_score"] < 0)
            positive_sig = z_df["is_outlier"] & (z_df["z_score"] > 0)
            z_df.loc[negative_sig, "tipo_valor_atipico"] = "BAJO"
            z_df.loc[positive_sig, "tipo_valor_atipico"] = "ALTO"

        z_df.loc[
            z_df["is_outlier"] & (z_df["severidad_valor_atipico"] == "NO_ATIPICO"),
            "severidad_valor_atipico",
        ] = "SIGNIFICATIVO"

        z_df = z_df.sort_values(
            ["is_outlier", "p_value", "abs_z_score"],
            ascending=[False, True, False],
            na_position="last",
        )

        return z_df

class QuantileStrategy(Strategy):

    def __init__(self):
        super().__init__("quantile", requires_weights=False)

    def run(self, res, gdf, detector, coords, output_dir, model_name, params_for_method, **kwargs):

        res = np.asarray(res).reshape(-1)
        quantile_outliers = detector.quantile_outliers(
            y=res,
            coords=coords,
            lower_q=params_for_method.get("lower_q", 0.05),
            upper_q=params_for_method.get("upper_q", 0.95)
        )

        outlier_idx = quantile_outliers["outlier_idx"]
        gdf_quant = gdf.iloc[outlier_idx].copy()

        if not gdf_quant.empty:
            residual_ranks = pd.Series(res).rank(method="average", pct=True).to_numpy()
            outlier_percentiles = residual_ranks[outlier_idx]

            gdf_quant["residuo"] = res[outlier_idx]
            gdf_quant["tipo_valor_atipico"] = np.where(
                gdf_quant["residuo"] < quantile_outliers["q_low"],
                "BAJO",
                "ALTO",
            )
            gdf_quant["quantile_percentil"] = outlier_percentiles
            gdf_quant["distancia_percentil_50"] = np.abs(outlier_percentiles - 0.5)
            gdf_quant["outlier_score"] = gdf_quant["distancia_percentil_50"]
            gdf_quant["ranking_outlier"] = (
                gdf_quant["outlier_score"]
                .rank(method="dense", ascending=False)
                .astype(int)
            )
            gdf_quant = gdf_quant.sort_values("outlier_score", ascending=False)

        return gdf_quant

class LISAStrategy(Strategy):

    def __init__(self):
        super().__init__("lisa", requires_weights=True)

    def run(self, res, gdf, detector, coords, w, output_dir, model_name, **kwargs):

        lisa_outliers = detector.local_morans_I(
            y=res,
            w=w,
            coords=coords
        )

        spatial_lh = np.where(lisa_outliers["quadrant"] == "LH")[0]

        gdf_lisa_lh = gdf.iloc[spatial_lh]

        return gdf_lisa_lh


class CombinedZLisaStrategy(Strategy):

    def __init__(self):
        super().__init__("combined_z_lisa", requires_weights=True)

    def run(self, res, gdf, detector, w, coords, output_dir, model_name, params_for_method, **kwargs):
        alpha = params_for_method.get("alpha", 0.05)  # Nivel de significancia
        permutations = params_for_method.get("permutations", 999)
        tail = params_for_method.get("tail", "lower")

        # 1. Z-test robusto y espacial (cola inferior)
        z_outliers = detector.z_test_outliers(
            y=res,
            w=w,
            robust=True,  # Siempre robusto con MAD
            z_threshold=params_for_method.get("z_threshold", 3.0),
            tail=tail,
        )
        z_scores = z_outliers["z_scores"]
        p_values_z = z_outliers["p_values"]

        # 2. LISA con Local Moran's I
        lisa_results = detector.local_morans_I(
            y=res,
            w=w,
            coords=coords,
            permutations=permutations,
        )
        p_values_lisa = lisa_results.get("p_sim")
        if p_values_lisa is None:
            p_values_lisa = lisa_results.get("p_values")
        if p_values_lisa is None:
            raise ValueError(
                "local_morans_I no devolvio p-values. "
                "Verifica que permutations sea mayor a 0."
            )
        quadrants = lisa_results["quadrant"]

        # 3. Score principal: solo evidencia del z-test para cola inferior.
        z_evidence = np.where(z_scores < 0, 1 - p_values_z, 0.0)
        score = z_evidence

        # Crear DataFrame con todos los valores
        gdf_combined = gdf.copy()
        gdf_combined["z_score"] = z_scores
        gdf_combined["p_value_z"] = p_values_z
        gdf_combined["p_value_lisa"] = p_values_lisa
        gdf_combined["quadrant"] = quadrants
        gdf_combined["combined_score"] = score
        gdf_combined["residuo"] = res
        gdf_combined["is_outlier"] = np.isfinite(p_values_z) & (p_values_z < alpha)
        y_true = kwargs.get("y_true")
        y_pred = kwargs.get("y_pred")
        if y_true is not None:
            gdf_combined["valor_observado"] = np.asarray(y_true).reshape(-1)
        if y_pred is not None:
            pred_values = np.asarray(y_pred).reshape(-1)
            gdf_combined["valor_predicho"] = pred_values
            gdf_combined["valor_esperado"] = pred_values
        _maybe_add_price_columns(gdf_combined, y_true=y_true, y_pred=y_pred)

        # Guardar CSV con todos los valores y la marca de outlier
        return gdf_combined
