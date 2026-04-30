import numpy as np
import pandas as pd
from scipy import stats

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

        z_outliers = detector.z_test_outliers(
            y=res,
            robust=params_for_method.get("robust", False),
            w=w,
            z_threshold=params_for_method.get("z_threshold", 3.0),
            z_threshold_min=params_for_method.get("z_threshold_min")
        )

        z_scores = z_outliers["z_scores"]

        def enrich_with_z_scores(df, idx):
            enriched = df.copy()
            scores = z_scores[idx]
            enriched["z_score"] = scores
            enriched["abs_z_score"] = np.abs(scores)
            return enriched

        gdf_low_ext = enrich_with_z_scores(gdf.iloc[z_outliers["low_outliers_idx"]], z_outliers["low_outliers_idx"])
        gdf_high_ext = enrich_with_z_scores(gdf.iloc[z_outliers["high_outliers_idx"]], z_outliers["high_outliers_idx"])

        gdf_low_ext["tipo_valor_atipico"] = "BAJO"
        gdf_high_ext["tipo_valor_atipico"] = "ALTO"

        if z_outliers.get("borderline_outliers_idx") is None:
            z_df = pd.concat([gdf_low_ext, gdf_high_ext])
        else:
            gdf_low_mod = enrich_with_z_scores(
                gdf.iloc[z_outliers["borderline_low_outliers_idx"]],
                z_outliers["borderline_low_outliers_idx"]
            )
            gdf_high_mod = enrich_with_z_scores(
                gdf.iloc[z_outliers["borderline_high_outliers_idx"]],
                z_outliers["borderline_high_outliers_idx"]
            )

            gdf_low_mod["tipo_valor_atipico"] = "BAJO"
            gdf_high_mod["tipo_valor_atipico"] = "ALTO"

            gdf_low_ext["severidad_valor_atipico"] = "EXTREMO"
            gdf_high_ext["severidad_valor_atipico"] = "EXTREMO"
            gdf_low_mod["severidad_valor_atipico"] = "MODERADO"
            gdf_high_mod["severidad_valor_atipico"] = "MODERADO"

            z_df = pd.concat([gdf_low_mod, gdf_high_mod, gdf_low_ext, gdf_high_ext])

        z_df = z_df.sort_values("abs_z_score", ascending=False)

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

        # 1. Z-test robusto y espacial (cola inferior)
        z_outliers = detector.z_test_outliers(
            y=res,
            w=w,
            robust=True,  # Siempre robusto con MAD
            z_threshold=params_for_method.get("z_threshold", 3.0),
        )
        z_scores = z_outliers["z_scores"]

        # P-values para cola inferior (valores bajos atípicos)
        # Para z_score negativo (bajo), p_value = P(Z <= z_score) = norm.cdf(z_score)
        # Para z_score positivo, p_value = 1 (no atípico bajo)
        p_values_z = np.where(z_scores < 0, stats.norm.cdf(z_scores), 1.0)

        # 2. LISA con Local Moran's I
        lisa_results = detector.local_morans_I(
            y=res,
            w=w,
            coords=coords
        )
        p_values_lisa = lisa_results["p_sim"]
        quadrants = lisa_results["quadrant"]

        # 3. Score combinado
        # Factor basado en cuadrante
        factor = np.zeros(len(quadrants))
        factor[quadrants == "LL"] = 1
        factor[(quadrants == "LH") | (quadrants == "HH")] = 0
        factor[quadrants == "HL"] = -1

        # Score = (1 - p_Z) + (1 - p_LISA) * factor
        score = (1 - p_values_z) + (1 - p_values_lisa) * factor

        # Crear DataFrame con todos los valores
        gdf_combined = gdf.copy()
        gdf_combined["z_score"] = z_scores
        gdf_combined["p_value_z"] = p_values_z
        gdf_combined["p_value_lisa"] = p_values_lisa
        gdf_combined["quadrant"] = quadrants
        gdf_combined["combined_score"] = score
        gdf_combined["residuo"] = res
        gdf_combined["is_outlier"] = p_values_z < alpha  # Columna para marcar outliers

        # Guardar CSV con todos los valores y la marca de outlier
        return gdf_combined
