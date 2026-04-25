import numpy as np
import pandas as pd

class Strategy:

    def __init__(self, name):
        self.name = name

    def run(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")
    
class NegativeResidualsStrategy(Strategy):

    def __init__(self):
        super().__init__("negative")

    def run(self, res, gdf, output_dir, model_name, **kwargs):

        gdf_neg = (
            gdf.loc[res < 0]
            .assign(residuo=res[res < 0])
            .sort_values("residuo", ascending=True)
        )

        gdf_neg.to_csv(
            f"{output_dir}/residuos_negativos_{model_name}.csv",
            index=False
        )

        return gdf_neg

class ZTestStrategy(Strategy):

    def __init__(self):
        super().__init__("ztest")

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

        z_df.to_csv(
            f"{output_dir}/outliers_ztest_{model_name}.csv",
            index=False
        )

        return z_df

class QuantileStrategy(Strategy):

    def __init__(self):
        super().__init__("quantile")

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

        gdf_quant.to_csv(
            f"{output_dir}/outliers_quantile_{model_name}.csv",
            index=False
        )

        return gdf_quant

class LISAStrategy(Strategy):

    def __init__(self):
        super().__init__("lisa")

    def run(self, res, gdf, detector, coords, w, output_dir, model_name, **kwargs):

        lisa_outliers = detector.local_morans_I(
            y=res,
            w=w,
            coords=coords
        )

        spatial_lh = np.where(lisa_outliers["quadrant"] == "LH")[0]

        gdf_lisa_lh = gdf.iloc[spatial_lh]

        gdf_lisa_lh.to_csv(
            f"{output_dir}/outliers_LISA_{model_name}.csv",
            index=False
        )

        return gdf_lisa_lh
