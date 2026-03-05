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

    def run(self, res, gdf, detector, w, robust, output_dir, model_name, **kwargs):

        z_outliers = detector.z_test_outliers(
            y=res,
            robust=robust,
            w=w
        )

        gdf_low = gdf.iloc[z_outliers["low_outliers_idx"]].copy()
        gdf_high = gdf.iloc[z_outliers["high_outliers_idx"]].copy()

        gdf_low["tipo_valor_atipico"] = "BAJO"
        gdf_high["tipo_valor_atipico"] = "ALTO"

        z_df = pd.concat([gdf_low, gdf_high])

        z_df.to_csv(
            f"{output_dir}/outliers_ztest_{model_name}.csv",
            index=False
        )

        return z_df

class QuantileStrategy(Strategy):

    def __init__(self):
        super().__init__("quantile")

    def run(self, res, gdf, detector, coords, lower_q, upper_q, output_dir, model_name, **kwargs):

        quantile_outliers = detector.quantile_outliers(
            y=res,
            coords=coords,
            lower_q=lower_q,
            upper_q=upper_q
        )

        gdf_quant = gdf.iloc[quantile_outliers["outlier_idx"]]

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