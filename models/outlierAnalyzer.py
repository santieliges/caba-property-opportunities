import numpy as np
import geopandas as gpd
from esda.moran import Moran_Local
from libpysal.weights import Kernel

import numpy as np

def weighted_median(x, w):
    x = np.asarray(x)
    w = np.asarray(w)

    mask = (w > 0) & np.isfinite(x)
    x = x[mask]
    w = w[mask]

    if len(x) == 0:
        return np.nan

    idx = np.argsort(x)
    x_sorted = x[idx]
    w_sorted = w[idx]

    cum_w = np.cumsum(w_sorted)
    cutoff = 0.5 * np.sum(w_sorted)

    return x_sorted[np.searchsorted(cum_w, cutoff)]

def weighted_mad(x, w, normalize=True):
    x = np.asarray(x)
    w = np.asarray(w)

    mask = (w > 0) & np.isfinite(x)
    x = x[mask]
    w = w[mask]

    if len(x) == 0:
        return np.nan

    med = weighted_median(x, w)
    mad = weighted_median(np.abs(x - med), w)

    if normalize:
        return 1.4826 * mad
    else:
        return mad


class SpatialOutlierAnalyzer:
    def __init__(
        self,
        k=8,
        kernel="gaussian",
        fixed=False,
        alpha=0.05
    ):
        self.k = k
        self.kernel = kernel
        self.fixed = fixed
        self.alpha = alpha

        self.w_ = None
        self.residuals_ = None
        self.lisa_ = None
        self.res_loc_z_ = None

    def compute_residuals(self, model, gdf, features, target):
        coords = np.column_stack([
            gdf.geometry.x,
            gdf.geometry.y
        ])

        y_true = gdf[target].values
        y_pred = model.predict(gdf, features_vars=features)

        self.residuals_ = y_true - y_pred
        return self.residuals_

    def build_weights(self, gdf):
        coords = np.column_stack([
            gdf.geometry.x,
            gdf.geometry.y
        ])

        self.w_ = Kernel(
            coords,
            k=self.k,
            function=self.kernel,
            fixed=self.fixed
        )
        return self.w_

    def lisa_outliers(self):
        if self.residuals_ is None or self.w_ is None:
            raise RuntimeError("Run compute_residuals and build_weights first")

        self.lisa_ = Moran_Local(self.residuals_, self.w_)

        outliers = (
            (self.lisa_.q == 3) &  # Low-High
            (self.lisa_.p_sim < self.alpha)
        )

        return outliers
    
    def robust_local_z(self):
        if self.residuals_ is None or self.w_ is None:
            raise RuntimeError("Run compute_residuals and build_weights first")

        res = self.residuals_
        z = np.zeros_like(res)

        for i in range(len(res)):
            neighs = self.w_.neighbors[i]
            weights = self.w_.weights[i]

            local_vals = res[neighs]

            med = weighted_median(local_vals, weights)
            sigma = weighted_mad(local_vals, weights)

            z[i] = (res[i] - med) / sigma if sigma > 0 else 0

        self.res_loc_z_ = z
        return z

    def analyze(self, model, gdf, features, target):
        self.compute_residuals(model, gdf, features, target)
        self.build_weights(gdf)

        lisa_mask = self.lisa_outliers()
        z_scores = self.robust_local_z()

        return {
            "lisa_outliers": lisa_mask,
            "robust_z": z_scores
        }
    