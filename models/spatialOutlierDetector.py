import numpy as np


class SpatialOutlierDetector:
    def __init__(self):
        pass
    
    def z_test_spatial_outliers(
        self,
        X,
        y,
        coords,
        w,
        z_threshold=3.0,
        return_coords=True
    ):

        y = np.asarray(y).ravel()
        coords = np.asarray(coords)

        n = len(y)
        res_loc_std = np.zeros(n)

        for i in range(n):
            neighs = w.neighbors[i]
            local_vals = y[neighs]

            med = self.weighted_median(local_vals, w.weights[i])
            sigma = self.weighted_mad(local_vals, w.weights[i])

            res_loc_std[i] = (y[i] - med) / sigma if sigma > 0 else 0

        outlier_mask = np.abs(res_loc_std) > z_threshold
        outlier_idx = np.where(outlier_mask)[0]

        result = {
            "z_scores": res_loc_std,
            "outlier_idx": outlier_idx,
            "outlier_mask": outlier_mask,
        }

        if return_coords:
            result["outlier_coords"] = coords[outlier_idx]

        return result

    
    ### helpers ###

    def weighted_median(self, x, w):
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

    def weighted_mad(self, x, w, normalize=True):
        x = np.asarray(x)
        w = np.asarray(w)

        mask = (w > 0) & np.isfinite(x)
        x = x[mask]
        w = w[mask]

        if len(x) == 0:
            return np.nan

        med = self.weighted_median(x, w)
        mad = self.weighted_median(np.abs(x - med), w)

        if normalize:
            return 1.4826 * mad
        else:
            return mad
