import numpy as np


class SpatialOutlierDetector:
    def __init__(self):
        pass
    
    def z_test_outliers(
        self,
        y,
        w=None,
        z_threshold=3.0,
        min_neighbors=3
    ):
        """
        Robust z-test for outlier detection.
        If w is provided -> spatial (local).
        If w is None     -> global (non-spatial).
        """

        y = np.asarray(y).ravel()
        n = len(y)
        eps = 1e-8

        z_scores = np.full(n, np.nan)

        # ─────────────────────────────────────
        # 🔹 GLOBAL (NO ESPACIAL)
        # ─────────────────────────────────────
        if w is None:
            med = np.median(y)
            mad = np.median(np.abs(y - med))

            if mad < eps:
                raise ValueError("Global MAD is zero")

            z_scores = (y - med) / mad

        # ─────────────────────────────────────
        # 🔹 LOCAL (ESPACIAL)
        # ─────────────────────────────────────
        else:
            for i in range(n):
                neighs = []
                weights = []

                for j, w_ij in zip(w.neighbors[i], w.weights[i]):
                    if j == i:
                        continue
                    neighs.append(j)
                    weights.append(w_ij)

                if len(neighs) < min_neighbors:
                    continue

                local_vals = y[neighs]
                local_w = np.asarray(weights)

                med = self.weighted_median(local_vals, local_w)
                mad = self.weighted_mad(local_vals, local_w)

                if not np.isfinite(med) or mad < eps:
                    continue

                z_scores[i] = (y[i] - med) / mad

        outlier_mask = np.abs(z_scores) > z_threshold

        return {
            "z_scores": z_scores,
            "outlier_mask": outlier_mask,
            "outlier_idx": np.where(outlier_mask)[0],
            "high_outliers_mask": z_scores > z_threshold,
            "high_outliers_idx": np.where(z_scores > z_threshold)[0],
            "low_outliers_mask": z_scores < -z_threshold,
            "low_outliers_idx": np.where(z_scores < -z_threshold)[0],
        }



    def quantile_outliers(
        self,
        y,
        coords=None,
        lower_q=0.01,
        upper_q=0.99,
        return_coords=True
    ):
        """
        Detecta outliers usando cuantiles globales.

        Parameters
        ----------
        y : array-like
            Variable de interés
        coords : array-like, optional
            Coordenadas espaciales
        lower_q : float
            Cuantil inferior
        upper_q : float
            Cuantil superior
        return_coords : bool
            Si devuelve las coordenadas de los outliers

        Returns
        -------
        dict
        """

        y = np.asarray(y).ravel()
        n = len(y)

        q_low = np.quantile(y, lower_q)
        q_high = np.quantile(y, upper_q)

        outlier_mask = (y < q_low) | (y > q_high)
        outlier_idx = np.where(outlier_mask)[0]

        result = {
            "q_low": q_low,
            "q_high": q_high,
            "outlier_idx": outlier_idx,
            "outlier_mask": outlier_mask,
        }

        if return_coords and coords is not None:
            coords = np.asarray(coords)
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
        
    def local_morans_I(
        self,
        y,
        w,
        coords=None,
        permutations=0,
        return_coords=True
    ):
        """
        Local Moran's I (LISA).

        Parameters
        ----------
        y : array-like
            Variable de interés
        w : spatial weights object
        coords : array-like, optional
            Coordenadas
        permutations : int
            Si > 0, calcula p-values por permutación
        return_coords : bool

        Returns
        -------
        dict
        """

        y = np.asarray(y).ravel()
        n = len(y)

        z = (y - y.mean()) / y.std(ddof=1)

        I = np.zeros(n)

        for i in range(n):
            neighs = w.neighbors[i]
            weights = w.weights[i]

            I[i] = z[i] * np.sum(weights * z[neighs])

        result = {
            "local_I": I,
            "z": z
        }

        # Clasificación HH, LL, HL, LH
        quad = np.empty(n, dtype=object)
        quad[(z > 0) & (I > 0)] = "HH"
        quad[(z < 0) & (I > 0)] = "LL"
        quad[(z > 0) & (I < 0)] = "HL"
        quad[(z < 0) & (I < 0)] = "LH"

        result["quadrant"] = quad

        # Permutaciones (opcional)
        if permutations > 0:
            pvals = np.zeros(n)

            for i in range(n):
                sims = np.zeros(permutations)
                neighs = w.neighbors[i]
                weights = w.weights[i]

                for k in range(permutations):
                    perm_z = np.random.permutation(z)
                    sims[k] = z[i] * np.sum(weights * perm_z[neighs])

                pvals[i] = (np.sum(np.abs(sims) >= abs(I[i])) + 1) / (permutations + 1)

            result["p_values"] = pvals

        if return_coords and coords is not None:
            coords = np.asarray(coords)
            result["coords"] = coords

        return result
