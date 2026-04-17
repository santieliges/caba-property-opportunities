import numpy as np
import pandas as pd


class SpatialOutlierDetector:
    def __init__(self):
        pass
    
    def z_test_outliers(
        self,
        y,
        w=None,
        z_threshold=3.0,
        z_threshold_min=None,
        min_neighbors=3,
        robust=True
    ):
        """
        Z-test for outlier detection.
        If robust=True  -> median / MAD
        If robust=False -> mean / std

        If w is provided -> spatial (local).
        If w is None     -> global (non-spatial).

        Parameters
        ----------
        z_threshold : float
            Umbral "extremo": |z| > z_threshold.
        z_threshold_min : float | None
            Si se provee, también devuelve un segundo grupo de outliers "moderados":
            z_threshold_min <= |z| <= z_threshold.
        """

        y = np.asarray(y).ravel()
        n = len(y)
        eps = 1e-8

        z_scores = np.full(n, np.nan)

        # ----------------------------
        # GLOBAL (no spatial weights)
        # ----------------------------
        if w is None:
            if robust:
                # Global robust: median / MAD (normalizado)
                loc = np.median(y)
                scale = 1.4826 * np.median(np.abs(y - loc))
            else:
                loc = np.mean(y)
                scale = np.std(y, ddof=1)

            if scale < eps:
                raise ValueError("Escala nula o casi nula")

            z_scores = (y - loc) / scale

        # ----------------------------
        # SPATIAL CASE
        # ----------------------------
        else:

            # Iterar SOLO sobre nodos existentes en w
            for i in w.neighbors.keys():

                neigh_ids = w.neighbors.get(i, [])
                neigh_weights = w.weights.get(i, [])

                neighs = []
                weights = []

                for j, w_ij in zip(neigh_ids, neigh_weights):
                    if j == i:
                        continue
                    neighs.append(j)
                    weights.append(w_ij)

                if len(neighs) < min_neighbors:
                    continue

                local_vals = y[neighs]
                local_w = np.asarray(weights)

                if robust:
                    loc = self.weighted_median(local_vals, local_w)
                    scale = self.weighted_mad(local_vals, local_w)
                else:
                    loc = np.average(local_vals, weights=local_w)
                    scale = np.sqrt(
                        np.average((local_vals - loc) ** 2, weights=local_w)
                    )

                if not np.isfinite(loc) or scale < eps:
                    continue

                z_scores[i] = (y[i] - loc) / scale

        # --------------------------------
        # Outlier detection
        # --------------------------------
        if z_threshold_min is not None and z_threshold_min >= z_threshold:
            raise ValueError("z_threshold_min debe ser menor que z_threshold")

        abs_z = np.abs(z_scores)
        outlier_mask = abs_z > z_threshold

        borderline_outliers_mask = None
        borderline_outliers_idx = None
        borderline_high_outliers_mask = None
        borderline_high_outliers_idx = None
        borderline_low_outliers_mask = None
        borderline_low_outliers_idx = None

        if z_threshold_min is not None:
            borderline_outliers_mask = (abs_z >= z_threshold_min) & (abs_z <= z_threshold)
            borderline_outliers_idx = np.where(borderline_outliers_mask)[0]
            borderline_high_outliers_mask = (z_scores >= z_threshold_min) & (z_scores <= z_threshold)
            borderline_high_outliers_idx = np.where(borderline_high_outliers_mask)[0]
            borderline_low_outliers_mask = (z_scores <= -z_threshold_min) & (z_scores >= -z_threshold)
            borderline_low_outliers_idx = np.where(borderline_low_outliers_mask)[0]

        return {
            "z_scores": z_scores,
            "outlier_mask": outlier_mask,
            "outlier_idx": np.where(outlier_mask)[0],
            "inlier_mask": ~outlier_mask,
            "high_outliers_mask": z_scores > z_threshold,
            "high_outliers_idx": np.where(z_scores > z_threshold)[0],
            "low_outliers_mask": z_scores < -z_threshold,
            "low_outliers_idx": np.where(z_scores < -z_threshold)[0],
            "borderline_outliers_mask": borderline_outliers_mask,
            "borderline_outliers_idx": borderline_outliers_idx,
            "borderline_high_outliers_mask": borderline_high_outliers_mask,
            "borderline_high_outliers_idx": borderline_high_outliers_idx,
            "borderline_low_outliers_mask": borderline_low_outliers_mask,
            "borderline_low_outliers_idx": borderline_low_outliers_idx,
        }


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

    def consensus_voting(
        self,
        dfs,
        method_names,
        id_col,
        min_votes=2
    ):
        """
        Consensus voting over multiple outlier detectors.

        Parameters
        ----------
        dfs : list[pd.DataFrame]
            List of DataFrames, each containing detected outliers.
        method_names : list[str]
            Names of the methods (same order as dfs).
        id_col : str
            Column used as unique identifier.
        min_votes : int
            Minimum number of votes required to keep an observation.

        Returns
        -------
        votes_df : pd.DataFrame
            Binary vote matrix + vote count.
        consensus_df : pd.DataFrame
            Subset of observations with votes >= min_votes.
        """

        if len(dfs) != len(method_names):
            raise ValueError("dfs and method_names must have the same length")
        #concat de todos los ids de todos los dfs
        all_ids = pd.Index(
            pd.concat([df[id_col] for df in dfs]).unique(),
            name=id_col
        )
        votes_df = pd.DataFrame(index=all_ids)

        #del conjunto de ids, marcamos de que metodo vienen
        for df, name in zip(dfs, method_names):
            votes_df[name] = votes_df.index.isin(df[id_col]).astype(int)

        #sumamos por id la cantidad de apareiciones en todos los métodos y filtramo spor min_votes
        votes_df["n_votes"] = votes_df.sum(axis=1)  
        consensus_df = votes_df[votes_df["n_votes"] >= min_votes].copy()

        return votes_df.reset_index(), consensus_df.reset_index()

    
    ### helpers ###


    def LOO_median(self, y):
        """
        Leave-One-Out median for each observation.
        """
        y = np.asarray(y).ravel()
        n = len(y)

        loo_median = np.empty(n)

        for i in range(n):
            y_loo = np.delete(y, i)
            loo_median[i] = np.median(y_loo)

        return loo_median

    def LOO_mad(self, y, normalize=True):
        """
        Leave-One-Out MAD for each observation.
        """
        y = np.asarray(y).ravel()
        n = len(y)

        loo_mad = np.empty(n)

        for i in range(n):
            y_loo = np.delete(y, i)
            med = np.median(y_loo)
            mad = np.median(np.abs(y_loo - med))

            if normalize:
                mad *= 1.4826  # consistency for Normal

            loo_mad[i] = mad

        return loo_mad

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
        
    
