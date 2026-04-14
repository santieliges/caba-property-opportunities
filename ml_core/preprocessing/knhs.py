"""Generador de grafos combinando proximidad espacial y similitud de features.

La clase KNHS (KNN con Haversine + Similarity) arma un grafo dirigido donde,
para cada nodo, se buscan primero los vecinos dentro de un radio geográfico
`radius_km` usando distancia ortodrómica (haversine). Sobre ese subconjunto se
elige el top-k de vecinos más similares según distancia euclídea de sus
features. Se generan aristas `i -> j` (y opcionalmente la inversa) con
atributos [distancia_km, distancia_feature].

El resultado es un par `(edge_index, edge_attr)` listo para consumirse por los
modelos GAT/GCN de este proyecto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class KNHS:
    def __init__(
        self,
        lat_col: str = "lat",
        lon_col: str = "lon",
        feature_cols: list[str] | None = None,
        weight_cols: list[str] | None = None,
        radius_km: float = 2.0,
        k: int = 5,
        distance: str = "euclidean",  # "euclidean" | "local_weighted"
        add_reverse: bool = True,
    ):
        if radius_km <= 0:
            raise ValueError("radius_km debe ser > 0")
        if k <= 0:
            raise ValueError("k debe ser > 0")
        if distance not in {"euclidean", "local_weighted"}:
            raise ValueError("distance debe ser 'euclidean' o 'local_weighted'")

        self.lat_col = lat_col
        self.lon_col = lon_col
        self.feature_cols = feature_cols
        self.weight_cols = weight_cols
        self.radius_km = radius_km
        self.k = k
        self.distance = distance
        self.add_reverse = add_reverse

    # --- helpers -----------------------------------------------------
    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
        """Devuelve distancia ortodrómica en km entre pares de puntos."""
        R = 6371.0  # radio medio de la Tierra en km
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    # --- API ---------------------------------------------------------
    def build(self, df: pd.DataFrame):
        """Genera edge_index y edge_attr a partir de un DataFrame.

        Args:
            df: DataFrame con columnas de coordenadas y features.
        Returns:
            edge_index: np.ndarray shape (2, E) con pares (src, dst).
            edge_attr: np.ndarray shape (E, 2) con [dist_km, dist_feat].
        """

        if self.feature_cols is None:
            # todas excepto coord
            self.feature_cols = [c for c in df.columns if c not in {self.lat_col, self.lon_col}]

        if self.distance == "local_weighted":
            if self.weight_cols is None:
                raise ValueError("Para distance='local_weighted' debes indicar weight_cols")
            if len(self.weight_cols) != len(self.feature_cols):
                raise ValueError("weight_cols debe tener mismo largo que feature_cols")

        coords = df[[self.lat_col, self.lon_col]].to_numpy(dtype=float)
        feats = df[self.feature_cols].to_numpy(dtype=float)

        if self.distance == "local_weighted":
            weights = df[self.weight_cols].to_numpy(dtype=float)
        else:
            weights = None

        lat_rad = np.deg2rad(coords[:, 0])
        lon_rad = np.deg2rad(coords[:, 1])

        n = len(df)
        src_list = []
        dst_list = []
        attr_list = []

        for i in range(n):
            # distancias espaciales desde i al resto
            d_km = self._haversine_km(lat_rad[i], lon_rad[i], lat_rad, lon_rad)
            mask = (d_km <= self.radius_km) & (np.arange(n) != i)

            if not np.any(mask):
                continue  # sin vecinos dentro del radio

            candidates_idx = np.nonzero(mask)[0]
            # distancia de features
            if self.distance == "euclidean":
                feat_d = np.linalg.norm(feats[candidates_idx] - feats[i], axis=1)
            else:  # local_weighted
                w_mean = 0.5 * (weights[candidates_idx] + weights[i])  # [m, d]
                diff = feats[candidates_idx] - feats[i]
                feat_d = np.sqrt(np.sum(w_mean * diff * diff, axis=1))

            top_k = np.argsort(feat_d)[: self.k]
            neighbors = candidates_idx[top_k]

            for j, dist_feat in zip(neighbors, feat_d[top_k]):
                src_list.append(i)
                dst_list.append(j)
                attr_list.append([float(d_km[j]), float(dist_feat)])

                if self.add_reverse:
                    src_list.append(j)
                    dst_list.append(i)
                    attr_list.append([float(d_km[j]), float(dist_feat)])

        if not src_list:
            raise ValueError("No se generaron aristas; revisa radius_km/k o las coordenadas")

        edge_index = np.vstack([src_list, dst_list]).astype(np.int64)
        edge_attr = np.asarray(attr_list, dtype=np.float32)
        return edge_index, edge_attr


__all__ = ["KNHS"]
