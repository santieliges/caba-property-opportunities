"""Generador de grafos combinando proximidad espacial y similitud de features.

La clase KNHS (KNN con Haversine + Similarity) arma un grafo dirigido donde,
para cada nodo, se buscan primero los vecinos dentro de un radio geográfico
`radius_km` usando distancia ortodrómica (haversine). Sobre ese subconjunto se
elige el top-k de vecinos más similares según distancia euclídea de sus
features. Se generan aristas `i -> j` (y opcionalmente la inversa) con
atributos [distancia_km, distancia_feature].

El resultado es un par `(edge_index, edge_attr)` listo para consumirse por los
modelos GAT/GCN de este proyecto. Opcionalmente la clase puede ajustar y
reutilizar un scaler propio para `edge_attr`, de forma de escalar train una
sola vez y aplicar exactamente la misma transformación en validación/predicción.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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
        scale_edge_features: bool = True,
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
        self.scale_edge_features = scale_edge_features
        self.edge_scaler_ = StandardScaler() if scale_edge_features else None
        self.edge_scaler_fitted_ = False

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

    def _resolve_feature_cols(self, df: pd.DataFrame) -> list[str]:
        if self.feature_cols is None:
            return [c for c in df.columns if c not in {self.lat_col, self.lon_col}]
        return list(self.feature_cols)

    def _resolve_weights(
        self,
        df: pd.DataFrame,
        feats: np.ndarray,
        feature_cols: list[str],
    ) -> np.ndarray | None:
        if self.distance != "local_weighted":
            return None

        if self.weight_cols is not None and len(self.weight_cols) != len(feature_cols):
            raise ValueError("weight_cols debe tener mismo largo que feature_cols")

        if self.weight_cols is None:
            # Fallback neutro: todos los features pesan igual.
            return np.ones_like(feats, dtype=float)

        return df[self.weight_cols].to_numpy(dtype=float)

    def fit_edge_scaler(self, edge_attr: np.ndarray):
        """Ajusta el scaler de aristas usando solo aristas de train."""

        edge_attr_arr = np.asarray(edge_attr, dtype=float)
        if edge_attr_arr.ndim != 2:
            raise ValueError(
                "edge_attr debe tener shape (E, edge_dim) para ajustar el scaler. "
                f"Recibido: {edge_attr_arr.shape}."
            )
        if edge_attr_arr.shape[0] == 0:
            raise ValueError("No se puede ajustar el scaler de aristas con 0 aristas.")
        if not self.scale_edge_features:
            return self

        self.edge_scaler_.fit(edge_attr_arr)
        self.edge_scaler_fitted_ = True
        return self

    def transform_edge_attr(self, edge_attr: np.ndarray) -> np.ndarray:
        """Transforma edge_attr con el scaler ya ajustado."""

        edge_attr_arr = np.asarray(edge_attr, dtype=float)
        if edge_attr_arr.ndim != 2:
            raise ValueError(
                "edge_attr debe tener shape (E, edge_dim) para transformar aristas. "
                f"Recibido: {edge_attr_arr.shape}."
            )
        if edge_attr_arr.shape[0] == 0:
            return edge_attr_arr.astype(np.float32, copy=False)
        if not self.scale_edge_features:
            return edge_attr_arr.astype(np.float32, copy=False)
        if not self.edge_scaler_fitted_:
            raise ValueError(
                "El scaler de aristas no está ajustado. "
                "Llama antes a fit_edge_scaler() o usa build(..., fit_edge_scaler=True)."
            )

        return self.edge_scaler_.transform(edge_attr_arr).astype(np.float32, copy=False)

    def _maybe_scale_edge_attr(
        self,
        edge_attr: np.ndarray,
        *,
        fit_edge_scaler: bool = False,
        scale_edge_attr: bool = True,
    ) -> np.ndarray:
        edge_attr_arr = np.asarray(edge_attr, dtype=float)
        if not scale_edge_attr:
            return edge_attr_arr.astype(np.float32, copy=False)
        if fit_edge_scaler:
            self.fit_edge_scaler(edge_attr_arr)
        if self.scale_edge_features and self.edge_scaler_fitted_:
            return self.transform_edge_attr(edge_attr_arr)
        return edge_attr_arr.astype(np.float32, copy=False)

    def _build_raw(self, df: pd.DataFrame):
        """Genera edge_index y edge_attr sin escalar."""

        feature_cols = self._resolve_feature_cols(df)

        coords = df[[self.lat_col, self.lon_col]].to_numpy(dtype=float)
        feats = df[feature_cols].to_numpy(dtype=float)
        weights = self._resolve_weights(df, feats, feature_cols)

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

    # --- API ---------------------------------------------------------
    def build(
        self,
        df: pd.DataFrame,
        *,
        fit_edge_scaler: bool = False,
        scale_edge_attr: bool = True,
    ):
        """Genera edge_index y edge_attr a partir de un DataFrame.

        Args:
            df: DataFrame con columnas de coordenadas y features.
        Returns:
            edge_index: np.ndarray shape (2, E) con pares (src, dst).
            edge_attr: np.ndarray shape (E, 2) con [dist_km, dist_feat].
        """
        edge_index, edge_attr = self._build_raw(df)
        edge_attr = self._maybe_scale_edge_attr(
            edge_attr,
            fit_edge_scaler=fit_edge_scaler,
            scale_edge_attr=scale_edge_attr,
        )
        return edge_index, edge_attr

    def build_cross_split(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        *,
        fit_edge_scaler_on_source: bool = False,
        scale_edge_attr: bool = True,
    ):
        """Construye un grafo combinado source+target con aristas source->target.

        Se preservan las aristas internas de `source_df` generadas por `build()`.
        Para cada nodo target, se buscan vecinos solo dentro de source según la
        misma lógica KNHS y se agregan aristas source->target.

        Returns:
            combined_df, edge_index, edge_attr, target_mask
        """

        feature_cols = self._resolve_feature_cols(source_df)

        edge_index_source, edge_attr_source = self._build_raw(source_df)

        coords_source = source_df[[self.lat_col, self.lon_col]].to_numpy(dtype=float)
        coords_target = target_df[[self.lat_col, self.lon_col]].to_numpy(dtype=float)
        feats_source = source_df[feature_cols].to_numpy(dtype=float)
        feats_target = target_df[feature_cols].to_numpy(dtype=float)
        weights_source = self._resolve_weights(source_df, feats_source, feature_cols)
        weights_target = self._resolve_weights(target_df, feats_target, feature_cols)

        lat_source = np.deg2rad(coords_source[:, 0])
        lon_source = np.deg2rad(coords_source[:, 1])
        lat_target = np.deg2rad(coords_target[:, 0])
        lon_target = np.deg2rad(coords_target[:, 1])

        src_list = [int(x) for x in edge_index_source[0]]
        dst_list = [int(x) for x in edge_index_source[1]]
        attr_list = edge_attr_source.tolist()

        target_offset = len(source_df)
        for i in range(len(target_df)):
            d_km = self._haversine_km(lat_target[i], lon_target[i], lat_source, lon_source)
            candidate_idx = np.nonzero(d_km <= self.radius_km)[0]
            if len(candidate_idx) == 0:
                continue

            if self.distance == "euclidean":
                feat_d = np.linalg.norm(feats_source[candidate_idx] - feats_target[i], axis=1)
            else:
                w_mean = 0.5 * (weights_source[candidate_idx] + weights_target[i])
                diff = feats_source[candidate_idx] - feats_target[i]
                feat_d = np.sqrt(np.sum(w_mean * diff * diff, axis=1))

            top_k = np.argsort(feat_d)[: self.k]
            neighbors = candidate_idx[top_k]
            target_idx = target_offset + i
            for j, dist_feat in zip(neighbors, feat_d[top_k]):
                src_list.append(int(j))
                dst_list.append(int(target_idx))
                attr_list.append([float(d_km[j]), float(dist_feat)])

        combined_df = pd.concat([source_df, target_df], ignore_index=True)
        edge_index = np.vstack([src_list, dst_list]).astype(np.int64)
        edge_attr = np.asarray(attr_list, dtype=np.float32)
        if fit_edge_scaler_on_source:
            self.fit_edge_scaler(edge_attr_source)
        edge_attr = self._maybe_scale_edge_attr(
            edge_attr,
            fit_edge_scaler=False,
            scale_edge_attr=scale_edge_attr,
        )
        target_mask = np.zeros(len(combined_df), dtype=bool)
        target_mask[target_offset:] = True
        return combined_df, edge_index, edge_attr, target_mask


__all__ = ["KNHS"]
