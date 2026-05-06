"""Generador de vecindad espacial + similitud para grafos y pesos locales.

El módulo separa tres responsabilidades:

- `KNHSSchema`: define de forma explícita qué columnas representan
  coordenadas, features de similitud y pesos de similitud.
- `NeighborGraphBuilder`: construye `edge_index` y `edge_attr` para grafos.
- `LocalWeightBuilder`: construye pesos locales compatibles con el detector
  espacial a partir de la misma vecindad.

La clase `KNHS` se mantiene como fachada backward-compatible, pero ahora delega el trabajo interno a estas piezas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass(frozen=True)
class KNHSSchema:
    """Schema explícito de columnas usadas por KNHS."""

    lat_col: str = "lat"
    lon_col: str = "lon"
    similarity_feature_cols: list[str] | None = None
    similarity_weight_cols: list[str] | None = None

    def resolve_similarity_feature_cols(
        self,
        df: pd.DataFrame,
        *,
        expected_cols: list[str] | None = None,
    ) -> list[str]:
        if expected_cols is not None:
            missing = [col for col in expected_cols if col not in df.columns]
            if missing:
                raise ValueError(
                    "Al DataFrame le faltan columnas de similitud esperadas: "
                    f"{missing}"
                )
            return list(expected_cols)

        if self.similarity_feature_cols is None:
            return [
                col
                for col in df.columns
                if col not in {self.lat_col, self.lon_col}
            ]

        missing = [col for col in self.similarity_feature_cols if col not in df.columns]
        if missing:
            raise ValueError(
                "Al DataFrame le faltan columnas de similitud configuradas: "
                f"{missing}"
            )
        return list(self.similarity_feature_cols)

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        *,
        expected_feature_cols: list[str] | None = None,
    ) -> list[str]:
        required = [self.lat_col, self.lon_col]
        missing_required = [col for col in required if col not in df.columns]
        if missing_required:
            raise ValueError(
                "Al DataFrame le faltan columnas de coordenadas requeridas: "
                f"{missing_required}"
            )

        feature_cols = self.resolve_similarity_feature_cols(
            df,
            expected_cols=expected_feature_cols,
        )
        if self.similarity_weight_cols is not None:
            missing_weights = [
                col for col in self.similarity_weight_cols if col not in df.columns
            ]
            if missing_weights:
                raise ValueError(
                    "Al DataFrame le faltan columnas de pesos de similitud: "
                    f"{missing_weights}"
                )
            if len(self.similarity_weight_cols) != len(feature_cols):
                raise ValueError(
                    "similarity_weight_cols debe tener el mismo largo que "
                    "similarity_feature_cols."
                )

        return feature_cols

    def extract_coords_deg(self, df: pd.DataFrame) -> np.ndarray:
        return df[[self.lat_col, self.lon_col]].to_numpy(dtype=float)

    def extract_similarity_features(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> np.ndarray:
        return df[feature_cols].to_numpy(dtype=float)

    def extract_similarity_weights(
        self,
        df: pd.DataFrame,
        feats: np.ndarray,
        feature_cols: list[str],
        *,
        feature_distance_mode: str,
    ) -> np.ndarray | None:
        if feature_distance_mode != "local_weighted":
            return None
        if self.similarity_weight_cols is None:
            return np.ones_like(feats, dtype=float)
        if len(self.similarity_weight_cols) != len(feature_cols):
            raise ValueError(
                "similarity_weight_cols debe tener el mismo largo que "
                "similarity_feature_cols."
            )
        return df[self.similarity_weight_cols].to_numpy(dtype=float)


@dataclass(frozen=True)
class KNHSWeightSpec:
    """Config explícita para construir pesos espaciales KNHS.

    Esta especificación separa:
    - el schema de columnas de similitud/coordenadas (`schema`)
    - los hiperparámetros del vecindario/pesos
    - los pesos de similitud opcionales, que pueden venir por fuera del DataFrame
      principal del modelo.
    """

    schema: KNHSSchema
    radius_km: float = 2.0
    k: int | None = None
    feature_distance_mode: str = "euclidean"
    lambda_distance: float = 0.5
    bandwidth_mode: str = "adaptive"
    bandwidth: float = 1.0
    row_standardize: bool = True
    neighbor_selector: NeighborSelector | None = None
    similarity_weights: pd.DataFrame | np.ndarray | None = None
    similarity_weight_cols: list[str] | None = None


@dataclass
class PreparedKNHSData:
    """Representación preparada y consistente de un dataset para KNHS."""

    schema: KNHSSchema
    similarity_feature_cols: list[str]
    coords_deg: np.ndarray
    lat_rad: np.ndarray
    lon_rad: np.ndarray
    similarity_features: np.ndarray
    similarity_weights: np.ndarray | None

    @property
    def n_rows(self) -> int:
        return int(self.coords_deg.shape[0])


class PreparedKNHS:
    """Contexto preparado de KNHS para reutilizar grafo y pesos locales."""

    def __init__(
        self,
        *,
        knhs: KNHS,
        df: pd.DataFrame,
        prepared_data: PreparedKNHSData,
    ):
        self.knhs = knhs
        self.df = df.reset_index(drop=True)
        self.prepared_data = prepared_data
        self._edge_index_raw = None
        self._edge_attr_raw = None

    def build_graph(
        self,
        *,
        fit_edge_scaler: bool = False,
        scale_edge_attr: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        edge_index, edge_attr = self.knhs._build_graph_from_prepared(
            self.prepared_data
        )
        self._edge_index_raw = edge_index
        self._edge_attr_raw = edge_attr
        edge_attr = self.knhs._maybe_scale_edge_attr(
            edge_attr,
            fit_edge_scaler=fit_edge_scaler,
            scale_edge_attr=scale_edge_attr,
        )
        return edge_index, edge_attr

    def build_cross_graph(
        self,
        target: PreparedKNHS | pd.DataFrame,
        *,
        fit_edge_scaler_on_source: bool = False,
        scale_edge_attr: bool = True,
    ):
        if not isinstance(target, PreparedKNHS):
            target = self.knhs.prepare(
                target,
                expected_feature_cols=self.prepared_data.similarity_feature_cols,
            )

        edge_index, edge_attr = self.knhs._build_cross_graph_from_prepared(
            source=self,
            target=target,
            fit_edge_scaler_on_source=fit_edge_scaler_on_source,
            scale_edge_attr=scale_edge_attr,
        )
        combined_df = pd.concat([self.df, target.df], ignore_index=True)
        target_mask = np.zeros(len(combined_df), dtype=bool)
        target_mask[len(self.df):] = True
        return combined_df, edge_index, edge_attr, target_mask

    def build_local_weights(
        self,
        *,
        lambda_distance: float | None = None,
        kernel: str | None = None,
        bandwidth_mode: str | None = None,
        bandwidth: float | None = None,
        row_standardize: bool | None = None,
        eps: float = 1e-12,
    ) -> GraphSpatialWeights:
        return self.knhs._build_local_weights_from_prepared(
            self.prepared_data,
            lambda_distance=lambda_distance,
            kernel=kernel,
            bandwidth_mode=bandwidth_mode,
            bandwidth=bandwidth,
            row_standardize=row_standardize,
            eps=eps,
        )


class NeighborSelector:
    """Estrategia para elegir vecinos a partir de candidatos ya filtrados."""

    def select(
        self,
        *,
        candidate_indices: np.ndarray,
        spatial_distances_km: np.ndarray,
        feature_distances: np.ndarray,
        k: int,
        source_index: int,
        target_index: int | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


@dataclass
class TopKFeatureNeighborSelector(NeighborSelector):
    """Selector histórico: ordena por distancia de features y toma top-k."""

    def select(
        self,
        *,
        candidate_indices: np.ndarray,
        spatial_distances_km: np.ndarray,
        feature_distances: np.ndarray,
        k: int,
        source_index: int,
        target_index: int | None = None,
    ) -> np.ndarray:
        del spatial_distances_km, source_index, target_index
        top_k = np.argsort(feature_distances)[:k]
        return np.asarray(candidate_indices[top_k], dtype=np.int64)


class EdgeFeatureBuilder:
    """Construye el vector de atributos para cada arista."""

    def build(
        self,
        *,
        src_index: int,
        dst_index: int,
        spatial_distance_km: float,
        feature_distance: float,
        src_features: np.ndarray,
        dst_features: np.ndarray,
        src_coord_deg: np.ndarray,
        dst_coord_deg: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


@dataclass
class DistanceEdgeFeatureBuilder(EdgeFeatureBuilder):
    """Builder histórico: `[dist_km, dist_feat]`."""

    def build(
        self,
        *,
        src_index: int,
        dst_index: int,
        spatial_distance_km: float,
        feature_distance: float,
        src_features: np.ndarray,
        dst_features: np.ndarray,
        src_coord_deg: np.ndarray,
        dst_coord_deg: np.ndarray,
    ) -> np.ndarray:
        del src_index, dst_index, src_features, dst_features, src_coord_deg, dst_coord_deg
        return np.asarray([spatial_distance_km, feature_distance], dtype=np.float32)


@dataclass
class GraphSpatialWeights:
    """Contenedor liviano de pesos espaciales compatible con el detector local."""

    neighbors: dict[int, np.ndarray]
    weights: dict[int, np.ndarray]
    spatial_distances: dict[int, np.ndarray] | None = None
    feature_distances: dict[int, np.ndarray] | None = None
    combined_distances: dict[int, np.ndarray] | None = None


class _NeighborSearchCore:
    """Lógica común de preparación y consulta de vecindad."""

    def __init__(
        self,
        *,
        schema: KNHSSchema,
        radius_km: float,
        k: int,
        feature_distance_mode: str,
        neighbor_selector: NeighborSelector,
    ):
        if radius_km <= 0:
            raise ValueError("radius_km debe ser > 0")
        if k <= 0:
            raise ValueError("k debe ser > 0")
        if feature_distance_mode not in {"euclidean", "local_weighted"}:
            raise ValueError(
                "feature_distance_mode debe ser 'euclidean' o 'local_weighted'"
            )

        self.schema = schema
        self.radius_km = float(radius_km)
        self.k = int(k)
        self.feature_distance_mode = str(feature_distance_mode)
        self.neighbor_selector = neighbor_selector

    @staticmethod
    def haversine_km(
        lat1: float,
        lon1: float,
        lat2: np.ndarray,
        lon2: np.ndarray,
    ) -> np.ndarray:
        """Devuelve distancia ortodrómica en km entre pares de puntos."""

        earth_radius_km = 6371.0
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return earth_radius_km * c

    def prepare(
        self,
        df: pd.DataFrame,
        *,
        expected_feature_cols: list[str] | None = None,
    ) -> PreparedKNHSData:
        feature_cols = self.schema.validate_dataframe(
            df,
            expected_feature_cols=expected_feature_cols,
        )
        coords_deg = self.schema.extract_coords_deg(df)
        similarity_features = self.schema.extract_similarity_features(
            df,
            feature_cols,
        )
        similarity_weights = self.schema.extract_similarity_weights(
            df,
            similarity_features,
            feature_cols,
            feature_distance_mode=self.feature_distance_mode,
        )
        lat_rad = np.deg2rad(coords_deg[:, 0])
        lon_rad = np.deg2rad(coords_deg[:, 1])
        return PreparedKNHSData(
            schema=self.schema,
            similarity_feature_cols=list(feature_cols),
            coords_deg=coords_deg,
            lat_rad=lat_rad,
            lon_rad=lon_rad,
            similarity_features=similarity_features,
            similarity_weights=similarity_weights,
        )

    def _compute_feature_distances(
        self,
        source_feature: np.ndarray,
        candidate_features: np.ndarray,
        source_weights: np.ndarray | None,
        candidate_weights: np.ndarray | None,
    ) -> np.ndarray:
        if self.feature_distance_mode == "euclidean":
            return np.linalg.norm(candidate_features - source_feature, axis=1)

        w_mean = 0.5 * (candidate_weights + source_weights)
        diff = candidate_features - source_feature
        return np.sqrt(np.sum(w_mean * diff * diff, axis=1))

    def _select_neighbors(
        self,
        *,
        candidate_indices: np.ndarray,
        spatial_distances_km: np.ndarray,
        feature_distances: np.ndarray,
        source_index: int,
        target_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        selected_idx = self.neighbor_selector.select(
            candidate_indices=candidate_indices,
            spatial_distances_km=spatial_distances_km,
            feature_distances=feature_distances,
            k=self.k,
            source_index=source_index,
            target_index=target_index,
        )
        selected_idx = np.asarray(selected_idx, dtype=np.int64).reshape(-1)
        if selected_idx.size == 0:
            empty = np.asarray([], dtype=float)
            return selected_idx, empty, empty

        pos_by_candidate = {
            int(candidate): pos
            for pos, candidate in enumerate(candidate_indices)
        }
        try:
            selected_pos = np.asarray(
                [pos_by_candidate[int(idx)] for idx in selected_idx],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(
                "neighbor_selector devolvió índices que no pertenecen a los candidatos."
            ) from exc

        return (
            selected_idx,
            np.asarray(spatial_distances_km[selected_pos], dtype=float),
            np.asarray(feature_distances[selected_pos], dtype=float),
        )

    def query_neighbors(
        self,
        *,
        source_data: PreparedKNHSData,
        source_index: int,
        target_data: PreparedKNHSData | None = None,
        target_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if target_data is None:
            target_data = source_data

        same_dataset = target_data is source_data
        d_km = self.haversine_km(
            source_data.lat_rad[source_index],
            source_data.lon_rad[source_index],
            target_data.lat_rad,
            target_data.lon_rad,
        )
        mask = d_km <= self.radius_km
        if same_dataset:
            mask = mask & (np.arange(target_data.n_rows) != source_index)
        if not np.any(mask):
            empty_idx = np.asarray([], dtype=np.int64)
            empty_dist = np.asarray([], dtype=float)
            return empty_idx, empty_dist, empty_dist

        candidate_idx = np.nonzero(mask)[0]
        feature_distances = self._compute_feature_distances(
            source_data.similarity_features[source_index],
            target_data.similarity_features[candidate_idx],
            None
            if source_data.similarity_weights is None
            else source_data.similarity_weights[source_index],
            None
            if target_data.similarity_weights is None
            else target_data.similarity_weights[candidate_idx],
        )
        return self._select_neighbors(
            candidate_indices=candidate_idx,
            spatial_distances_km=d_km[candidate_idx],
            feature_distances=feature_distances,
            source_index=source_index,
            target_index=target_index,
        )


class NeighborGraphBuilder:
    """Construye grafos de vecinos usando un esquema y una búsqueda común."""

    def __init__(
        self,
        *,
        neighbor_search: _NeighborSearchCore,
        edge_feature_builder: EdgeFeatureBuilder,
        add_reverse: bool = True,
    ):
        self.neighbor_search = neighbor_search
        self.edge_feature_builder = edge_feature_builder
        self.add_reverse = bool(add_reverse)

    def _build_edge_attr(
        self,
        *,
        src_index: int,
        dst_index: int,
        spatial_distance_km: float,
        feature_distance: float,
        src_data: PreparedKNHSData,
        dst_data: PreparedKNHSData,
    ) -> np.ndarray:
        edge_attr = self.edge_feature_builder.build(
            src_index=src_index,
            dst_index=dst_index,
            spatial_distance_km=spatial_distance_km,
            feature_distance=feature_distance,
            src_features=src_data.similarity_features[src_index],
            dst_features=dst_data.similarity_features[dst_index],
            src_coord_deg=src_data.coords_deg[src_index],
            dst_coord_deg=dst_data.coords_deg[dst_index],
        )
        edge_attr = np.asarray(edge_attr, dtype=np.float32).reshape(-1)
        if edge_attr.size == 0:
            raise ValueError("edge_feature_builder devolvió un vector vacío.")
        return edge_attr

    def build_raw(
        self,
        prepared_data: PreparedKNHSData,
    ) -> tuple[np.ndarray, np.ndarray]:
        src_list: list[int] = []
        dst_list: list[int] = []
        attr_list: list[np.ndarray] = []

        for source_index in range(prepared_data.n_rows):
            neighbors, spatial_distances, feature_distances = self.neighbor_search.query_neighbors(
                source_data=prepared_data,
                source_index=source_index,
            )
            for dst_index, dist_km_ij, dist_feat_ij in zip(
                neighbors,
                spatial_distances,
                feature_distances,
            ):
                src_list.append(source_index)
                dst_list.append(int(dst_index))
                attr_list.append(
                    self._build_edge_attr(
                        src_index=source_index,
                        dst_index=int(dst_index),
                        spatial_distance_km=float(dist_km_ij),
                        feature_distance=float(dist_feat_ij),
                        src_data=prepared_data,
                        dst_data=prepared_data,
                    )
                )

                if self.add_reverse:
                    src_list.append(int(dst_index))
                    dst_list.append(source_index)
                    attr_list.append(
                        self._build_edge_attr(
                            src_index=int(dst_index),
                            dst_index=source_index,
                            spatial_distance_km=float(dist_km_ij),
                            feature_distance=float(dist_feat_ij),
                            src_data=prepared_data,
                            dst_data=prepared_data,
                        )
                    )

        if not src_list:
            raise ValueError(
                "No se generaron aristas; revisa radius_km/k o las coordenadas."
            )

        edge_index = np.vstack([src_list, dst_list]).astype(np.int64)
        edge_attr = np.asarray(attr_list, dtype=np.float32)
        return edge_index, edge_attr

    def build_cross_split_raw(
        self,
        *,
        source_data: PreparedKNHSData,
        target_data: PreparedKNHSData,
        source_edge_index: np.ndarray | None = None,
        source_edge_attr: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if source_edge_index is None or source_edge_attr is None:
            source_edge_index, source_edge_attr = self.build_raw(source_data)

        src_list = [int(x) for x in source_edge_index[0]]
        dst_list = [int(x) for x in source_edge_index[1]]
        attr_list = [np.asarray(row, dtype=np.float32) for row in source_edge_attr]

        target_offset = source_data.n_rows
        for target_local_index in range(target_data.n_rows):
            neighbors, spatial_distances, feature_distances = self.neighbor_search.query_neighbors(
                source_data=target_data,
                source_index=target_local_index,
                target_data=source_data,
                target_index=target_local_index,
            )
            target_idx = target_offset + target_local_index
            for src_index, dist_km_ij, dist_feat_ij in zip(
                neighbors,
                spatial_distances,
                feature_distances,
            ):
                src_list.append(int(src_index))
                dst_list.append(int(target_idx))
                attr_list.append(
                    self._build_edge_attr(
                        src_index=int(src_index),
                        dst_index=target_local_index,
                        spatial_distance_km=float(dist_km_ij),
                        feature_distance=float(dist_feat_ij),
                        src_data=source_data,
                        dst_data=target_data,
                    )
                )

        edge_index = np.vstack([src_list, dst_list]).astype(np.int64)
        edge_attr = np.asarray(attr_list, dtype=np.float32)
        return edge_index, edge_attr


class LocalWeightBuilder:
    """Construye pesos locales a partir de la misma vecindad KNHS."""

    def __init__(self, *, neighbor_search: _NeighborSearchCore):
        self.neighbor_search = neighbor_search

    @staticmethod
    def _scale_1d(
        values: np.ndarray,
        scaler: MinMaxScaler | None = None,
    ) -> tuple[np.ndarray, MinMaxScaler | None]:
        values_arr = np.asarray(values, dtype=float).reshape(-1, 1)
        if values_arr.shape[0] == 0:
            return np.asarray([], dtype=float), scaler
        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(values_arr)
        scaled = scaler.transform(values_arr).reshape(-1)
        return np.asarray(scaled, dtype=float), scaler

    def _collect_local_neighbor_data(
        self,
        prepared_data: PreparedKNHSData,
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray, np.ndarray]:
        node_data = []
        all_spatial = []
        all_feature = []

        for source_index in range(prepared_data.n_rows):
            neighbors, spatial_distances, feature_distances = self.neighbor_search.query_neighbors(
                source_data=prepared_data,
                source_index=source_index,
            )
            node_data.append(
                {
                    "neighbors": np.asarray(neighbors, dtype=np.int64),
                    "spatial_distances": np.asarray(spatial_distances, dtype=float),
                    "feature_distances": np.asarray(feature_distances, dtype=float),
                }
            )
            if neighbors.size > 0:
                all_spatial.append(np.asarray(spatial_distances, dtype=float))
                all_feature.append(np.asarray(feature_distances, dtype=float))

        spatial_concat = (
            np.concatenate(all_spatial).astype(float, copy=False)
            if all_spatial
            else np.asarray([], dtype=float)
        )
        feature_concat = (
            np.concatenate(all_feature).astype(float, copy=False)
            if all_feature
            else np.asarray([], dtype=float)
        )
        return node_data, spatial_concat, feature_concat

    def build(
        self,
        prepared_data: PreparedKNHSData,
        *,
        lambda_distance: float = 0.5,
        kernel: str = "gaussian",
        bandwidth_mode: str = "adaptive",
        bandwidth: float = 1.0,
        row_standardize: bool = True,
        eps: float = 1e-12,
    ) -> GraphSpatialWeights:
        if not 0.0 <= lambda_distance <= 1.0:
            raise ValueError("lambda_distance debe estar entre 0 y 1.")
        if kernel != "gaussian":
            raise ValueError("Por ahora solo se soporta kernel='gaussian'.")
        if bandwidth_mode not in {"adaptive", "fixed"}:
            raise ValueError("bandwidth_mode debe ser 'adaptive' o 'fixed'.")
        if bandwidth <= 0:
            raise ValueError("bandwidth debe ser > 0.")

        node_data, spatial_concat, feature_concat = self._collect_local_neighbor_data(
            prepared_data
        )
        _, spatial_scaler = self._scale_1d(spatial_concat, None)
        _, feature_scaler = self._scale_1d(feature_concat, None)

        neighbors_dict: dict[int, np.ndarray] = {}
        weights_dict: dict[int, np.ndarray] = {}
        spatial_dict: dict[int, np.ndarray] = {}
        feature_dict: dict[int, np.ndarray] = {}
        combined_dict: dict[int, np.ndarray] = {}

        for node_index, info in enumerate(node_data):
            neighbors = np.asarray(info["neighbors"], dtype=np.int64)
            raw_spatial = np.asarray(info["spatial_distances"], dtype=float)
            raw_feature = np.asarray(info["feature_distances"], dtype=float)

            neighbors_dict[node_index] = neighbors
            spatial_dict[node_index] = raw_spatial
            feature_dict[node_index] = raw_feature

            if neighbors.size == 0:
                combined_dict[node_index] = np.asarray([], dtype=float)
                weights_dict[node_index] = np.asarray([], dtype=float)
                continue

            spatial_scaled, _ = self._scale_1d(raw_spatial, spatial_scaler)
            feature_scaled, _ = self._scale_1d(raw_feature, feature_scaler)
            combined_distance = (
                lambda_distance * spatial_scaled
                + (1.0 - lambda_distance) * feature_scaled
            )

            if bandwidth_mode == "adaptive":
                local_bandwidth = float(np.max(combined_distance))
                if local_bandwidth <= eps:
                    local_bandwidth = 1.0
            else:
                local_bandwidth = float(bandwidth)

            local_weights = np.exp(
                -0.5 * np.square(combined_distance / max(local_bandwidth, eps))
            )
            if row_standardize:
                weight_sum = float(np.sum(local_weights))
                if weight_sum > eps:
                    local_weights = local_weights / weight_sum

            combined_dict[node_index] = np.asarray(combined_distance, dtype=float)
            weights_dict[node_index] = np.asarray(local_weights, dtype=float)

        return GraphSpatialWeights(
            neighbors=neighbors_dict,
            weights=weights_dict,
            spatial_distances=spatial_dict,
            feature_distances=feature_dict,
            combined_distances=combined_dict,
        )


class KNHS:
    """Interfaz high-level para grafo y pesos locales KNHS.

    `KNHS` actúa como orquestador: el usuario le pide grafo, cross-split o
    pesos locales, y la clase delega el trabajo a `KNHSSchema`,
    `NeighborSearchCore`, `NeighborGraphBuilder` y `LocalWeightBuilder`.

    La API legacy basada en `lat_col/feature_cols/...` se mantiene por
    compatibilidad, pero también puede construirse a partir de `schema` y
    `weight_spec`.
    """

    def __init__(
        self,
        *,
        schema: KNHSSchema | None = None,
        weight_spec: KNHSWeightSpec | None = None,
        radius_km: float = 2.0,
        k: int = 5,
        feature_distance_mode: str = "euclidean",
        add_reverse: bool = True,
        scale_edge_features: bool = True,
        neighbor_selector: NeighborSelector | None = None,
        edge_feature_builder: EdgeFeatureBuilder | None = None,
    ):
        if weight_spec is not None:
            if schema is not None and schema != weight_spec.schema:
                raise ValueError(
                    "schema y weight_spec.schema no coinciden. "
                    "Pasá solo uno o asegurate de que sean equivalentes."
                )
            schema = weight_spec.schema
            radius_km = float(weight_spec.radius_km)
            if weight_spec.k is not None:
                k = int(weight_spec.k)
            feature_distance_mode = str(weight_spec.feature_distance_mode)
            neighbor_selector = weight_spec.neighbor_selector or neighbor_selector

        if schema is None:
            raise ValueError(
                "KNHS requiere `schema` o `weight_spec` para definir las columnas de similitud."
            )

        self.schema = schema
        self.weight_spec = weight_spec
        self.lat_col = self.schema.lat_col
        self.lon_col = self.schema.lon_col
        self.feature_cols = self.schema.similarity_feature_cols
        self.weight_cols = self.schema.similarity_weight_cols
        self.radius_km = float(radius_km)
        self.k = int(k)
        self.feature_distance_mode = str(feature_distance_mode)
        self.distance = self.feature_distance_mode
        self.add_reverse = bool(add_reverse)
        self.scale_edge_features = bool(scale_edge_features)
        self.neighbor_selector = neighbor_selector or TopKFeatureNeighborSelector()
        self.edge_feature_builder = edge_feature_builder or DistanceEdgeFeatureBuilder()
        self.edge_scaler_ = StandardScaler() if scale_edge_features else None
        self.edge_scaler_fitted_ = False

        self._neighbor_search = _NeighborSearchCore(
            schema=self.schema,
            radius_km=self.radius_km,
            k=self.k,
            feature_distance_mode=self.feature_distance_mode,
            neighbor_selector=self.neighbor_selector,
        )
        self._graph_builder = NeighborGraphBuilder(
            neighbor_search=self._neighbor_search,
            edge_feature_builder=self.edge_feature_builder,
            add_reverse=self.add_reverse,
        )
        self._local_weight_builder = LocalWeightBuilder(
            neighbor_search=self._neighbor_search,
        )

    @staticmethod
    def _coerce_feature_dataframe(
        feature_data,
        *,
        feature_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        if isinstance(feature_data, pd.DataFrame):
            return feature_data.copy()

        feature_arr = np.asarray(feature_data)
        if feature_arr.ndim != 2:
            raise ValueError(
                "feature_data debe ser un DataFrame o un array 2D."
            )

        if feature_cols is None:
            feature_cols = [f"feature_{idx}" for idx in range(feature_arr.shape[1])]
        if len(feature_cols) != feature_arr.shape[1]:
            raise ValueError(
                "feature_cols debe tener el mismo largo que las columnas de feature_data."
            )

        return pd.DataFrame(feature_arr, columns=list(feature_cols))

    def _attach_coords(
        self,
        df: pd.DataFrame,
        coords,
        *,
        coords_order: str = "lon_lat",
    ) -> pd.DataFrame:
        coords_arr = np.asarray(coords, dtype=float)
        if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
            raise ValueError(
                "coords debe tener shape (n, 2). "
                f"Recibido: {coords_arr.shape}."
            )
        if len(df) != len(coords_arr):
            raise ValueError(
                "coords y feature_data deben tener la misma cantidad de filas."
            )

        if coords_order not in {"lon_lat", "lat_lon"}:
            raise ValueError("coords_order debe ser 'lon_lat' o 'lat_lon'.")

        if coords_order == "lon_lat":
            lon_vals = coords_arr[:, 0]
            lat_vals = coords_arr[:, 1]
        else:
            lat_vals = coords_arr[:, 0]
            lon_vals = coords_arr[:, 1]

        df = df.copy()
        df[self.lat_col] = lat_vals
        df[self.lon_col] = lon_vals
        return df

    @staticmethod
    def _coerce_similarity_weights_frame(
        similarity_weights,
        *,
        feature_cols: list[str] | None,
        weight_cols: list[str] | None,
        n_rows: int,
    ) -> tuple[pd.DataFrame, list[str]]:
        resolved_weight_cols = (
            list(weight_cols)
            if weight_cols is not None
            else (
                [f"__sim_weight_{col}" for col in feature_cols]
                if feature_cols is not None
                else None
            )
        )
        if resolved_weight_cols is None:
            raise ValueError(
                "No se pudieron resolver los nombres de columnas para similarity_weights. "
                "Pasá similarity_weight_cols explícitamente o definí "
                "schema.similarity_feature_cols."
            )

        if isinstance(similarity_weights, pd.DataFrame):
            weights_df = similarity_weights.copy()
            if len(weights_df.columns) != len(resolved_weight_cols):
                raise ValueError(
                    "similarity_weights tiene una cantidad de columnas distinta a los pesos esperados."
                )
            weights_df.columns = resolved_weight_cols
        else:
            weights_arr = np.asarray(similarity_weights, dtype=float)
            if weights_arr.ndim != 2:
                raise ValueError(
                    "similarity_weights debe ser un DataFrame o un array 2D."
                )
            if weights_arr.shape[1] != len(resolved_weight_cols):
                raise ValueError(
                    "similarity_weights debe tener el mismo número de columnas que "
                    "similarity_feature_cols."
                )
            weights_df = pd.DataFrame(weights_arr, columns=resolved_weight_cols)

        if len(weights_df) == 1 and n_rows > 1:
            weights_df = pd.DataFrame(
                np.repeat(weights_df.to_numpy(dtype=float), n_rows, axis=0),
                columns=resolved_weight_cols,
            )

        if len(weights_df) != n_rows:
            raise ValueError(
                "similarity_weights y feature_data deben tener la misma cantidad de filas, "
                "o similarity_weights debe tener exactamente una fila para usar pesos "
                "globales por feature."
            )

        return weights_df, resolved_weight_cols

    def _resolve_similarity_weights(self, similarity_weights):
        if similarity_weights is not None:
            return similarity_weights
        if self.weight_spec is not None:
            return self.weight_spec.similarity_weights
        return None

    def _prepare_dataframe_for_schema(
        self,
        data,
        *,
        coords=None,
        similarity_weights=None,
        coords_order: str = "lon_lat",
    ) -> tuple[pd.DataFrame, KNHSSchema]:
        if not isinstance(data, pd.DataFrame):
            data = self._coerce_feature_dataframe(
                data,
                feature_cols=self.schema.similarity_feature_cols,
            )
        else:
            data = data.copy()

        if coords is not None:
            data = self._attach_coords(data, coords, coords_order=coords_order)

        effective_schema = self.schema
        resolved_similarity_weights = self._resolve_similarity_weights(similarity_weights)
        if resolved_similarity_weights is not None:
            weights_df, resolved_weight_cols = self._coerce_similarity_weights_frame(
                resolved_similarity_weights,
                feature_cols=self.schema.similarity_feature_cols,
                weight_cols=(
                    self.weight_spec.similarity_weight_cols
                    if self.weight_spec is not None and self.weight_spec.similarity_weight_cols is not None
                    else self.schema.similarity_weight_cols
                ),
                n_rows=len(data),
            )
            data = pd.concat(
                [data.reset_index(drop=True), weights_df.reset_index(drop=True)],
                axis=1,
            )
            effective_schema = KNHSSchema(
                lat_col=self.schema.lat_col,
                lon_col=self.schema.lon_col,
                similarity_feature_cols=(
                    list(self.schema.similarity_feature_cols)
                    if self.schema.similarity_feature_cols is not None
                    else None
                ),
                similarity_weight_cols=list(resolved_weight_cols),
            )

        return data.reset_index(drop=True), effective_schema

    def _build_neighbor_search_for_schema(
        self,
        schema: KNHSSchema,
    ) -> _NeighborSearchCore:
        return _NeighborSearchCore(
            schema=schema,
            radius_km=self.radius_km,
            k=self.k,
            feature_distance_mode=self.feature_distance_mode,
            neighbor_selector=self.neighbor_selector,
        )

    def prepare(
        self,
        data,
        *,
        coords=None,
        similarity_weights=None,
        expected_feature_cols: list[str] | None = None,
        coords_order: str = "lon_lat",
    ) -> PreparedKNHS:
        df, effective_schema = self._prepare_dataframe_for_schema(
            data,
            coords=coords,
            similarity_weights=similarity_weights,
            coords_order=coords_order,
        )
        neighbor_search = self._build_neighbor_search_for_schema(effective_schema)
        prepared_data = neighbor_search.prepare(
            df,
            expected_feature_cols=expected_feature_cols,
        )
        return PreparedKNHS(
            knhs=self,
            df=df,
            prepared_data=prepared_data,
        )

    def _prepare(
        self,
        df: pd.DataFrame,
        *,
        expected_feature_cols: list[str] | None = None,
    ) -> PreparedKNHSData:
        prepared = self.prepare(
            df,
            expected_feature_cols=expected_feature_cols,
        )
        return prepared.prepared_data

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

    def _build_raw(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        prepared = self.prepare(df)
        return self._build_graph_from_prepared(prepared.prepared_data)

    def _build_graph_from_prepared(
        self,
        prepared_data: PreparedKNHSData,
    ) -> tuple[np.ndarray, np.ndarray]:
        graph_builder = NeighborGraphBuilder(
            neighbor_search=self._build_neighbor_search_for_schema(prepared_data.schema),
            edge_feature_builder=self.edge_feature_builder,
            add_reverse=self.add_reverse,
        )
        return graph_builder.build_raw(prepared_data)

    def _build_cross_graph_from_prepared(
        self,
        *,
        source: PreparedKNHS,
        target: PreparedKNHS,
        fit_edge_scaler_on_source: bool = False,
        scale_edge_attr: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        graph_builder = NeighborGraphBuilder(
            neighbor_search=self._build_neighbor_search_for_schema(source.prepared_data.schema),
            edge_feature_builder=self.edge_feature_builder,
            add_reverse=self.add_reverse,
        )
        source_edge_index = source._edge_index_raw
        source_edge_attr = source._edge_attr_raw
        if source_edge_index is None or source_edge_attr is None:
            source_edge_index, source_edge_attr = graph_builder.build_raw(source.prepared_data)
            source._edge_index_raw = source_edge_index
            source._edge_attr_raw = source_edge_attr

        edge_index, edge_attr = graph_builder.build_cross_split_raw(
            source_data=source.prepared_data,
            target_data=target.prepared_data,
            source_edge_index=source_edge_index,
            source_edge_attr=source_edge_attr,
        )
        if fit_edge_scaler_on_source:
            self.fit_edge_scaler(source_edge_attr)
        edge_attr = self._maybe_scale_edge_attr(
            edge_attr,
            fit_edge_scaler=False,
            scale_edge_attr=scale_edge_attr,
        )
        return edge_index, edge_attr

    def _build_local_weights_from_prepared(
        self,
        prepared_data: PreparedKNHSData,
        *,
        lambda_distance: float | None = None,
        kernel: str | None = None,
        bandwidth_mode: str | None = None,
        bandwidth: float | None = None,
        row_standardize: bool | None = None,
        eps: float = 1e-12,
    ) -> GraphSpatialWeights:
        if self.weight_spec is not None:
            if lambda_distance is None:
                lambda_distance = self.weight_spec.lambda_distance
            if bandwidth_mode is None:
                bandwidth_mode = self.weight_spec.bandwidth_mode
            if bandwidth is None:
                bandwidth = self.weight_spec.bandwidth
            if row_standardize is None:
                row_standardize = self.weight_spec.row_standardize
        if lambda_distance is None:
            lambda_distance = 0.5
        if kernel is None:
            kernel = "gaussian"
        if bandwidth_mode is None:
            bandwidth_mode = "adaptive"
        if bandwidth is None:
            bandwidth = 1.0
        if row_standardize is None:
            row_standardize = True

        local_weight_builder = LocalWeightBuilder(
            neighbor_search=self._build_neighbor_search_for_schema(prepared_data.schema),
        )
        return local_weight_builder.build(
            prepared_data,
            lambda_distance=lambda_distance,
            kernel=kernel,
            bandwidth_mode=bandwidth_mode,
            bandwidth=bandwidth,
            row_standardize=row_standardize,
            eps=eps,
        )

    def build(
        self,
        data,
        *,
        coords=None,
        similarity_weights=None,
        coords_order: str = "lon_lat",
        fit_edge_scaler: bool = False,
        scale_edge_attr: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Genera edge_index y edge_attr a partir de datos y coordenadas."""
        prepared = self.prepare(
            data,
            coords=coords,
            similarity_weights=similarity_weights,
            coords_order=coords_order,
        )
        return prepared.build_graph(
            fit_edge_scaler=fit_edge_scaler,
            scale_edge_attr=scale_edge_attr,
        )

    def build_cross_split(
        self,
        source_data,
        target_data,
        *,
        source_coords=None,
        target_coords=None,
        source_similarity_weights=None,
        target_similarity_weights=None,
        coords_order: str = "lon_lat",
        fit_edge_scaler_on_source: bool = False,
        scale_edge_attr: bool = True,
    ):
        """Construye un grafo combinado source+target con aristas source->target."""
        source_prepared = self.prepare(
            source_data,
            coords=source_coords,
            similarity_weights=source_similarity_weights,
            coords_order=coords_order,
        )
        target_prepared = self.prepare(
            target_data,
            coords=target_coords,
            similarity_weights=target_similarity_weights,
            expected_feature_cols=source_prepared.prepared_data.similarity_feature_cols,
            coords_order=coords_order,
        )
        return source_prepared.build_cross_graph(
            target_prepared,
            fit_edge_scaler_on_source=fit_edge_scaler_on_source,
            scale_edge_attr=scale_edge_attr,
        )

    def build_local_weights(
        self,
        data,
        *,
        coords=None,
        similarity_weights=None,
        coords_order: str = "lon_lat",
        lambda_distance: float | None = None,
        kernel: str | None = None,
        bandwidth_mode: str | None = None,
        bandwidth: float | None = None,
        row_standardize: bool | None = None,
        eps: float = 1e-12,
    ) -> GraphSpatialWeights:
        """Construye pesos locales combinando geografía y similitud."""
        prepared = self.prepare(
            data,
            coords=coords,
            similarity_weights=similarity_weights,
            coords_order=coords_order,
        )
        return prepared.build_local_weights(
            lambda_distance=lambda_distance,
            kernel=kernel,
            bandwidth_mode=bandwidth_mode,
            bandwidth=bandwidth,
            row_standardize=row_standardize,
            eps=eps,
        )


NeighborSearchCore = _NeighborSearchCore


__all__ = [
    "KNHS",
    "KNHSSchema",
    "KNHSWeightSpec",
    "PreparedKNHS",
    "PreparedKNHSData",
    "NeighborSelector",
    "TopKFeatureNeighborSelector",
    "EdgeFeatureBuilder",
    "DistanceEdgeFeatureBuilder",
    "NeighborSearchCore",
    "NeighborGraphBuilder",
    "LocalWeightBuilder",
    "GraphSpatialWeights",
]
