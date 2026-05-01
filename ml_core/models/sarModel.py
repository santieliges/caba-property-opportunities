import numpy as np
import libpysal
from libpysal.weights import W
from sklearn.metrics import mean_squared_error
from spreg import ML_Lag

from .baseModel import BaseModel
from ..preprocessing.knhs import KNHS


class SpatialAutoregressiveModel(BaseModel):

    def __init__(
        self,
        sar_config=None,
        coord_feature_names=("longitud", "latitud"),
    ):
        super().__init__()

        # Parámetros del modelo
        self.sar_params = sar_config or {}

        # Extraer k (vecinos KNN)
        self.k = int(self.sar_params.get("k", 5))
        self.radius_km = float(self.sar_params.get("radius_km", 3.0))
        self.coord_feature_names = tuple(coord_feature_names)

        self.w = None
        self.model_ = None

    def _build_knhs_frame(self, X, coords):
        coords_arr = np.asarray(coords, dtype=float)
        if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
            raise ValueError(
                f"`coords` debe ser (n, 2). Recibido {coords_arr.shape}."
            )

        names = tuple(name.lower() for name in self.coord_feature_names)
        if len(names) == 2 and "long" in names[0] and "lat" in names[1]:
            lon_deg = coords_arr[:, 0]
            lat_deg = coords_arr[:, 1]
        else:
            lat_deg = coords_arr[:, 0]
            lon_deg = coords_arr[:, 1]

        graph_df = X[self.feature_names_].copy().reset_index(drop=True)
        graph_df["lat_deg"] = lat_deg
        graph_df["lon_deg"] = lon_deg
        return graph_df

    def _build_knhs_builder(self):
        return KNHS(
            lat_col="lat_deg",
            lon_col="lon_deg",
            feature_cols=self.feature_names_,
            weight_cols=None,
            radius_km=self.radius_km,
            k=self.k,
            distance="euclidean",
            add_reverse=True,
        )

    def _build_weights(self, X, coords):
        builder = self._build_knhs_builder()
        graph_df = self._build_knhs_frame(X, coords)
        edge_index, _ = builder.build(graph_df)

        neighbors = {i: [] for i in range(len(graph_df))}
        weights = {i: [] for i in range(len(graph_df))}

        for src, dst in edge_index.T:
            src_i = int(src)
            dst_i = int(dst)
            if dst_i not in neighbors[src_i]:
                neighbors[src_i].append(dst_i)
                weights[src_i].append(1.0)

        w = W(neighbors, weights)
        w.transform = "r"
        self._knhs_builder_ = builder
        self._graph_train_ = graph_df
        return w

    @staticmethod
    def _target_neighbor_indices(edge_index, target_offset, n_target):
        target_neighbors = []
        for local_idx in range(n_target):
            target_idx = target_offset + local_idx
            mask = edge_index[1] == target_idx
            neighbor_idx = edge_index[0][mask]
            target_neighbors.append(np.asarray(neighbor_idx, dtype=int))
        return target_neighbors

    def fit(self, X, y, coords):

        self.feature_names_ = list(X.columns)

        self.X_train_ = X.copy()
        self.coords_train_ = np.asarray(coords).copy()
        self.y_train_ = np.asarray(y).reshape(-1, 1)

        y = np.asarray(y).reshape(-1, 1)
        X = np.asarray(X)

        # Construir matriz espacial
        self.w = self._build_weights(self.X_train_, self.coords_train_)

        # Separar parámetros para ML_Lag
        ml_params = self.sar_params.copy()

        # Quitar params que no pertenecen a ML_Lag
        ml_params.pop("k", None)
        ml_params.pop("radius_km", None)

        # Parámetros por defecto
        default_params = {
            "name_y": "y",
            "name_x": self.feature_names_
        }

        params = {**default_params, **ml_params}

        # Ajustar modelo SAR
        self.model_ = ML_Lag(
            y,
            X,
            w=self.w,
            **params
        )

        self.is_fitted_ = True

        return self

    def in_sample_predictions(self):

        if not self.is_fitted_:
            raise ValueError("Model not fitted.")

        return self.model_.predy.flatten()

    def predict_one_out_of_sample_point(self, X, coord):

        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado")

        X_o = X[self.feature_names_]

        coords_o = np.asarray(coord)

        if coords_o.ndim != 2 or coords_o.shape[1] != 2:
            raise ValueError(
                f"`coord(s)` debe ser (n, 2). Recibido {coords_o.shape}."
            )

        graph_target = self._build_knhs_frame(X_o, coords_o)
        _, edge_index, _, target_mask = self._knhs_builder_.build_cross_split(
            self._graph_train_,
            graph_target,
        )
        indices = self._target_neighbor_indices(
            edge_index=edge_index,
            target_offset=len(self._graph_train_),
            n_target=int(np.asarray(target_mask).sum()),
        )

        n_features = len(self.feature_names_)

        beta_x = np.asarray(
            self.model_.betas[1:1 + n_features]
        )

        intercept = float(
            np.asarray(self.model_.betas[0]).ravel()[0]
        )

        linear_interpolation = (
            np.asarray(X_o) @ beta_x + intercept
        )

        spatial_lag_terms = []
        y_train_flat = self.y_train_.reshape(-1)
        for neighbor_idx in indices:
            if len(neighbor_idx) == 0:
                spatial_lag_terms.append(0.0)
            else:
                spatial_lag_terms.append(float(self.model_.rho) * y_train_flat[neighbor_idx].mean())
        spatial_lag = np.asarray(spatial_lag_terms, dtype=float).reshape(-1, 1)

        y_pred = linear_interpolation + spatial_lag

        return y_pred

    def predict(self, X, coords):

        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado")

        X_o = X[self.feature_names_]

        coords_o = np.asarray(coords)

        if coords_o.ndim != 2 or coords_o.shape[1] != 2:
            raise ValueError(
                f"`coords` debe ser (n, 2). Recibido {coords_o.shape}."
            )

        graph_target = self._build_knhs_frame(X_o, coords_o)
        _, edge_index, _, target_mask = self._knhs_builder_.build_cross_split(
            self._graph_train_,
            graph_target,
        )
        indices = self._target_neighbor_indices(
            edge_index=edge_index,
            target_offset=len(self._graph_train_),
            n_target=int(np.asarray(target_mask).sum()),
        )

        n_features = len(self.feature_names_)

        beta_x = np.asarray(
            self.model_.betas[1:1 + n_features]
        )

        intercept = float(
            np.asarray(self.model_.betas[0]).ravel()[0]
        )

        linear_interpolation = (
            np.asarray(X_o) @ beta_x + intercept
        )

        spatial_lag_terms = []
        y_train_flat = self.y_train_.reshape(-1)
        for neighbor_idx in indices:
            if len(neighbor_idx) == 0:
                spatial_lag_terms.append(0.0)
            else:
                spatial_lag_terms.append(float(self.model_.rho) * y_train_flat[neighbor_idx].mean())
        spatial_lag = np.asarray(spatial_lag_terms, dtype=float).reshape(-1, 1)

        return (
            linear_interpolation + spatial_lag
        ).reshape(-1, 1)

    def tune_hyperparameters(
        self,
        X,
        y,
        coords,
        k_values=None
    ):
        """Tune SAR hyperparameters over candidate KNN neighborhood sizes."""

        if k_values is None:
            k_values = [5, 10, 20, 30, 50]

        y_array = np.asarray(y).reshape(-1, 1)
        best_rmse = np.inf
        best_k = None

        for k in k_values:
            candidate_params = self.sar_params.copy()
            candidate_params["k"] = int(k)

            candidate = SpatialAutoregressiveModel(sar_config=candidate_params)
            candidate.fit(X, y_array, coords)

            preds = candidate.predict(X, coords).reshape(-1)
            rmse = np.sqrt(mean_squared_error(y_array.flatten(), preds))

            if rmse < best_rmse:
                best_rmse = rmse
                best_k = int(k)

        if best_k is None:
            raise ValueError("No valid k value was found during hyperparameter tuning.")

        self.k = best_k
        self.k_ = best_k
        self.sar_params["k"] = best_k
        self.best_params_ = {"k": best_k}

        self.fit(X, y_array, coords)
        return self

    def summary(self):

        if not self.is_fitted_:
            raise ValueError("Model not fitted.")

        return self.model_.summary

    def get_params(self):

        return self.sar_params
