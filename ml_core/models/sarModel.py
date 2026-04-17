import numpy as np
import libpysal
from libpysal.weights import KNN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from spreg import ML_Lag

from .baseModel import BaseModel


class SpatialAutoregressiveModel(BaseModel):

    def __init__(self, sar_config=None):
        super().__init__()

        # Parámetros del modelo
        self.sar_params = sar_config or {}

        # Extraer k (vecinos KNN)
        self.k = int(self.sar_params.get("k", 5))

        self.w = None
        self.nbrs_ = None
        self.model_ = None

    def _build_weights(self, coords):

        w = KNN.from_array(coords, k=self.k)
        w.transform = 'r'

        return w

    def fit(self, X, y, coords):

        self.feature_names_ = list(X.columns)

        self.X_train_ = X.copy()
        self.coords_train_ = np.asarray(coords).copy()
        self.y_train_ = np.asarray(y).reshape(-1, 1)

        y = np.asarray(y).reshape(-1, 1)
        X = np.asarray(X)

        # Construir matriz espacial
        self.w = self._build_weights(coords)

        # KNN para predicciones out-of-sample
        self.nbrs_ = NearestNeighbors(
            n_neighbors=self.k
        ).fit(self.coords_train_)

        # Separar parámetros para ML_Lag
        ml_params = self.sar_params.copy()

        # Quitar k (no pertenece a ML_Lag)
        ml_params.pop("k", None)

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

        if self.nbrs_ is None:
            self.nbrs_ = NearestNeighbors(
                n_neighbors=self.k
            ).fit(self.coords_train_)

        _, indices = self.nbrs_.kneighbors(coords_o)

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

        y_neighbors = self.y_train_.reshape(-1)[indices]

        spatial_lag = (
            float(self.model_.rho)
            * y_neighbors.mean(axis=1).reshape(-1, 1)
        )

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

        if self.nbrs_ is None:
            self.nbrs_ = NearestNeighbors(
                n_neighbors=self.k
            ).fit(self.coords_train_)

        _, indices = self.nbrs_.kneighbors(coords_o)

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

        y_neighbors = self.y_train_.reshape(-1)[indices]

        spatial_lag = (
            float(self.model_.rho)
            * y_neighbors.mean(axis=1).reshape(-1, 1)
        )

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