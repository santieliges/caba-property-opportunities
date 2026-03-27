import numpy as np
import libpysal
from libpysal.weights import KNN
from sklearn.neighbors import NearestNeighbors
from spreg import ML_Lag
from .baseModel import BaseModel

class SpatialAutoregressiveModel(BaseModel):

    def __init__(self, k=5):
        super().__init__()
        self.k = k
        self.w = None
        self.nbrs_ = None

    def _build_weights(self, coords):
        w = KNN.from_array(coords, k=self.k)
        w.transform = 'r'
        return w

    def fit(self, X, y, coords, k=None):
        self.feature_names_ = list(X.columns)
        self.X_train_ = X.copy()
        self.coords_train_ = np.asarray(coords).copy()
        self.y_train_ = np.asarray(y).reshape(-1, 1)
        
        # k se interpreta como cantidad de vecinos para construir la matriz de pesos (KNN).
        # Si no se pasa, se usa self.k (configurado en __init__).
        if k is not None:
            k = int(k)
            if k < 1:
                raise ValueError(f"k invalido: {k}. Debe ser >= 1.")
            self.k = k

        y = np.asarray(y).reshape(-1, 1)
        X = np.asarray(X)
        
        self.w = self._build_weights(coords)
        self.nbrs_ = NearestNeighbors(n_neighbors=self.k).fit(self.coords_train_)

        self.model_ = ML_Lag(
            y,
            X,
            w=self.w,
            name_y="y",
            name_x=self.feature_names_
        )

        self.is_fitted_ = True
        return self

    def in_sample_predictions(self):
        if not self.is_fitted_:
            raise ValueError("Model not fitted.")

        return self.model_.predy.flatten()

    def predict_one_out_of_sample_point(self, X, coord):
        # Trend to signal prediction para un unico punto out of sample como se define en el paper Goulard et al. (2016)

        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado")

        X_o = X[self.feature_names_]

        coords_o = np.asarray(coord)
        if coords_o.ndim != 2 or coords_o.shape[1] != 2:
            raise ValueError(f"`coord(s)` debe ser (n, 2). Recibido {coords_o.shape}.")
        if self.nbrs_ is None:
            self.nbrs_ = NearestNeighbors(n_neighbors=self.k).fit(self.coords_train_)
        _, indices = self.nbrs_.kneighbors(coords_o)

        # ML_Lag.betas incluye: [const] + betas_X + [rho]. Para el termino lineal, excluimos rho.
        n_features = len(self.feature_names_)
        beta_x = np.asarray(self.model_.betas[1:1 + n_features])  # (n_features, 1)
        intercept = float(np.asarray(self.model_.betas[0]).ravel()[0])
        linear_interpolation = np.asarray(X_o) @ beta_x + intercept

        y_neighbors = self.y_train_.reshape(-1)[indices]  # (n, k)
        spatial_lag = float(self.model_.rho) * y_neighbors.mean(axis=1).reshape(-1, 1)

        y_pred = linear_interpolation + spatial_lag
        return y_pred
    
    def predict(self, X, coords):
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado")

        X_o = X[self.feature_names_]
        coords_o = np.asarray(coords)
        if coords_o.ndim != 2 or coords_o.shape[1] != 2:
            raise ValueError(f"`coords` debe ser (n, 2). Recibido {coords_o.shape}.")
        if self.nbrs_ is None:
            self.nbrs_ = NearestNeighbors(n_neighbors=self.k).fit(self.coords_train_)

        _, indices = self.nbrs_.kneighbors(coords_o)

        n_features = len(self.feature_names_)
        beta_x = np.asarray(self.model_.betas[1:1 + n_features])  # (n_features, 1)
        intercept = float(np.asarray(self.model_.betas[0]).ravel()[0])
        linear_interpolation = np.asarray(X_o) @ beta_x + intercept  # (n, 1)

        y_neighbors = self.y_train_.reshape(-1)[indices]  # (n, k)
        spatial_lag = float(self.model_.rho) * y_neighbors.mean(axis=1).reshape(-1, 1)

        return (linear_interpolation + spatial_lag).reshape(-1, 1)
    

    def summary(self):
        if not self.is_fitted_:
            raise ValueError("Model not fitted.")
        return self.model_.summary
