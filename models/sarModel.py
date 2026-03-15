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

    def _build_weights(self, coords):
        w = KNN.from_array(coords, k=self.k)
        w.transform = 'r'
        return w

    def fit(self, X, y, coords, bw=None):
        self.feature_names_ = list(X.columns)
        self.X_train_ = X.copy()
        self.coords_train_ = np.asarray(coords).copy()
        self.y_train_ = np.asarray(y).reshape(-1, 1)
        
        # bw se interpreta como cantidad de vecinos para construir la matriz de pesos (KNN).
        # Si no se pasa, se usa self.k (configurado en __init__).
        if bw is not None:
            bw = int(bw)
            if bw < 1:
                raise ValueError(f"bw invalido: {bw}. Debe ser >= 1.")
            self.k = bw

        y = np.asarray(y).reshape(-1, 1)
        X = np.asarray(X)
        
        self.w = self._build_weights(coords)

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
        nbrs = NearestNeighbors(n_neighbors=self.k).fit(self.coords_train_)
        distances, indices = nbrs.kneighbors(coords_o)
        w_o = np.zeros((len(coords_o), len(self.coords_train_)))

        # ML_Lag.betas incluye: [const] + betas_X + [rho]. Para el termino lineal, excluimos rho.
        n_features = len(self.feature_names_)
        beta_x = np.asarray(self.model_.betas[1:1 + n_features])  # (n_features, 1)
        intercept = float(np.asarray(self.model_.betas[0]).ravel()[0])
        linear_interpolation = np.asarray(X_o) @ beta_x + intercept

        # Pesos out-of-sample: conectan el punto nuevo con sus k vecinos en train (fila-estandarizado).
        for row_idx, neigh_idx in enumerate(indices):
            w_o[row_idx, neigh_idx] = 1.0 / len(neigh_idx)

        spatial_lag = float(self.model_.rho) * (w_o @ self.y_train_)

        Y_pred = linear_interpolation + spatial_lag
        return Y_pred
    
    def predict(self, X, coords):
        preds = [
            self.predict_one_out_of_sample_point(X.iloc[i:i+1], coords[i].reshape(1, -1))
            for i in range(len(X))
        ]
        return np.array(preds).reshape(-1, 1)
    

    def summary(self):
        if not self.is_fitted_:
            raise ValueError("Model not fitted.")
        return self.model_.summary
