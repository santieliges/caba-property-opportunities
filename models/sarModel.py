import numpy as np
import libpysal
from libpysal.weights import KNN
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

    def fit(self, X, y, coords):
        self.feature_names_ = list(X.columns)
        self.X_train_ = X.copy()
        self.coords_train_ = np.asarray(coords).copy()
        self.y_train_ = np.asarray(y).reshape(-1, 1)
        

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

    def summary(self):
        if not self.is_fitted_:
            raise ValueError("Model not fitted.")
        return self.model_.summary