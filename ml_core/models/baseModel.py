
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class BaseModel:
    def __init__(self):

        self.model_ = None
        self.best_params_ = None
        self.is_fitted_ = False
        self.feature_names_ = None
        
        self.coords_train_ = None
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X, y, coords, **fit_params):
        raise NotImplementedError

    def predict(self, X, coords):
        raise NotImplementedError

    def tune_hyperparameters(self, X, y, coords):
        raise NotImplementedError

    def evaluate(
        self,
        X,
        y,
        coords,
        metrics=("rmse", "mae", "r2", "r2_adj")
    ):
        y_pred = self.predict(X, coords)
        results = {}

        n = len(y)
        p = X.shape[1]

        if "rmse" in metrics:
            results["rmse"] = mean_squared_error(y, y_pred, squared=False)

        if "mae" in metrics:
            results["mae"] = mean_absolute_error(y, y_pred)

        if "r2" in metrics or "r2_adj" in metrics:
            r2 = r2_score(y, y_pred)
            results["r2"] = r2

        if "r2_adj" in metrics:
            if n <= p + 1:
                results["r2_adj"] = np.nan
            else:
                results["r2_adj"] = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        return results

