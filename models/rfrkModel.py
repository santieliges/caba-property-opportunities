from baseModel import BaseModel
from pykrige.rk import RegressionKriging
from libpysal.weights import KNN
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from typing import Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

class RegressionKrigingModel(BaseModel):
    def __init__(self, rf_params=None, kriging_params=None):
        self.rf_params = rf_params or {}
        self.kriging_params = kriging_params or {}

        self.rf_ = None
        self.model_ = None
        self.best_params_ = None
        self.is_fitted_ = False

    def _build_rf(self):
        return RandomForestRegressor(
            n_estimators=self.rf_params.get("n_estimators", 200),
            max_depth=self.rf_params.get("max_depth", None),
            min_samples_split=self.rf_params.get("min_samples_split", 2),
            min_samples_leaf=self.rf_params.get("min_samples_leaf", 1),
            max_features=self.rf_params.get("max_features", "sqrt"),
            bootstrap=self.rf_params.get("bootstrap", True),
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X, y, coords):
        self.rf_ = self._build_rf()

        self.model_ = RegressionKriging(
            regression_model=self.rf_,
            n_closest_points=self.kriging_params.get("n_closest_points", 10),
        )

        self.model_.fit(X, coords, y)
        self.is_fitted_ = True
        return self

    def predict(self, X, coords):
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado")
        return self.model_.predict(X, coords)

    def tune_hyperparameters(
        self,
        X,
        y,
        coords,
        rf_param_grid=None,
        kriging_points=(5, 10, 20),
        cv=5
    ):
        if rf_param_grid is None:
            rf_param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [None, 15],
                "min_samples_leaf": [1, 3],
                "max_features": ["sqrt", "log2"],
            }

        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        grid = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            rf_param_grid,
            scoring="neg_root_mean_squared_error",
            cv=kf,
            n_jobs=-1
        )
        grid.fit(X, y)

        self.best_params_ = grid.best_params_
        self.rf_params.update(self.best_params_)

        # Barrido simple sobre kriging
        best_rmse = np.inf
        best_k = None

        for k in kriging_points:
            model = RegressionKriging(
                regression_model=self._build_rf(),
                n_closest_points=k
            )
            model.fit(X, coords, y)
            preds = model.predict(X, coords)
            rmse = mean_squared_error(y, preds, squared=False)

            if rmse < best_rmse:
                best_rmse = rmse
                best_k = k

        self.kriging_params["n_closest_points"] = best_k

        # Refit final
        self.fit(X, y, coords)
        return self
