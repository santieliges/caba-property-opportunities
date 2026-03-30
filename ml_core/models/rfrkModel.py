from .baseModel import BaseModel
from pykrige.rk import RegressionKriging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
from scipy.linalg import LinAlgError

class RegressionKrigingModel(BaseModel):
    def __init__(self, rf_params=None, kriging_params=None):
        super().__init__()
        self.rf_params = rf_params or {}
        self.kriging_params = kriging_params or {}

        self.rf_ = None
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

    def fit(
        self,
        X,
        y,
        coords,
        n_closest_points=None,
        method=None,
        variogram_model=None,
        variogram_parameters=None,
        nlags=None,
        weight=None,
        exact_values=None,
        pseudo_inv=None,
        pseudo_inv_type=None,
        variogram_function=None,
        anisotropy_scaling=None,
        anisotropy_angle=None,
        enable_statistics=None,
        coordinates_type=None,
        drift_terms=None,
        point_drift=None,
        ext_drift_grid=None,
        functional_drift=None,
    ):
        self.feature_names_ = X.columns

        self.X_train_ = X.copy()
        self.coords_train_ = np.asarray(coords).copy()
        self.y_train_ = np.asarray(y).ravel()

        self.rf_ = self._build_rf()

        rk_kwargs = dict(self.kriging_params)
        rk_kwargs.pop("regression_model", None)

        # PyKrige puede encontrar matrices singulares o casi singulares
        # en ventanas locales durante la prediccion OOF. Usamos
        # pseudoinversa por defecto para volver el sistema mas estable.
        rk_kwargs.setdefault("pseudo_inv", True)
        rk_kwargs.setdefault("pseudo_inv_type", "pinvh")

        if n_closest_points is not None:
            n_closest_points = int(n_closest_points)
            if n_closest_points < 1:
                raise ValueError(
                    f"n_closest_points invalido: {n_closest_points}. Debe ser >= 1."
                )
            rk_kwargs["n_closest_points"] = n_closest_points

        if method is not None:
            rk_kwargs["method"] = method
        if variogram_model is not None:
            rk_kwargs["variogram_model"] = variogram_model
        if variogram_parameters is not None:
            rk_kwargs["variogram_parameters"] = variogram_parameters
        if nlags is not None:
            rk_kwargs["nlags"] = nlags
        if weight is not None:
            rk_kwargs["weight"] = weight
        if exact_values is not None:
            rk_kwargs["exact_values"] = exact_values
        if pseudo_inv is not None:
            rk_kwargs["pseudo_inv"] = pseudo_inv
        if pseudo_inv_type is not None:
            rk_kwargs["pseudo_inv_type"] = pseudo_inv_type
        if variogram_function is not None:
            rk_kwargs["variogram_function"] = variogram_function
        if anisotropy_scaling is not None:
            rk_kwargs["anisotropy_scaling"] = anisotropy_scaling
        if anisotropy_angle is not None:
            rk_kwargs["anisotropy_angle"] = anisotropy_angle
        if enable_statistics is not None:
            rk_kwargs["enable_statistics"] = enable_statistics
        if coordinates_type is not None:
            rk_kwargs["coordinates_type"] = coordinates_type
        if drift_terms is not None:
            rk_kwargs["drift_terms"] = drift_terms
        if point_drift is not None:
            rk_kwargs["point_drift"] = point_drift
        if ext_drift_grid is not None:
            rk_kwargs["ext_drift_grid"] = ext_drift_grid
        if functional_drift is not None:
            rk_kwargs["functional_drift"] = functional_drift

        self.model_ = RegressionKriging(
            regression_model=self.rf_,
            **rk_kwargs,
        )

        self.model_.fit(self.X_train_, self.coords_train_, self.y_train_)
        self.is_fitted_ = True
        return self


    def predict(self, X, coords):
        X = X[self.feature_names_]
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado")

        coords = np.asarray(coords)

        try:
            return self.model_.predict(X, coords).reshape(-1, 1)
        except LinAlgError as exc:
            warnings.warn(
                "PyKrige encontro una matriz singular en predict(); se usa fallback robusto punto a punto.",
                RuntimeWarning,
            )

            preds = np.empty((len(X), 1), dtype=float)
            fallback_count = 0

            for i in range(len(X)):
                X_i = X.iloc[[i]]
                coords_i = coords[i:i+1]
                try:
                    preds[i, 0] = float(np.asarray(self.model_.predict(X_i, coords_i)).reshape(-1)[0])
                except LinAlgError:
                    preds[i, 0] = float(np.asarray(self.rf_.predict(X_i)).reshape(-1)[0])
                    fallback_count += 1

            if fallback_count > 0:
                warnings.warn(
                    f"RegressionKrigingModel uso fallback de RandomForest en {fallback_count} punto(s) por matrices singulares de kriging.",
                    RuntimeWarning,
                )

            return preds
    
    def in_sample_predictions(self):
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado")

        return self.model_.predict(
            self.X_train_,
            self.coords_train_
        ).reshape(-1, 1)


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
            rmse = np.sqrt(mean_squared_error(y.flatten(), preds))

            if rmse < best_rmse:
                best_rmse = rmse
                best_k = k

        self.kriging_params["n_closest_points"] = best_k

        # Refit final
        self.fit(X, y, coords)
        return self

    def feature_importances_(self):
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado")
        return self.rf_.feature_importances_
