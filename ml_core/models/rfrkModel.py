from .baseModel import BaseModel
from joblib import parallel_backend
from pykrige.rk import RegressionKriging
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import numpy as np
import pandas as pd
import warnings
from scipy.linalg import LinAlgError

class RegressionKrigingModel(BaseModel):
    def __init__(
        self,
        rf_params=None,
        kriging_params=None,
        use_kriging=True,
        coord_feature_names=("longitud", "latitud"),
    ):
        super().__init__()
        self.rf_params = rf_params or {}
        self.kriging_params = kriging_params or {}
        self.use_kriging = use_kriging
        self.coord_feature_names = tuple(coord_feature_names)

        self.rf_ = None
        self.model_ = None
        self.best_params_ = None
        self.is_fitted_ = False
        self.rf_feature_names_ = None

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

    def _validate_coords(self, coords, n_rows):
        coords_arr = np.asarray(coords)
        if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
            raise ValueError(
                f"`coords` debe ser (n, 2). Recibido {coords_arr.shape}."
            )
        if len(coords_arr) != n_rows:
            raise ValueError(
                f"X y coords deben tener la misma cantidad de filas. "
                f"Recibido: len(X)={n_rows}, len(coords)={len(coords_arr)}."
            )
        return coords_arr

    def _augment_features_with_coords(self, X, coords):
        X_df = X.copy()
        coords_arr = self._validate_coords(coords, len(X_df))
        lon_col, lat_col = self.coord_feature_names
        X_df[lon_col] = coords_arr[:, 0]
        X_df[lat_col] = coords_arr[:, 1]
        return X_df, coords_arr

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
        self.feature_names_ = list(X.columns)

        self.X_train_ = X.copy()
        self.coords_train_ = self._validate_coords(coords, len(X)).copy()
        self.y_train_ = np.asarray(y).ravel()
        self.X_train_rf_, _ = self._augment_features_with_coords(
            self.X_train_,
            self.coords_train_,
        )
        self.rf_feature_names_ = list(self.X_train_rf_.columns)

        self.rf_ = self._build_rf()
        self.model_ = None

        if not self.use_kriging:
            self.rf_.fit(self.X_train_rf_, self.y_train_)
            self.is_fitted_ = True
            return self

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

        self.model_.fit(self.X_train_rf_, self.coords_train_, self.y_train_)
        self.is_fitted_ = True
        return self


    def predict(self, X, coords):
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado")
        X = X[self.feature_names_]
        X_rf, coords_arr = self._augment_features_with_coords(X, coords)

        if not self.use_kriging:
            return np.asarray(self.rf_.predict(X_rf)).reshape(-1)

        try:
            return np.asarray(self.model_.predict(X_rf, coords_arr)).reshape(-1)
        except LinAlgError as exc:
            warnings.warn(
                "PyKrige encontro una matriz singular en predict(); se usa fallback robusto punto a punto.",
                RuntimeWarning,
            )

            preds = np.empty(len(X), dtype=float)
            fallback_count = 0

            for i in range(len(X)):
                X_i = X_rf.iloc[[i]]
                coords_i = coords_arr[i:i+1]
                try:
                    preds[i] = float(np.asarray(self.model_.predict(X_i, coords_i)).reshape(-1)[0])
                except LinAlgError:
                    preds[i] = float(np.asarray(self.rf_.predict(X_i)).reshape(-1)[0])
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

        if not self.use_kriging:
            return np.asarray(self.rf_.predict(self.X_train_rf_)).reshape(-1)

        return np.asarray(self.model_.predict(
            self.X_train_rf_,
            self.coords_train_
        )).reshape(-1)


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
                "n_estimators": [200, 500],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 3, 5],
                "max_features": ["sqrt", "log2", 0.5],
                "bootstrap": [True],
            }

        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        X_rf, coords_arr = self._augment_features_with_coords(X, coords)
        rf_feature_names = list(X_rf.columns)

        grid = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            rf_param_grid,
            scoring="neg_root_mean_squared_error",
            cv=kf,
            n_jobs=-1
        )
        grid.fit(X_rf, y)

        self.best_params_ = grid.best_params_
        self.rf_params.update(self.best_params_)

        if not self.use_kriging:
            self.fit(X, y, coords)
            return self

        # Barrido simple sobre kriging
        best_rmse = np.inf
        best_k = None

        for k in kriging_points:
            model = RegressionKriging(
                regression_model=self._build_rf(),
                n_closest_points=k
            )
            model.fit(X_rf, coords_arr, y)
            preds = model.predict(X_rf, coords_arr)
            rmse = np.sqrt(mean_squared_error(y.flatten(), preds))

            if rmse < best_rmse:
                best_rmse = rmse
                best_k = k

        self.kriging_params["n_closest_points"] = best_k

        # Refit final
        self.fit(X, y, coords)
        return self

    def feature_importances_(
        self,
        X=None,
        y=None,
        coords=None,
        method="permutation",
        n_repeats=10,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        as_frame=False,
    ):
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado")

        method = method.lower()

        if method in {"rf", "impurity", "mdi"}:
            importances = np.asarray(
                self.rf_.feature_importances_
            )
            if as_frame:
                return pd.DataFrame(
                    {
                        "feature": list(self.rf_feature_names_),
                        "importance": importances,
                    }
                ).sort_values(
                    "importance",
                    ascending=False,
                ).reset_index(drop=True)
            return importances

        if method != "permutation":
            raise ValueError(
                "method debe ser 'permutation' "
                "o uno de {'rf', 'impurity', 'mdi'}."
            )

        if X is None:
            X = self.X_train_
        if y is None:
            y = self.y_train_
        if coords is None:
            coords = self.coords_train_

        X_eval = X.copy()
        y_eval = np.asarray(y).ravel()
        coords_eval = self._validate_coords(coords, len(X_eval))

        if len(X_eval) != len(y_eval):
            raise ValueError(
                "X e y deben tener la misma cantidad de filas."
            )

        if len(X_eval) != len(coords_eval):
            raise ValueError(
                "X y coords deben tener la misma cantidad de filas."
            )

        def _permutation_scorer(estimator, X_perm, y_true):
            y_pred = np.asarray(
                estimator.predict(X_perm, coords_eval)
            ).reshape(-1)
            return self._score_predictions(
                y_true=y_true,
                y_pred=y_pred,
                scoring=scoring,
            )

        # Con procesos, joblib intenta serializar el estimador completo y
        # memmapear arrays grandes a disco temporal, lo que puede explotar
        # en notebooks con datasets pesados. Con hilos evitamos pickling.
        with parallel_backend("threading"):
            result = permutation_importance(
                estimator=self,
                X=X_eval,
                y=y_eval,
                scoring=_permutation_scorer,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=n_jobs,
            )

        if as_frame:
            return pd.DataFrame(
                {
                    "feature": list(X_eval.columns),
                    "importance_mean": result.importances_mean,
                    "importance_std": result.importances_std,
                }
            ).sort_values(
                "importance_mean",
                ascending=False,
            ).reset_index(drop=True)

        return result.importances_mean

    def _score_predictions(
        self,
        y_true,
        y_pred,
        scoring,
    ):

        if callable(scoring):
            return float(scoring(y_true, y_pred))

        if scoring in {None, "neg_root_mean_squared_error"}:
            return -float(
                np.sqrt(mean_squared_error(y_true, y_pred))
            )

        if scoring == "rmse":
            return float(
                np.sqrt(mean_squared_error(y_true, y_pred))
            )

        if scoring == "neg_mean_squared_error":
            return -float(
                mean_squared_error(y_true, y_pred)
            )

        if scoring == "mse":
            return float(
                mean_squared_error(y_true, y_pred)
            )

        if scoring in {"neg_mean_absolute_error", "neg_mae"}:
            return -float(
                mean_absolute_error(y_true, y_pred)
            )

        if scoring in {"mean_absolute_error", "mae"}:
            return float(
                mean_absolute_error(y_true, y_pred)
            )

        if scoring == "r2":
            return float(
                r2_score(y_true, y_pred)
            )

        raise ValueError(
            "scoring no soportado. Usa uno de: "
            "'neg_root_mean_squared_error', 'rmse', "
            "'neg_mean_squared_error', 'mse', "
            "'neg_mean_absolute_error', 'mae', 'r2', "
            "o un callable(y_true, y_pred)."
        )
