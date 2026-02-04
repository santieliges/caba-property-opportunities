
from baseModel import BaseModel
from sklearn.preprocessing import StandardScaler
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class GWRModel(BaseModel):
    def __init__(self, gwr_params=None):
        self.gwr_params = gwr_params or {}

        self.scaler_ = None
        self.bw_ = None
        self.model_ = None

        self.X_train_ = None
        self.coords_train_ = None
        self.y_train_ = None

        self.is_fitted_ = False

    def fit(self, X, y, coords):
        self.scaler_ = StandardScaler()
        X_std = self.scaler_.fit_transform(X)

        self.coords_train_ = coords
        self.X_train_ = X_std
        self.y_train_ = y.reshape(-1, 1)

        bw_selector = Sel_BW(
            coords,
            self.y_train_,
            X_std,
            spherical=False
        )
        self.bw_ = bw_selector.search()

        self.model_ = GWR(
            coords,
            self.y_train_,
            X_std,
            bw=self.bw_,
            kernel=self.gwr_params.get("kernel", "bisquare"),
            fixed=self.gwr_params.get("fixed", False),
        )

        self.model_.fit()
        self.is_fitted_ = True
        return self

    def predict(self, X, coords):
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado")

        X_std = self.scaler_.transform(X)

        preds = self.model_.predict(
            coords,
            X_std
        ).predictions.flatten()

        return preds

    def tune_hyperparameters(
    self,
    X,
    y,
    coords,
    kernels=("bisquare", "gaussian"),
    fixed_options=(False, True)
    ):
        self.scaler_ = StandardScaler()
        X_std = self.scaler_.fit_transform(X)
        y = y.reshape(-1, 1)

        best_score = np.inf
        best_params = None

        for kernel in kernels:
            for fixed in fixed_options:

                bw_selector = Sel_BW(
                    coords,
                    y,
                    X_std,
                    spherical=self.gwr_params.get("spherical", False),
                    fixed=fixed,
                    kernel=kernel
                )
                bw = bw_selector.search()

                model = GWR(
                    coords,
                    y,
                    X_std,
                    bw=bw,
                    kernel=kernel,
                    fixed=fixed
                )

                res = model.fit()
                preds = res.predictions.flatten()

                rmse = mean_squared_error(y.flatten(), preds, squared=False)

                if rmse < best_score:
                    best_score = rmse
                    best_params = {
                        "kernel": kernel,
                        "fixed": fixed,
                        "bw": bw
                    }

        # Guardar mejores parámetros
        self.gwr_params.update({
            "kernel": best_params["kernel"],
            "fixed": best_params["fixed"],
        })
        self.bw_ = best_params["bw"]

        # Refit final limpio
        self.fit(X, y.flatten(), coords)

        return self


