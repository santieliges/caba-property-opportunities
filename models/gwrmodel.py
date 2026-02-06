
from .baseModel import BaseModel
from sklearn.preprocessing import StandardScaler
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import Rbf
from shapely.geometry import Point


class GWRModel(BaseModel):
    def __init__(self, gwr_params=None):
        self.gwr_params = gwr_params or {}

        self.scaler_ = None
        self.bw_ = None
        self.model_ = None
        self.results_ = None

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

        self.results_ = self.model_.fit()
        self.is_fitted_ = True
        return self

    def predict(self, X, coords):
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado")

        X_std = self.scaler_.transform(X)

        model_pred = GWR(
            coords=self.coords_train_,
            y=self.y_train_,
            X=self.X_train_,
            bw=self.bw_,
            kernel=self.gwr_params.get("kernel", "bisquare"),
            fixed=self.gwr_params.get("fixed", False),
        )

        model_pred.fit()

        preds = model_pred.predict(coords, X_std).predictions.flatten()
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
            
                preds = res.mu.flatten()

                rmse = np.sqrt(mean_squared_error(y.flatten(), preds))

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
    
    def plot_gwr_surfaces(
    self,
    gdf,
    feature_names,
    barrios=None,
    mask=None,
    polygon=None,
    grid_size=300,
    rbf_function="thin_plate",
    rbf_smooth=1.0,
    cmap="RdBu_r",
    return_surfaces=False
    ):
        """
        Plotea superficies suavizadas de coeficientes GWR.
        """

        gdf = gdf.copy()
        res = self.results_

        # ──────────────────────────────
        # Coeficientes locales
        # ──────────────────────────────
        coef_df = pd.DataFrame(
            res.params[:, 1:],      # sin intercepto
            columns=feature_names,
            index=gdf.index
        )

        for v in feature_names:
            gdf[f"beta_{v}"] = coef_df[v]

        gdf["intercept"] = res.params[:, 0]
        gdf["local_R2"] = res.localR2

        # ──────────────────────────────
        # Dominio espacial (POLÍGONO)
        # ──────────────────────────────
        if polygon is not None:
            domain_geom = polygon
        elif barrios is not None:
            domain_geom = barrios.unary_union
        else:
            domain_geom = gdf.unary_union

        xmin, ymin, xmax, ymax = domain_geom.bounds

        # ──────────────────────────────
        # Grid espacial
        # ──────────────────────────────
        xi = np.linspace(xmin, xmax, grid_size)
        yi = np.linspace(ymin, ymax, grid_size)
        Xi, Yi = np.meshgrid(xi, yi)

        extent = (xmin, xmax, ymin, ymax)

        # Coordenadas de los puntos (para la RBF)
        x = gdf.geometry.x.values
        y = gdf.geometry.y.values

        # ──────────────────────────────
        # Máscara espacial
        # ──────────────────────────────
        if mask is None:
            mask = np.array([
                domain_geom.contains(Point(xx, yy))
                for xx, yy in zip(Xi.ravel(), Yi.ravel())
            ]).reshape(Xi.shape)

        # ──────────────────────────────
        # Coeficientes a plotear
        # ──────────────────────────────
        coef_names = (
            ["intercept", "local_R2"] +
            [f"beta_{v}" for v in feature_names]
        )

        n = len(coef_names)
        ncols = 2
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(10 * ncols, 8 * nrows)
        )
        axes = axes.flatten()

        surfaces = {}

        # ──────────────────────────────
        # Loop de superficies
        # ──────────────────────────────
        for ax, coef in zip(axes, coef_names):

            Zi = self._rbf_surface(
                gdf[coef].values,
                rbf_function,
                rbf_smooth,
                x, y,
                Xi, Yi
            )

            Zi = np.where(mask, Zi, np.nan)
            surfaces[coef] = Zi

            zvals = Zi[np.isfinite(Zi)]

            # Si no hay superficie válida, no ploteamos
            if zvals.size == 0:
                ax.set_title(f"{coef} (sin señal espacial)")
                ax.axis("off")
                continue

            # Normalización centrada en 0 si cruza signos
            if zvals.min() < 0 < zvals.max():
                norm = TwoSlopeNorm(
                    vmin=zvals.min(),
                    vcenter=0.0,
                    vmax=zvals.max()
                )
            else:
                norm = None

            im = ax.imshow(
                Zi,
                extent=extent,
                origin="lower",
                cmap=cmap,
                norm=norm,
                alpha=0.9
            )

            if barrios is not None:
                barrios.boundary.plot(
                    ax=ax,
                    color="black",
                    linewidth=0.8
                )

            plt.colorbar(im, ax=ax, fraction=0.036, pad=0.01)

            if coef.startswith("beta_"):
                title = f"Influencia local de {coef.replace('beta_', '')}"
            elif coef == "intercept":
                title = "Intercepto local"
            elif coef == "local_R2":
                title = "R² local"

            ax.set_title(title, fontsize=14)
            ax.axis("off")

        # Apagar ejes sobrantes
        for ax in axes[len(coef_names):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

        if return_surfaces:
            return surfaces


    

    #Helper
    def _rbf_surface(self, z, rbf_function, rbf_smooth, x, y, Xi, Yi):
        rbf = Rbf(
            x, y, z,
            function=rbf_function,
            smooth=rbf_smooth
        )
        return rbf(Xi, Yi)

    def summary(self):
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado")
        return self.results_.summary()