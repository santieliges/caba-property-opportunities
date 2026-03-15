
from .baseModel import BaseModel
from sklearn.preprocessing import StandardScaler
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR, _compute_betas_gwr
from spglm.family import Gaussian
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
        super().__init__()
        self.gwr_params = gwr_params or {}

        self.scaler_ = None
        self.bw_ = None
        self.results_ = None
        self.summary_ = None

        self.X_train_ = None
        self.coords_train_ = None
        self.y_train_ = None

        self.is_fitted_ = False

    def fit(self, X, y, coords, bw=None):
        self.feature_names_ = X.columns

        self.scaler_ = StandardScaler()
        X_std = self.scaler_.fit_transform(X)

        self.X_train_ = X.copy()
        self.coords_train_ = np.asarray(coords).copy()
        self.y_train_ = np.asarray(y).reshape(-1, 1)

        # bw (bandwidth) controla el tamanio del vecindario local.
        # Si fixed=False (default), bw es "cantidad de vecinos" (adaptativo).
        # Si bw no se pasa, lo seleccionamos automaticamente con Sel_BW.
        if bw is None:
            bw_selector = Sel_BW(
                coords,
                self.y_train_,
                X_std,
                spherical=self.gwr_params.get("spherical", False),
                fixed=self.gwr_params.get("fixed", False),
                kernel=self.gwr_params.get("kernel", "bisquare"),
            )
            bw = bw_selector.search()

        bw = int(bw)
        if bw < 2:
            raise ValueError(f"bw invalido: {bw}. Debe ser >= 2.")
        self.bw_ = bw

        self.model_ = GWR(
            coords,
            self.y_train_,
            X_std,
            bw=self.bw_,
            kernel=self.gwr_params.get("kernel", "bisquare"),
            fixed=self.gwr_params.get("fixed", False),
        )

        self.results_ = self.model_.fit()
        self.summary_ = self.results_.summary()
        self.is_fitted_ = True
        return self

    def predict(self, X, coords):
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado")

        X = X[self.feature_names_]
        X_std = self.scaler_.transform(X)

        coords = np.asarray(coords)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"`coords` debe ser (n, 2). Recibido {coords.shape}.")
        if len(coords) != len(X_std):
            raise ValueError(
                f"Mismatch entre filas: len(coords)={len(coords)} vs X.shape[0]={len(X_std)}. "
                "Para predecir en una grilla, X y coords deben tener el mismo n."
            )

        # Workaround mgwr 2.2.1: GWR.predict rompe si len(points) > n_train (indexa self.X/self.y por i).
        if not isinstance(self.model_.family, Gaussian):
            raise NotImplementedError("Este wrapper de predict solo soporta GWR Gaussiano.")

        if self.model_.constant:
            P = np.hstack([np.ones((len(X_std), 1)), X_std])
        else:
            P = X_std

        orig_points = self.model_.points
        try:
            self.model_.points = coords
            preds = np.empty((len(coords), 1), dtype=float)
            for i in range(len(coords)):
                wi = self.model_._build_wi(i, self.model_.bw).reshape(-1, 1)
                betas, _ = _compute_betas_gwr(self.model_.y, self.model_.X, wi)
                preds[i, 0] = float(np.dot(P[i], betas))
        finally:
            self.model_.points = orig_points

        return preds.flatten()

    def in_sample_predictions(self):
        if not self.is_fitted_:
            raise RuntimeError("El modelo no está entrenado")

        return self.results_.predy.flatten()

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
        return self.summary_
