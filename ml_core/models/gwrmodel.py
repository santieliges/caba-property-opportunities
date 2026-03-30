
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
import warnings
from numpy.linalg import LinAlgError


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

        n_obs = len(self.y_train_)
        n_features = X_std.shape[1] + 1  # +1 por intercepto
        min_bw = max(2, n_features + 2)
        fixed = self.gwr_params.get("fixed", False)
        max_bw = n_obs - 1 if not fixed else n_obs
        max_bw = max(min_bw, max_bw)

        # bw (bandwidth) controla el tamanio del vecindario local.
        # Si fixed=False (default), bw es "cantidad de vecinos" (adaptativo).
        # Si bw no se pasa, intentamos seleccionarlo automaticamente, pero
        # evitamos regiones de busqueda demasiado chicas que suelen inducir
        # matrices singulares en algunos folds OOF.
        if bw is None:
            bw_selector = Sel_BW(
                coords,
                self.y_train_,
                X_std,
                spherical=self.gwr_params.get("spherical", False),
                fixed=fixed,
                kernel=self.gwr_params.get("kernel", "bisquare"),
            )

            search_starts = []
            for candidate_min in (
                min_bw,
                int(np.ceil(n_obs * 0.05)),
                int(np.ceil(n_obs * 0.10)),
                int(np.ceil(n_obs * 0.15)),
                int(np.ceil(n_obs * 0.20)),
            ):
                candidate_min = min(max(candidate_min, min_bw), max_bw)
                if candidate_min not in search_starts:
                    search_starts.append(candidate_min)

            selected_bw = None
            last_search_error = None
            for bw_min in search_starts:
                try:
                    selected_bw = bw_selector.search(bw_min=bw_min, bw_max=max_bw)
                    if bw_min != min_bw:
                        warnings.warn(
                            f"GWRModel selecciono bw con bw_min={bw_min} para evitar singularidades durante la busqueda.",
                            RuntimeWarning,
                        )
                    break
                except Exception as exc:
                    message = str(exc).lower()
                    if isinstance(exc, LinAlgError) or "singular" in message or "ill-conditioned" in message:
                        last_search_error = exc
                        continue
                    raise

            if selected_bw is None:
                selected_bw = max(min_bw, int(np.ceil(n_obs * 0.20)))
                selected_bw = min(selected_bw, max_bw)
                warnings.warn(
                    f"GWRModel no pudo seleccionar bw automaticamente de forma estable; usa bw heuristico={selected_bw}. Ultimo error: {last_search_error}",
                    RuntimeWarning,
                )

            bw = selected_bw

        bw = int(bw)
        bw = max(bw, min_bw)

        candidate_bws = []
        for candidate in (bw, int(np.ceil(bw * 1.15)), int(np.ceil(bw * 1.35)), int(np.ceil(bw * 1.6)), int(np.ceil(bw * 2.0)), max_bw):
            candidate = min(max(candidate, min_bw), max_bw)
            if candidate not in candidate_bws:
                candidate_bws.append(candidate)

        last_error = None
        for candidate_bw in candidate_bws:
            try:
                self.model_ = GWR(
                    coords,
                    self.y_train_,
                    X_std,
                    bw=candidate_bw,
                    kernel=self.gwr_params.get("kernel", "bisquare"),
                    fixed=self.gwr_params.get("fixed", False),
                )
                self.results_ = self.model_.fit()
                self.bw_ = candidate_bw
                self.summary_ = self.results_.summary()
                self.is_fitted_ = True

                if candidate_bw != bw:
                    warnings.warn(
                        f"GWRModel aumento bw de {bw} a {candidate_bw} para evitar una matriz singular o mal condicionada.",
                        RuntimeWarning,
                    )

                return self
            except Exception as exc:
                message = str(exc).lower()
                if isinstance(exc, LinAlgError) or "singular" in message or "ill-conditioned" in message:
                    last_error = exc
                    continue
                raise

        raise RuntimeError(
            f"GWRModel no pudo ajustarse de forma estable. Bandwidths probados: {candidate_bws}. Ultimo error: {last_error}"
        )

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
                preds[i, 0] = np.asarray(np.dot(P[i], betas)).reshape(-1)[0]
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
