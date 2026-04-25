
import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.ops import unary_union

def generar_grid_predicciones(
    model,
    gdf_val_clean,
    features,
    barrios_path="../GeoData/barrios.geojson",
    grid_size=200,
    prediction_scale="log_precio",
    coord_feature_names=("longitud", "latitud"),
):

    barrios = gpd.read_file(barrios_path)
    barrios = barrios.to_crs(gdf_val_clean.crs)

    caba_polygon = unary_union(barrios.geometry)

    # 🆕 Crear std_depto automáticamente desde features
    std_depto = {}

    for col in features:

        if col in gdf_val_clean.columns:

            if pd.api.types.is_numeric_dtype(gdf_val_clean[col]):

                if col in ["ambientes", "banos", "cocheras", "antiguedad"]:
                    std_depto[col] = int(
                        round(gdf_val_clean[col].median())
                    )
                else:
                    std_depto[col] = gdf_val_clean[col].median()

            else:
                std_depto[col] = 0

        else:
            # columnas dummy que no estén
            std_depto[col] = 0

    xmin, ymin, xmax, ymax = caba_polygon.bounds

    xs = np.linspace(xmin, xmax, grid_size)
    ys = np.linspace(ymin, ymax, grid_size)

    xx, yy = np.meshgrid(xs, ys)

    coords_grid = np.column_stack([xx.ravel(), yy.ravel()])

    df_grid = gpd.GeoDataFrame(
        geometry=[Point(xy) for xy in coords_grid],
        crs=gdf_val_clean.crs
    )

    df_grid = df_grid[df_grid.within(caba_polygon)]

    coords_grid = np.array(
        [(p.x, p.y) for p in df_grid.geometry]
    )

    # 🆕 crear DataFrame con TODAS las features
    X_std_grid = pd.DataFrame(
        [std_depto] * len(coords_grid)
    )

    if coord_feature_names is not None:
        if len(coord_feature_names) != 2:
            raise ValueError(
                "coord_feature_names debe tener exactamente dos nombres de columna."
            )
        lon_col, lat_col = coord_feature_names
        if lon_col in X_std_grid.columns:
            X_std_grid[lon_col] = coords_grid[:, 0]
        if lat_col in X_std_grid.columns:
            X_std_grid[lat_col] = coords_grid[:, 1]

    # 🆕 asegurar columnas correctas
    X_std_grid = X_std_grid[features]

    coords_are_features = False
    if coord_feature_names is not None:
        coords_are_features = all(col in X_std_grid.columns for col in coord_feature_names)
    model_injects_coords = hasattr(model, "coord_feature_names")

    if (
        getattr(model, "use_kriging", None) is False
        and not coords_are_features
        and not model_injects_coords
    ):
        warnings.warn(
            "El modelo fue configurado con use_kriging=False. "
            "Como latitud/longitud no están presentes en `features` ni el modelo "
            "las inyecta internamente, el mapper usa el mismo vector de features "
            "en toda la grilla y las coordenadas no afectan la predicción, por lo "
            "que el mapa tenderá a salir uniforme.",
            RuntimeWarning,
        )

    y_pred = np.asarray(model.predict(
        X_std_grid,
        coords_grid
    )).reshape(-1)

    if prediction_scale == "log_precio":
        precio = np.exp(y_pred)
    elif prediction_scale == "precio":
        precio = y_pred
    else:
        raise ValueError(
            "prediction_scale debe ser 'log_precio' o 'precio'. "
            f"Recibido: {prediction_scale!r}."
        )

    # usar área real del depto estándar
    area_real = std_depto["area_m2_total"]
    if not np.isfinite(area_real) or area_real <= 0:
        raise ValueError(
            "area_m2_total del departamento estándar debe ser positiva para calcular precio_m2. "
            f"Recibido: {area_real!r}."
        )

    precio_m2 = precio / area_real
    if not np.all(np.isfinite(precio_m2)):
        raise ValueError(
            "La grilla contiene valores no finitos en precio_m2. "
            "Revisá la escala de salida del modelo y el área usada en el mapa."
        )
    if np.any(precio_m2 < 0):
        raise ValueError(
            "La grilla contiene valores negativos en precio_m2. "
            "Esto no debería ocurrir para precios; revisá si el modelo devuelve "
            "residuos, otra transformación distinta de log-precio, o una escala "
            "incompatible con `prediction_scale`."
        )
    if np.allclose(precio_m2, precio_m2[0]):
        warnings.warn(
            "El mapa resultó prácticamente constante. "
            "Esto suele pasar cuando la grilla usa features fijas en todos los puntos "
            "y el modelo no incorpora coordenadas en predict(), o cuando "
            "latitud/longitud no están incluidas como features.",
            RuntimeWarning,
        )

    df_grid["precio_m2"] = precio_m2

    return df_grid, barrios, std_depto

def plot_mapa_precio(
    df_grid,
    barrios,
    cmap="viridis"
):

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    df_grid.plot(
        column="precio_m2",
        ax=ax,
        cmap=cmap,
        markersize=5,
        legend=True
    )

    barrios.boundary.plot(
        ax=ax,
        color="black",
        linewidth=0.5
    )

    ax.set_title(
        "Precio estimado por m² – Departamento estándar"
    )

    ax.axis("off")

    return fig, ax

class MapaPrecio:

    def __init__(self, df_grid, barrios):
        self.df_grid = df_grid
        self.barrios = barrios
        self.fig = None
        self.ax = None

    def plot(self, cmap="viridis"):

        self.fig, self.ax = plt.subplots(
            figsize=(10, 10)
        )

        self.df_grid.plot(
            column="precio_m2",
            ax=self.ax,
            cmap=cmap,
            markersize=5,
            legend=True
        )

        self.barrios.boundary.plot(
            ax=self.ax,
            color="black",
            linewidth=0.5
        )

        self.ax.set_title(
            "Precio estimado por m² – Departamento estándar"
        )

        self.ax.axis("off")

        return self.fig

    def save(self, path, dpi=300):

        if self.fig is None:
            raise ValueError("Primero ejecutar plot()")

        self.fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight"
        )
