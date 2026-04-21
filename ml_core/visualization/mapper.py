
import json
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
    grid_size=200
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

    # 🆕 asegurar columnas correctas
    X_std_grid = X_std_grid[features]

    y_pred = model.predict(
        X_std_grid,
        coords_grid
    )

    # revertir log
    precio = np.exp(y_pred)

    # usar área real del depto estándar
    area_real = std_depto["area_m2_total"]

    df_grid["precio_m2"] = (
        precio / area_real
    )

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