
import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from shapely.ops import unary_union
import folium
from branca.element import Element
from pathlib import Path
from abc import ABC, abstractmethod

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


# Clase base para visualizadores de mapas de outliers
class OutlierMapVisualizer(ABC):
    def __init__(self, gdf_all, results_df, method_name, barrios_path=None):
        self.gdf_all = gdf_all
        self.results_df = results_df.copy()
        if "method" in self.results_df.columns and method_name is not None:
            self.results_df = self.results_df.loc[self.results_df["method"] == method_name].copy()
        self.method_name = method_name
        self.barrios_path = barrios_path or "../GeoData/barrios.geojson"
        self.map_df = None
        self.folium_map = None

    def prepare_data(self):
        """Prepara los datos fusionando gdf_all con results."""
        if self.results_df.empty:
            raise ValueError(f"No hay resultados para el método {self.method_name}.")

        # Mantener solo columnas relevantes y eliminar duplicados
        keep_cols = [col for col in self.results_df.columns if col not in ["method", "fold"]]
        self.results_df = self.results_df[keep_cols].drop_duplicates(subset=["idx"], keep="first")

        self.map_df = self.gdf_all.merge(self.results_df, on="idx", how="left")
        self._add_default_columns()

    def _add_default_columns(self):
        """Método hook para subclases agregar columnas por defecto."""
        pass

    def create_base_map(self):
        """Crea el mapa base con Folium."""
        centro_caba = [self.map_df["latitud"].median(), self.map_df["longitud"].median()]
        self.folium_map = folium.Map(location=centro_caba, zoom_start=11, tiles="CartoDB positron")

        if self.barrios_path and Path(self.barrios_path).exists():
            barrios = gpd.read_file(self.barrios_path)
            if barrios.crs is not None:
                barrios = barrios.to_crs("EPSG:4326")
            folium.GeoJson(
                barrios.__geo_interface__,
                name="Barrios CABA",
                style_function=lambda _: {"fillColor": "#00000000", "color": "#4b5563", "weight": 1},
            ).add_to(self.folium_map)

    @abstractmethod
    def get_marker_style(self, row):
        """Define el estilo del marcador para cada fila."""
        pass

    @abstractmethod
    def get_tooltip(self, row):
        """Define el tooltip para cada marcador."""
        pass

    @abstractmethod
    def get_popup_html(self, row):
        """Define el popup HTML para cada marcador."""
        pass

    def add_markers(self):
        """Agrega marcadores al mapa."""
        for _, row in self.map_df.iterrows():
            style = self.get_marker_style(row)
            marker = folium.CircleMarker(
                location=[row["latitud"], row["longitud"]],
                **style
            )
            marker.add_child(folium.Tooltip(self.get_tooltip(row)))
            marker.add_child(folium.Popup(self.get_popup_html(row)))
            marker.add_to(self.folium_map)

    def build_map(self):
        """Construye el mapa completo."""
        self.prepare_data()
        self.create_base_map()
        self.add_markers()
        return self.folium_map

    def save_map(self, path):
        """Guarda el mapa como HTML."""
        if self.folium_map is None:
            self.build_map()
        self.folium_map.save(path)


# Subclase para ZTest
class ZTestMapVisualizer(OutlierMapVisualizer):
    def __init__(self, gdf_all, results_df, barrios_path=None):
        super().__init__(gdf_all, results_df, "ztest", barrios_path)

    def _add_default_columns(self):
        self.map_df["es_atipico_ztest"] = self.map_df["z_score"].notna()
        self.map_df["tipo_valor_atipico"] = self.map_df["tipo_valor_atipico"].fillna("NO_ATIPICO")
        self.map_df["severidad_valor_atipico"] = self.map_df["severidad_valor_atipico"].fillna("NO_ATIPICO")
        self.map_df["abs_z_score"] = self.map_df["abs_z_score"].fillna(0.0)

    def get_marker_style(self, row):
        if row["tipo_valor_atipico"] == "ALTO":
            intensity = min(row["abs_z_score"] / 3.0, 1.0)  # Normalizar a 3.0 como max
            color = self._lerp_color("#fca5a5", "#991b1b", intensity)
            return {"color": color, "fillColor": color, "radius": 4 + intensity * 4, "fillOpacity": 0.9, "weight": 1}
        elif row["tipo_valor_atipico"] == "BAJO":
            intensity = min(row["abs_z_score"] / 3.0, 1.0)
            color = self._lerp_color("#93c5fd", "#1d4ed8", intensity)
            return {"color": color, "fillColor": color, "radius": 4 + intensity * 4, "fillOpacity": 0.9, "weight": 1}
        else:
            return {"color": "#9ca3af", "fillColor": "#9ca3af", "radius": 2, "fillOpacity": 0.35, "weight": 0.5}

    def get_tooltip(self, row):
        if pd.notna(row.get("z_score")) and pd.notna(row.get("precio")):
            return f"{row['tipo_valor_atipico']} | z={row['z_score']:.2f} | USD {row['precio']:,.0f}"
        else:
            return f"Precio: USD {row['precio']:,.0f} | {row['area_m2_total']:,.0f} m² | {row['ambientes']:,.0f} amb"

    def get_popup_html(self, row):
        precio = row.get("precio")
        area = row.get("area_m2_total")
        ambientes = row.get("ambientes")
        antiguedad = row.get("antiguedad")
        url = row.get("url")
        z_score = row.get("z_score")
        tipo = row.get("tipo_valor_atipico")
        severidad = row.get("severidad_valor_atipico")

        precio_txt = f"USD {precio:,.0f}" if pd.notna(precio) else "N/D"
        area_txt = f"{area:,.0f} m²" if pd.notna(area) else "N/D"
        ambientes_txt = f"{ambientes:,.0f}" if pd.notna(ambientes) else "N/D"
        antiguedad_txt = f"{antiguedad:,.0f} años" if pd.notna(antiguedad) else "N/D"
        z_score_txt = f"{z_score:.2f}" if pd.notna(z_score) else "N/D"
        link_html = f'<a href="{url}" target="_blank">Ver publicación</a>' if url else "Sin link"

        return (
            f"<b>Precio:</b> {precio_txt}<br>"
            f"<b>Superficie:</b> {area_txt}<br>"
            f"<b>Ambientes:</b> {ambientes_txt}<br>"
            f"<b>Antigüedad:</b> {antiguedad_txt}<br>"
            f"<b>Tipo atípico:</b> {tipo}<br>"
            f"<b>Severidad:</b> {severidad}<br>"
            f"<b>Z-score:</b> {z_score_txt}<br>"
            f"{link_html}"
        )

    @staticmethod
    def _lerp_color(a, b, amount):
        ah = int(a[1:], 16)
        bh = int(b[1:], 16)
        ar, ag, ab = (ah >> 16) & 0xFF, (ah >> 8) & 0xFF, ah & 0xFF
        br, bg, bb = (bh >> 16) & 0xFF, (bh >> 8) & 0xFF, bh & 0xFF
        rr = int(ar + amount * (br - ar))
        rg = int(ag + amount * (bg - ag))
        rb = int(ab + amount * (bb - ab))
        return f"#{rr:02x}{rg:02x}{rb:02x}"


# Subclase para Combined Z + LISA
class CombinedZLisaMapVisualizer(OutlierMapVisualizer):
    def __init__(self, gdf_all, results_df, barrios_path=None):
        super().__init__(gdf_all, results_df, "combined_z_lisa", barrios_path)

    def _add_default_columns(self):
        self.map_df["is_outlier"] = self.map_df["is_outlier"].fillna(False)
        self.map_df["combined_score"] = self.map_df["combined_score"].fillna(0.0)

    def get_marker_style(self, row):
        if row["is_outlier"]:
            intensity = min(abs(row["combined_score"]) / 2.0, 1.0)  # Normalizar score
            if row["combined_score"] > 0:
                color = self._lerp_color("#fca5a5", "#991b1b", intensity)  # Rojos para positivos
            else:
                color = self._lerp_color("#93c5fd", "#1d4ed8", intensity)  # Azules para negativos
            return {"color": color, "fillColor": color, "radius": 4 + intensity * 4, "fillOpacity": 0.9, "weight": 1}
        else:
            return {"color": "#9ca3af", "fillColor": "#9ca3af", "radius": 2, "fillOpacity": 0.35, "weight": 0.5}

    def get_tooltip(self, row):
        if row["is_outlier"]:
            return f"Outlier | Score={row['combined_score']:.2f} | USD {row['precio']:,.0f}"
        else:
            return f"No outlier | USD {row['precio']:,.0f}"

    def get_popup_html(self, row):
        precio = row.get("precio")
        area = row.get("area_m2_total")
        ambientes = row.get("ambientes")
        url = row.get("url")
        score = row.get("combined_score")
        quadrant = row.get("quadrant")
        p_z = row.get("p_value_z")
        p_lisa = row.get("p_value_lisa")

        precio_txt = f"USD {precio:,.0f}" if pd.notna(precio) else "N/D"
        area_txt = f"{area:,.0f} m²" if pd.notna(area) else "N/D"
        ambientes_txt = f"{ambientes:,.0f}" if pd.notna(ambientes) else "N/D"
        score_txt = f"{score:.2f}" if pd.notna(score) else "N/D"
        quadrant_txt = quadrant if quadrant else "N/D"
        p_z_txt = f"{p_z:.3f}" if pd.notna(p_z) else "N/D"
        p_lisa_txt = f"{p_lisa:.3f}" if pd.notna(p_lisa) else "N/D"
        link_html = f'<a href="{url}" target="_blank">Ver publicación</a>' if url else "Sin link"

        return (
            f"<b>Precio:</b> {precio_txt}<br>"
            f"<b>Superficie:</b> {area_txt}<br>"
            f"<b>Ambientes:</b> {ambientes_txt}<br>"
            f"<b>Score combinado:</b> {score_txt}<br>"
            f"<b>Cuadrante LISA:</b> {quadrant_txt}<br>"
            f"<b>P-value Z:</b> {p_z_txt}<br>"
            f"<b>P-value LISA:</b> {p_lisa_txt}<br>"
            f"{link_html}"
        )

    @staticmethod
    def _lerp_color(a, b, amount):
        ah = int(a[1:], 16)
        bh = int(b[1:], 16)
        ar, ag, ab = (ah >> 16) & 0xFF, (ah >> 8) & 0xFF, ah & 0xFF
        br, bg, bb = (bh >> 16) & 0xFF, (bh >> 8) & 0xFF, bh & 0xFF
        rr = int(ar + amount * (br - ar))
        rg = int(ag + amount * (bg - ag))
        rb = int(ab + amount * (bb - ab))
        return f"#{rr:02x}{rg:02x}{rb:02x}"
