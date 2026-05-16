
import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from shapely.ops import unary_union
import folium
from branca.element import Element
from pathlib import Path
from abc import ABC, abstractmethod


def ensure_point_geodataframe(
    df,
    lon_col="longitud",
    lat_col="latitud",
    crs="EPSG:4326",
):
    if isinstance(df, gpd.GeoDataFrame):
        gdf = df.copy()
        if gdf.geometry is None:
            raise ValueError(
                "El GeoDataFrame no tiene columna geometry."
            )
        if gdf.crs is None and crs is not None:
            gdf = gdf.set_crs(crs)
        return gdf

    if lon_col not in df.columns or lat_col not in df.columns:
        raise ValueError(
            f"Se esperaban columnas {lon_col!r} y {lat_col!r} para construir geometry."
        )

    return gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=crs,
    )


def _load_boundary_gdf(boundary_gdf=None, boundary_path=None, target_crs=None):
    if boundary_gdf is None and boundary_path is None:
        return None

    if boundary_gdf is not None:
        boundary = boundary_gdf.copy()
    else:
        boundary = gpd.read_file(boundary_path)

    if target_crs is not None and boundary.crs is not None:
        boundary = boundary.to_crs(target_crs)

    return boundary


def plot_continuous_points_map(
    gdf,
    value_col,
    boundary_gdf=None,
    boundary_path=None,
    title=None,
    cmap="viridis",
    center=None,
    figsize=(10, 10),
    markersize=8,
    alpha=0.75,
    legend=True,
    edgecolor="none",
    missing_kwds=None,
):
    gdf_plot = ensure_point_geodataframe(gdf)

    if value_col not in gdf_plot.columns:
        raise ValueError(
            f"La columna {value_col!r} no existe en el GeoDataFrame."
        )

    values = pd.to_numeric(gdf_plot[value_col], errors="coerce")
    valid_mask = np.isfinite(values.to_numpy())
    if not valid_mask.any():
        raise ValueError(
            f"La columna {value_col!r} no contiene valores numéricos finitos."
        )

    gdf_plot = gdf_plot.loc[valid_mask].copy()
    gdf_plot[value_col] = values.loc[valid_mask]

    boundary = _load_boundary_gdf(
        boundary_gdf=boundary_gdf,
        boundary_path=boundary_path,
        target_crs=gdf_plot.crs,
    )

    fig, ax = plt.subplots(figsize=figsize)

    norm = None
    if center is not None:
        max_dev = float(np.nanmax(np.abs(gdf_plot[value_col] - center)))
        if max_dev > 0:
            norm = TwoSlopeNorm(
                vmin=center - max_dev,
                vcenter=center,
                vmax=center + max_dev,
            )

    gdf_plot.plot(
        column=value_col,
        ax=ax,
        cmap=cmap,
        norm=norm,
        markersize=markersize,
        alpha=alpha,
        legend=legend,
        edgecolor=edgecolor,
        missing_kwds=missing_kwds,
    )

    if boundary is not None:
        boundary.boundary.plot(
            ax=ax,
            color="black",
            linewidth=0.6,
        )

    ax.set_title(title or value_col)
    ax.axis("off")
    return fig, ax


def plot_residuals_map(
    gdf,
    residual_col="residual",
    boundary_gdf=None,
    boundary_path=None,
    title="Mapa de residuos",
    cmap="coolwarm",
    figsize=(10, 10),
    markersize=8,
    alpha=0.8,
    legend=True,
):
    return plot_continuous_points_map(
        gdf=gdf,
        value_col=residual_col,
        boundary_gdf=boundary_gdf,
        boundary_path=boundary_path,
        title=title,
        cmap=cmap,
        center=0.0,
        figsize=figsize,
        markersize=markersize,
        alpha=alpha,
        legend=legend,
    )

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
    area_cubierta = std_depto["area_m2_cubierta"]
    if not np.isfinite(area_cubierta) or area_cubierta <= 0:
        raise ValueError(
            "area_m2_cubierta del departamento estándar debe ser positiva para calcular precio_m2. "
            f"Recibido: {area_cubierta!r}."
        )

    precio_m2 = precio / area_cubierta
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
    def __init__(
        self,
        gdf_all,
        results_df,
        method_name,
        barrios_path=None,
        *,
        filter_config=None,
        popup_fields=None,
        popup_field_config=None,
    ):
        self.gdf_all = gdf_all
        self.results_df = results_df.copy()
        if "method" in self.results_df.columns and method_name is not None:
            self.results_df = self.results_df.loc[self.results_df["method"] == method_name].copy()
        self.method_name = method_name
        self.barrios_path = barrios_path or "../GeoData/barrios.geojson"
        self.filter_config = filter_config
        self.popup_fields = popup_fields
        self.popup_field_config = popup_field_config
        self.map_df = None
        self.folium_map = None
        self._marker_registry = []

    def _ensure_column(self, column_name, default_value):
        if column_name not in self.map_df.columns:
            self.map_df[column_name] = default_value
        return self.map_df[column_name]

    def prepare_data(self):
        """Prepara los datos fusionando gdf_all con results."""
        if self.results_df.empty:
            raise ValueError(f"No hay resultados para el método {self.method_name}.")

        # Mantener solo columnas propias del metodo para evitar sufijos
        # _x/_y al mergear con gdf_all, que ya contiene las columnas base
        # del inmueble (latitud, longitud, precio, etc.).
        keep_cols = [
            col for col in self.results_df.columns
            if col not in ["method", "fold"]
        ]
        self.results_df = self.results_df[keep_cols].drop_duplicates(
            subset=["idx"],
            keep="first",
        )

        base_cols = set(self.gdf_all.columns)
        method_cols = [
            col for col in self.results_df.columns
            if col == "idx" or col not in base_cols
        ]
        self.results_df = self.results_df[method_cols]

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
        filter_specs = self._prepare_filter_specs(self._get_filter_specs())
        self._marker_registry = []

        for _, row in self.map_df.iterrows():
            style = self.get_marker_style(row)
            marker = folium.CircleMarker(
                location=[row["latitud"], row["longitud"]],
                **style
            )
            marker.add_child(folium.Tooltip(self.get_tooltip(row)))
            marker.add_child(folium.Popup(self.get_popup_html(row)))
            marker.add_to(self.folium_map)

            if filter_specs:
                self._marker_registry.append(
                    self._build_marker_registry_item(
                        row=row,
                        marker_name=marker.get_name(),
                        filter_specs=filter_specs,
                    )
                )

        if filter_specs:
            self._add_filter_controls(filter_specs)

    def _numeric_bounds(self, column_name):
        values = pd.to_numeric(self.map_df[column_name], errors="coerce")
        values = values[np.isfinite(values)]
        if values.empty:
            return (0, 0)
        return (float(values.min()), float(values.max()))

    @staticmethod
    def _to_js_number(value):
        if pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _normalize_flag(value):
        if pd.isna(value):
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            if not np.isfinite(value):
                return None
            return bool(int(value))
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "si", "sí", "yes", "y", "pozo"}:
                return True
            if normalized in {"0", "false", "no", "n"}:
                return False
        return None

    @classmethod
    def _format_flag_text(cls, value):
        normalized = cls._normalize_flag(value)
        if normalized is None:
            return "N/D"
        return "Sí" if normalized else "No"

    @staticmethod
    def _infer_display_price(row):
        explicit_price = row.get("precio_estimado")
        if pd.notna(explicit_price):
            return float(explicit_price)

        precio = row.get("precio")
        valor_observado = row.get("valor_observado")
        valor_predicho = row.get("valor_predicho")
        if not (
            pd.notna(precio)
            and pd.notna(valor_observado)
            and pd.notna(valor_predicho)
        ):
            return np.nan

        precio = float(precio)
        valor_observado = float(valor_observado)
        valor_predicho = float(valor_predicho)
        if not np.isfinite(precio) or precio <= 0:
            return np.nan

        log_precio = np.log(precio)
        if abs(valor_observado - log_precio) < abs(valor_observado - precio):
            return float(np.exp(valor_predicho))
        return float(valor_predicho)

    @staticmethod
    def _humanize_field_name(field_name):
        return field_name.replace("_", " ").strip().capitalize()

    @staticmethod
    def _merge_field_config(default_config, override_config):
        merged = {
            field_name: dict(spec)
            for field_name, spec in (default_config or {}).items()
        }
        for field_name, spec in (override_config or {}).items():
            merged[field_name] = {
                **merged.get(field_name, {}),
                **dict(spec),
            }
        return merged

    def _default_filter_config(self):
        return {}

    def _resolved_filter_config(self):
        if self.filter_config is not None:
            return {
                field_name: dict(spec)
                for field_name, spec in self.filter_config.items()
            }
        return self._merge_field_config(
            self._default_filter_config(),
            None,
        )

    def _default_popup_fields(self):
        return []

    def _resolved_popup_fields(self):
        if self.popup_fields is not None:
            return list(self.popup_fields)
        return list(self._default_popup_fields())

    def _default_popup_field_config(self):
        return {
            "precio": {
                "label": "Precio",
                "formatter": "price",
            },
            "precio_estimado": {
                "label": "Precio estimado",
                "formatter": "estimated_price",
            },
            "area_m2_total": {
                "label": "Superficie",
                "formatter": "area",
            },
            "ambientes": {
                "label": "Ambientes",
                "formatter": "integer",
            },
            "antiguedad": {
                "label": "Antigüedad",
                "formatter": "years",
            },
            "pozo": {
                "label": "En pozo",
                "formatter": "flag",
            },
            "is_outlier": {
                "label": "Outlier",
                "formatter": "yes_no",
            },
            "tipo_valor_atipico": {
                "label": "Tipo atípico",
                "formatter": "text",
            },
            "severidad_valor_atipico": {
                "label": "Severidad",
                "formatter": "text",
            },
            "z_score": {
                "label": "Z-score",
                "formatter": "float",
                "decimals": 2,
            },
            "p_value": {
                "label": "P-value",
                "formatter": "float",
                "decimals": 4,
            },
            "residuo": {
                "label": "Residual",
                "formatter": "float",
                "decimals": 4,
            },
            "quadrant": {
                "label": "Cuadrante LISA",
                "formatter": "text",
            },
            "p_value_z": {
                "label": "P-value Z",
                "formatter": "float",
                "decimals": 3,
            },
            "p_value_lisa": {
                "label": "P-value LISA",
                "formatter": "float",
                "decimals": 3,
            },
            "url": {
                "label": "Link",
                "formatter": "link",
            },
        }

    def _resolved_popup_field_config(self):
        return self._merge_field_config(
            self._default_popup_field_config(),
            self.popup_field_config,
        )

    def _popup_raw_value(self, row, field_name):
        if field_name == "precio_estimado":
            return self._infer_display_price(row)
        return row.get(field_name)

    def _format_popup_value(self, row, field_name, spec):
        formatter = spec.get("formatter", "text")
        value = self._popup_raw_value(row, field_name)

        if formatter == "link":
            url = row.get("url")
            return f'<a href="{url}" target="_blank">Ver publicación</a>' if url else "Sin link"
        if formatter == "price":
            return f"USD {value:,.0f}" if pd.notna(value) else "N/D"
        if formatter == "estimated_price":
            return f"USD {value:,.0f}" if pd.notna(value) else "N/D"
        if formatter == "area":
            return f"{value:,.0f} m²" if pd.notna(value) else "N/D"
        if formatter == "integer":
            return f"{value:,.0f}" if pd.notna(value) else "N/D"
        if formatter == "years":
            return f"{value:,.0f} años" if pd.notna(value) else "N/D"
        if formatter == "flag":
            return self._format_flag_text(value)
        if formatter == "yes_no":
            if pd.isna(value):
                return "N/D"
            return "Sí" if bool(value) else "No"
        if formatter == "float":
            decimals = int(spec.get("decimals", 2))
            return f"{float(value):.{decimals}f}" if pd.notna(value) else "N/D"
        return "N/D" if pd.isna(value) else str(value)

    def _render_popup_lines(self, row, field_names):
        popup_config = self._resolved_popup_field_config()
        lines = []
        for field_name in field_names:
            spec = popup_config.get(
                field_name,
                {
                    "label": self._humanize_field_name(field_name),
                    "formatter": "text",
                },
            )
            rendered = self._format_popup_value(
                row,
                field_name,
                spec,
            )
            lines.append(f"<b>{spec['label']}:</b> {rendered}")
        return lines

    def build_filter_specs(
        self,
        filter_config,
    ):
        specs = []
        for field_name, field_spec in (filter_config or {}).items():
            if field_name not in self.map_df.columns:
                continue

            field_spec = dict(field_spec)
            kind = field_spec.get("kind", self._infer_filter_kind(field_name))
            spec = {
                "field": field_name,
                "label": field_spec.get(
                    "label",
                    self._humanize_field_name(field_name),
                ),
                "kind": kind,
            }

            if kind == "boolean":
                spec["boolean_labels"] = {
                    "all": "Todos",
                    "true": "Solo sí",
                    "false": "Solo no",
                    **field_spec.get("boolean_labels", {}),
                }
            elif kind == "categorical":
                values = self.map_df[field_name]
                options = sorted(
                    {
                        str(value)
                        for value in values
                        if pd.notna(value) and value != ""
                    }
                )
                spec["options"] = options

            specs.append(spec)

        return specs

    def _infer_filter_kind(self, field_name):
        values = self.map_df[field_name]

        normalized_bool = values.map(self._normalize_flag)
        non_null_values = values[values.notna()]
        if (
            len(non_null_values) > 0
            and normalized_bool.notna().sum() == len(non_null_values)
        ):
            return "boolean"

        numeric = pd.to_numeric(values, errors="coerce")
        if np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
            return "numeric"

        return "categorical"

    def _get_filter_specs(self):
        return self.build_filter_specs(
            self._resolved_filter_config(),
        )

    def _get_filter_panel_title(self):
        return f"Filtros {self.method_name}"

    def _get_filter_control_prefix(self):
        return (self.method_name or "map").replace("_", "-")

    def _prepare_filter_specs(self, filter_specs):
        prepared_specs = []
        prefix = self._get_filter_control_prefix()
        for spec in filter_specs or []:
            prepared = dict(spec)
            prepared["registry_key"] = spec.get("registry_key", spec["field"])
            prepared["control_key"] = spec.get("control_key", spec["field"])
            control_slug = prepared["control_key"].replace("_", "-")

            if prepared["kind"] == "numeric":
                bounds = self._numeric_bounds(prepared["field"])
                prepared["min_value"] = bounds[0]
                prepared["max_value"] = bounds[1]
                prepared["min_id"] = f"{prefix}-{control_slug}-min"
                prepared["max_id"] = f"{prefix}-{control_slug}-max"
            elif prepared["kind"] == "boolean":
                prepared["select_id"] = f"{prefix}-{control_slug}"
            elif prepared["kind"] == "categorical":
                prepared["container_id"] = f"{prefix}-{control_slug}-options"
                prepared["checkbox_class"] = f"{prefix}-{control_slug}-checkbox"
            else:
                raise ValueError(f"Tipo de filtro desconocido: {prepared['kind']!r}")

            prepared_specs.append(prepared)

        return prepared_specs

    def _serialize_filter_value(self, row, spec):
        field_name = spec["field"]
        value = row.get(field_name)
        if spec["kind"] == "numeric":
            return self._to_js_number(value)
        if spec["kind"] == "boolean":
            return self._normalize_flag(value)
        return None if pd.isna(value) else str(value)

    def _build_marker_registry_item(self, *, row, marker_name, filter_specs):
        item = {"marker_name": marker_name}
        for spec in filter_specs:
            item[spec["registry_key"]] = self._serialize_filter_value(row, spec)
        return item

    def _render_filter_control_html(self, filter_specs):
        prefix = self._get_filter_control_prefix()
        html_parts = [
            f"""
        <div id="{prefix}-filters" style="
            position: fixed;
            top: 16px;
            right: 16px;
            z-index: 9999;
            width: 280px;
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid #d1d5db;
            border-radius: 12px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
            padding: 14px 14px 12px 14px;
            font-family: Arial, sans-serif;
            color: #111827;
        ">
            <div style="font-size: 15px; font-weight: 700; margin-bottom: 10px;">{self._get_filter_panel_title()}</div>
        """
        ]

        for spec in filter_specs:
            if spec["kind"] == "numeric":
                html_parts.append(
                    f"""
            <div style="font-size: 12px; font-weight: 600; margin-bottom: 4px;">{spec["label"]}</div>
            <div style="display: flex; gap: 8px; margin-bottom: 10px;">
                <input id="{spec["min_id"]}" type="number" step="any" value="{spec["min_value"]}" style="width: 100%; padding: 6px;">
                <input id="{spec["max_id"]}" type="number" step="any" value="{spec["max_value"]}" style="width: 100%; padding: 6px;">
            </div>
                    """
                )
            elif spec["kind"] == "boolean":
                labels = spec["boolean_labels"]
                html_parts.append(
                    f"""
            <div style="font-size: 12px; font-weight: 600; margin-bottom: 4px;">{spec["label"]}</div>
            <div style="margin-bottom: 12px;">
                <select id="{spec["select_id"]}" style="width: 100%; padding: 6px;">
                    <option value="all">{labels["all"]}</option>
                    <option value="true">{labels["true"]}</option>
                    <option value="false">{labels["false"]}</option>
                </select>
            </div>
                    """
                )
            else:
                html_parts.append(
                    f"""
            <div style="font-size: 12px; font-weight: 600; margin-bottom: 4px;">{spec["label"]}</div>
            <div id="{spec["container_id"]}" style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; margin-bottom: 12px;"></div>
                    """
                )

        html_parts.append(
            f"""
            <div style="display: flex; gap: 8px;">
                <button id="apply-{prefix}-filters" type="button" style="flex: 1; padding: 8px; border: 0; border-radius: 8px; background: #1d4ed8; color: white; cursor: pointer;">Aplicar</button>
                <button id="reset-{prefix}-filters" type="button" style="flex: 1; padding: 8px; border: 1px solid #cbd5e1; border-radius: 8px; background: white; cursor: pointer;">Reset</button>
            </div>
        </div>
            """
        )

        return "".join(html_parts)

    def _add_filter_controls(self, filter_specs):
        if not self._marker_registry:
            return

        prefix = self._get_filter_control_prefix()
        map_name = self.folium_map.get_name()
        registry_json = json.dumps(self._marker_registry, ensure_ascii=False)
        specs_json = json.dumps(filter_specs, ensure_ascii=False)
        control_html = self._render_filter_control_html(filter_specs)

        script_html = f"""
        <script>
        (function() {{
            function initializeFilters() {{
                var missingMarkers = {registry_json}.some(function(item) {{
                    return !window[item.marker_name];
                }});
                if (missingMarkers) {{
                    window.setTimeout(initializeFilters, 120);
                    return;
                }}

                var map = {map_name};
                var markerRegistry = {registry_json};
                var filterSpecs = {specs_json};

                filterSpecs.forEach(function(spec) {{
                    if (spec.kind !== "categorical") return;
                    var container = document.getElementById(spec.container_id);
                    if (!container) return;
                    var options = spec.options || [];
                    container.innerHTML = options.map(function(option) {{
                        return '<label style="display:flex; align-items:center; gap:6px; font-size:12px;">'
                            + '<input type="checkbox" class="' + spec.checkbox_class + '" value="' + option + '" checked>'
                            + '<span>' + option + '</span>'
                            + '</label>';
                    }}).join("");
                }});

                function readNumber(id) {{
                    var raw = document.getElementById(id).value;
                    if (raw === "" || raw === null) return null;
                    var parsed = Number(raw);
                    return Number.isFinite(parsed) ? parsed : null;
                }}

                function passesRange(value, minValue, maxValue) {{
                    if (value === null || value === undefined || Number.isNaN(value)) {{
                        return false;
                    }}
                    if (minValue !== null && value < minValue) return false;
                    if (maxValue !== null && value > maxValue) return false;
                    return true;
                }}

                function selectedCategoricalOptions(spec) {{
                    return Array.from(document.querySelectorAll("." + spec.checkbox_class + ":checked"))
                        .map(function(el) {{ return el.value; }});
                }}

                function passesSpec(item, spec) {{
                    var value = item[spec.registry_key];
                    if (spec.kind === "numeric") {{
                        return passesRange(value, readNumber(spec.min_id), readNumber(spec.max_id));
                    }}
                    if (spec.kind === "boolean") {{
                        var mode = document.getElementById(spec.select_id).value;
                        if (mode === "true") return value === true;
                        if (mode === "false") return value === false;
                        return true;
                    }}
                    if (spec.kind === "categorical") {{
                        var selected = selectedCategoricalOptions(spec);
                        if (selected.length === 0) return true;
                        return selected.includes(String(value));
                    }}
                    return true;
                }}

                function applyFilters() {{
                    markerRegistry.forEach(function(item) {{
                        var marker = window[item.marker_name];
                        if (!marker) return;

                        var visible = filterSpecs.every(function(spec) {{
                            return passesSpec(item, spec);
                        }});

                        var onMap = map.hasLayer(marker);
                        if (visible && !onMap) {{
                            marker.addTo(map);
                        }} else if (!visible && onMap) {{
                            map.removeLayer(marker);
                        }}
                    }});
                }}

                function resetFilters() {{
                    filterSpecs.forEach(function(spec) {{
                        if (spec.kind === "numeric") {{
                            document.getElementById(spec.min_id).value = spec.min_value;
                            document.getElementById(spec.max_id).value = spec.max_value;
                        }} else if (spec.kind === "boolean") {{
                            document.getElementById(spec.select_id).value = "all";
                        }} else if (spec.kind === "categorical") {{
                            document.querySelectorAll("." + spec.checkbox_class).forEach(function(el) {{
                                el.checked = true;
                            }});
                        }}
                    }});
                    applyFilters();
                }}

                document.getElementById("apply-{prefix}-filters").addEventListener("click", applyFilters);
                document.getElementById("reset-{prefix}-filters").addEventListener("click", resetFilters);
                applyFilters();
            }}

            if (document.readyState === "loading") {{
                document.addEventListener("DOMContentLoaded", initializeFilters);
            }} else {{
                initializeFilters();
            }}
        }})();
        </script>
        """

        self.folium_map.get_root().html.add_child(Element(control_html))
        self.folium_map.get_root().html.add_child(Element(script_html))

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
    def __init__(
        self,
        gdf_all,
        results_df,
        barrios_path=None,
        *,
        significance_alpha=0.05,
        filter_config=None,
        popup_fields=None,
        popup_field_config=None,
    ):
        OutlierMapVisualizer.__init__(
            self,
            gdf_all,
            results_df,
            "ztest",
            barrios_path,
            filter_config=filter_config,
            popup_fields=popup_fields,
            popup_field_config=popup_field_config,
        )
        self.significance_alpha = float(significance_alpha)

    def _default_filter_config(self):
        return {
            "precio": {
                "kind": "numeric",
                "label": "Precio",
            },
            "ambientes": {
                "kind": "numeric",
                "label": "Ambientes",
            },
            "area_m2_total": {
                "kind": "numeric",
                "label": "Metros cuadrados",
            },
            "is_significant_outlier": {
                "kind": "boolean",
                "label": f"Atípico (p ≤ {self.significance_alpha:.2f})",
                "boolean_labels": {
                    "all": "Todos",
                    "true": "Solo atípicos",
                    "false": "Solo no atípicos",
                },
            },
            "pozo": {
                "kind": "boolean",
                "label": "En pozo",
                "boolean_labels": {
                    "all": "Todos",
                    "true": "Solo en pozo",
                    "false": "Solo no en pozo",
                },
            },
        }

    def _default_popup_fields(self):
        return [
            "precio",
            "precio_estimado",
            "area_m2_total",
            "ambientes",
            "antiguedad",
            "pozo",
            "is_outlier",
            "tipo_valor_atipico",
            "severidad_valor_atipico",
            "z_score",
            "p_value",
            "residuo",
            "url",
        ]

    def _add_default_columns(self):
        self.map_df["is_outlier"] = self._ensure_column(
            "is_outlier",
            False,
        ).fillna(False)
        self.map_df["tipo_valor_atipico"] = self._ensure_column(
            "tipo_valor_atipico",
            "NO_ATIPICO",
        ).fillna("NO_ATIPICO")
        self.map_df["severidad_valor_atipico"] = self._ensure_column(
            "severidad_valor_atipico",
            "NO_ATIPICO",
        ).fillna("NO_ATIPICO")
        self.map_df["abs_z_score"] = self._ensure_column(
            "abs_z_score",
            0.0,
        ).fillna(0.0)
        self.map_df["p_value"] = self._ensure_column(
            "p_value",
            np.nan,
        )
        p_values = pd.to_numeric(self.map_df["p_value"], errors="coerce")
        self.map_df["is_significant_outlier"] = (
            np.isfinite(p_values.to_numpy())
            & (p_values.to_numpy() <= self.significance_alpha)
        )
        self.map_df["es_atipico_ztest"] = self.map_df["is_significant_outlier"]

    def get_marker_style(self, row):
        p_value = pd.to_numeric(pd.Series([row.get("p_value")]), errors="coerce").iloc[0]
        if pd.notna(p_value):
            intensity = float(np.clip(1.0 - p_value, 0.0, 1.0))
            color = self._lerp_color("#dbeafe", "#1d4ed8", intensity)
            return {
                "color": color,
                "fillColor": color,
                "radius": 3 + intensity * 5,
                "fillOpacity": 0.35 + intensity * 0.55,
                "weight": 0.6 + intensity * 0.8,
            }
        return {
            "color": "#9ca3af",
            "fillColor": "#9ca3af",
            "radius": 2,
            "fillOpacity": 0.35,
            "weight": 0.5,
        }

    def get_tooltip(self, row):
        if pd.notna(row.get("z_score")) and pd.notna(row.get("precio")):
            p_value = row.get("p_value")
            p_value_txt = f"{p_value:.3f}" if pd.notna(p_value) else "N/D"
            return (
                f"{row['tipo_valor_atipico']} | z={row['z_score']:.2f} | "
                f"p={p_value_txt} | USD {row['precio']:,.0f}"
            )
        else:
            return f"Precio: USD {row['precio']:,.0f} | {row['area_m2_total']:,.0f} m² | {row['ambientes']:,.0f} amb"

    def get_popup_html(self, row):
        return "<br>".join(
            self._render_popup_lines(
                row,
                self._resolved_popup_fields(),
            )
        )

    def _get_filter_panel_title(self):
        return "Filtros Z-Test"

    def add_markers(self):
        OutlierMapVisualizer.add_markers(self)

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
    def __init__(
        self,
        gdf_all,
        results_df,
        barrios_path=None,
        *,
        filter_config=None,
        popup_fields=None,
        popup_field_config=None,
    ):
        OutlierMapVisualizer.__init__(
            self,
            gdf_all,
            results_df,
            "combined_z_lisa",
            barrios_path,
            filter_config=filter_config,
            popup_fields=popup_fields,
            popup_field_config=popup_field_config,
        )

    def _default_filter_config(self):
        return {
            "precio": {
                "kind": "numeric",
                "label": "Precio",
            },
            "ambientes": {
                "kind": "numeric",
                "label": "Ambientes",
            },
            "area_m2_total": {
                "kind": "numeric",
                "label": "Metros cuadrados",
            },
            "pozo": {
                "kind": "boolean",
                "label": "En pozo",
                "boolean_labels": {
                    "all": "Todos",
                    "true": "Solo en pozo",
                    "false": "Solo no en pozo",
                },
            },
            "quadrant": {
                "kind": "categorical",
                "label": "Cuadrante LISA",
            },
        }

    def _default_popup_fields(self):
        return [
            "precio",
            "precio_estimado",
            "area_m2_total",
            "ambientes",
            "pozo",
            "is_outlier",
            "quadrant",
            "p_value_z",
            "p_value_lisa",
            "url",
        ]

    def _add_default_columns(self):
        self.map_df["is_outlier"] = self.map_df["is_outlier"].fillna(False)
        self.map_df["quadrant"] = self.map_df["quadrant"].fillna("N/D")
        self.map_df["p_value_z"] = self._ensure_column(
            "p_value_z",
            np.nan,
        )
        self.map_df["p_value_lisa"] = self._ensure_column(
            "p_value_lisa",
            np.nan,
        )

    def get_marker_style(self, row):
        p_value_z = pd.to_numeric(pd.Series([row.get("p_value_z")]), errors="coerce").iloc[0]
        if pd.notna(p_value_z):
            intensity = float(np.clip(1.0 - p_value_z, 0.0, 1.0))
            color = self._lerp_color("#dbeafe", "#1d4ed8", intensity)
            return {
                "color": color,
                "fillColor": color,
                "radius": 3 + intensity * 5,
                "fillOpacity": 0.35 + intensity * 0.55,
                "weight": 0.6 + intensity * 0.8,
            }
        color = "#9ca3af"
        return {
            "color": color,
            "fillColor": color,
            "radius": 2,
            "fillOpacity": 0.35,
            "weight": 0.5,
        }

    def get_tooltip(self, row):
        p_z = row.get("p_value_z")
        precio = row.get("precio")
        quadrant = row.get("quadrant", "N/D")
        p_z_txt = f"{p_z:.3f}" if pd.notna(p_z) else "N/D"
        precio_txt = f"USD {precio:,.0f}" if pd.notna(precio) else "USD N/D"
        return f"p_z={p_z_txt} | {quadrant} | {precio_txt}"

    def get_popup_html(self, row):
        return "<br>".join(
            self._render_popup_lines(
                row,
                self._resolved_popup_fields(),
            )
        )

    def _get_filter_panel_title(self):
        return "Filtros Combined Z + LISA"

    def add_markers(self):
        OutlierMapVisualizer.add_markers(self)

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
