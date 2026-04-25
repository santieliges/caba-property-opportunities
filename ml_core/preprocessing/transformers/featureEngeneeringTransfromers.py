from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from osmnx import features_from_place
import geopandas as gpd
from sklearn.neighbors import BallTree


class LogPrecioTransformer(
    BaseEstimator,
    TransformerMixin
):
    """
    Agrega columna log_precio usando log1p(precio).

    Esto es feature engineering estructural,
    no depende del modelo.

    Parámetros
    ----------
    precio_col : str
        Nombre de la columna original.

    output_col : str
        Nombre de la nueva columna.

    use_log1p : bool
        Si True usa log1p, si False usa log.
    """

    def __init__(
        self,
        precio_col="precio",
        output_col="log_precio",
        use_log1p=True
    ):
        self.precio_col = precio_col
        self.output_col = output_col
        self.use_log1p = use_log1p

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):

        X = X.copy()

        if self.precio_col not in X.columns:
            raise ValueError(
                f"Columna '{self.precio_col}' "
                f"no encontrada en DataFrame."
            )

        if self.use_log1p:

            X[self.output_col] = np.log1p(
                X[self.precio_col]
            )

        else:

            if (X[self.precio_col] <= 0).any():
                raise ValueError(
                    "Hay valores <= 0 en precio. "
                    "Use log1p en su lugar."
                )

            X[self.output_col] = np.log(
                X[self.precio_col]
            )

        return X


class DisposicionEncoder(
    BaseEstimator,
    TransformerMixin
):

    def __init__(self):

        self.categories = [
            'Frente',
            'Contrafrente',
            'Lateral',
            'Interno'
        ]

    def fit(self, X, y=None):

        self.mode_ = (
            X['disposicion']
            .mode()
            .iloc[0]
        )

        return self

    def transform(self, X):

        X = X.copy()

        # Fill missing
        X['disposicion'] = (
            X['disposicion']
            .fillna(self.mode_)
        )

        # Replace unknown
        X.loc[
            ~X['disposicion'].isin(self.categories),
            'disposicion'
        ] = self.mode_

        dummies = pd.get_dummies(
            pd.Categorical(
                X['disposicion'],
                categories=self.categories
            ),
            prefix='disposicion',
            dtype=int  
        )

        return pd.concat([X, dummies], axis=1)
    

class EstadoOrdinalEncoder(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        column="estado",
        output_column="estado_num"
    ):

        self.column = column
        self.output_column = output_column

        self.mapping = {
            "Excelente": 5,
            "Muy Bueno": 4,
            "Bueno": 3,
            "Regular": 2,
            "A Refaccionar": 1
        }

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):

        X = X.copy()

        X[self.output_column] = (
            X[self.column]
            .map(self.mapping)
        )

        # fallback si aparece algo raro
        X[self.output_column] = (
            X[self.output_column]
            .fillna(
                np.median(
                    list(self.mapping.values())
                )
            )
        )

        return X
    

class FeatureScaler(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        columns,
        suffix="_scaled",
    ):

        self.columns = columns
        self.suffix = suffix
        self.scaler = StandardScaler()

    def fit(self, X, y=None):

        self.scaled_columns_ = [
            f"{column}{self.suffix}"
            for column in self.columns
        ]

        self.scaler.fit(
            X[self.columns]
        )
        self.fitted_ = True

        return self

    def transform(self, X):

        X = X.copy()

        scaled_values = self.scaler.transform(
            X[self.columns]
        )

        for idx, scaled_column in enumerate(
            self.scaled_columns_
        ):
            X[scaled_column] = scaled_values[:, idx]

        return X

class DistanceToPOITransformer(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        poi_geojson_path,
        prefix,
        lon_col="longitud",
        lat_col="latitud",
        poi_name_col=None,
        return_nearest_name=True,
        earth_radius=6371000
    ):

        self.poi_geojson_path = poi_geojson_path
        self.prefix = prefix

        self.lon_col = lon_col
        self.lat_col = lat_col

        self.poi_name_col = poi_name_col
        self.return_nearest_name = return_nearest_name

        self.earth_radius = earth_radius

    def fit(self, X, y=None):

        self.poi_gdf_ = gpd.read_file(
            self.poi_geojson_path
        )

        if self.poi_gdf_.crs is None:
            self.poi_gdf_.set_crs(
                "EPSG:4326",
                inplace=True
            )

        coords = np.vstack([
            self.poi_gdf_.geometry.y,
            self.poi_gdf_.geometry.x
        ]).T

        coords_rad = np.radians(coords)

        # construir BallTree
        self.tree_ = BallTree(
            coords_rad,
            metric="haversine"
        )

        # guardar nombres si existen
        if (
            self.return_nearest_name
            and self.poi_name_col is not None
        ):
            self.poi_names_ = (
                self.poi_gdf_[
                    self.poi_name_col
                ].astype(str).values
            )

        return self

    def transform(self, X):

        X = X.copy()

        # coords propiedades
        coords = np.vstack([
            X[self.lat_col],
            X[self.lon_col]
        ]).T

        coords_rad = np.radians(coords)

        # buscar vecino más cercano
        dist, idx = self.tree_.query(
            coords_rad,
            k=1
        )

        # convertir a metros
        dist_meters = (
            dist.flatten()
            * self.earth_radius
        )

        # agregar distancia
        X[f"dist_{self.prefix}"] = dist_meters

        # agregar nombre del POI
        if (
            self.return_nearest_name
            and self.poi_name_col is not None
        ):

            nearest_names = [
                self.poi_names_[i]
                for i in idx.flatten()
            ]

            X[f"nearest_{self.prefix}"] = (
                nearest_names
            )

        return X

class DistanceToPolygonTransformer(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        polygon_path,
        prefix,
        lon_col="longitud",
        lat_col="latitud"
    ):

        self.polygon_path = polygon_path
        self.prefix = prefix
        self.lon_col = lon_col
        self.lat_col = lat_col

    def fit(self, X, y=None):

        self.polygons_ = gpd.read_file(
            self.polygon_path
        )

        if self.polygons_.crs is None:
            self.polygons_.set_crs(
                "EPSG:4326",
                inplace=True
            )

        # proyectar a metros
        self.polygons_ = self.polygons_.to_crs(
            "EPSG:3857"
        )

        return self

    def transform(self, X):

        X = X.copy()

        gdf_points = gpd.GeoDataFrame(
            X,
            geometry=gpd.points_from_xy(
                X[self.lon_col],
                X[self.lat_col]
            ),
            crs="EPSG:4326"
        )

        gdf_points = gdf_points.to_crs(
            "EPSG:3857"
        )

        joined = gpd.sjoin_nearest(
            gdf_points,
            self.polygons_,
            how="left",
            distance_col=f"dist_{self.prefix}"
        )

        joined = joined.drop(
            columns=["geometry","index_right"],
            errors="ignore"
        )
        return joined
    
from sklearn.base import BaseEstimator, TransformerMixin
import geopandas as gpd
import numpy as np
from sklearn.neighbors import BallTree


class CountNearbyPOITransformer(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        poi_geojson_path,
        radius_meters,
        prefix,
        lon_col="longitud",
        lat_col="latitud",
        earth_radius=6371000
    ):

        self.poi_geojson_path = poi_geojson_path
        self.radius_meters = radius_meters
        self.prefix = prefix

        self.lon_col = lon_col
        self.lat_col = lat_col

        self.earth_radius = earth_radius

    def fit(self, X, y=None):

        # cargar POIs
        self.poi_gdf_ = gpd.read_file(
            self.poi_geojson_path
        )

        if self.poi_gdf_.crs is None:
            self.poi_gdf_.set_crs(
                "EPSG:4326",
                inplace=True
            )

        coords = np.vstack([
            self.poi_gdf_.geometry.y,
            self.poi_gdf_.geometry.x
        ]).T

        coords_rad = np.radians(coords)
        self.tree_ = BallTree(
            coords_rad,
            metric="haversine"
        )

        # convertir radio a radianes
        self.radius_rad_ = (
            self.radius_meters
            / self.earth_radius
        )

        return self

    def transform(self, X):

        X = X.copy()

        coords = np.vstack([
            X[self.lat_col],
            X[self.lon_col]
        ]).T

        coords_rad = np.radians(coords)

        ind = self.tree_.query_radius(
            coords_rad,
            r=self.radius_rad_
        )

        # contar vecinos
        counts = np.array([
            len(i) for i in ind
        ])

        X[f"n_{self.prefix}_{self.radius_meters}m"] = counts

        return X


class SalesVelocityTransformer(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        radius_meters,
        window_days,
        prefix="ventas",
        id_col="id",
        lon_col="longitud",
        lat_col="latitud",
        valid_from_col="valido_desde",
        valid_to_col="valido_hasta",
        earth_radius=6371000,
    ):

        self.radius_meters = radius_meters
        self.window_days = window_days
        self.prefix = prefix
        self.id_col = id_col
        self.lon_col = lon_col
        self.lat_col = lat_col
        self.valid_from_col = valid_from_col
        self.valid_to_col = valid_to_col
        self.earth_radius = earth_radius

    def fit(self, X, y=None):

        sales_events = self._build_sales_events(X)

        self.count_col_ = (
            f"n_{self.prefix}_{self.radius_meters}m_"
            f"{self.window_days}d"
        )
        self.velocity_col_ = (
            f"velocidad_{self.prefix}_{self.radius_meters}m_"
            f"{self.window_days}d"
        )

        self.radius_rad_ = (
            self.radius_meters
            / self.earth_radius
        )

        if sales_events.empty:
            self.sales_events_ = sales_events
            self.tree_ = None
            self.sale_dates_ = np.array([], dtype="datetime64[ns]")
            return self

        coords = np.vstack([
            self._get_column_series(
                sales_events,
                self.lat_col,
            ),
            self._get_column_series(
                sales_events,
                self.lon_col,
            ),
        ]).T

        coords_rad = np.radians(coords)

        self.tree_ = BallTree(
            coords_rad,
            metric="haversine",
        )
        self.sale_dates_ = (
            self._get_column_series(
                sales_events,
                self.valid_to_col,
            )
            .to_numpy(dtype="datetime64[ns]")
        )
        self.sales_events_ = sales_events.reset_index(
            drop=True
        )

        return self

    def transform(self, X):

        X = X.copy()

        if X.empty:
            X[self.count_col_] = pd.Series(
                dtype=int
            )
            X[self.velocity_col_] = pd.Series(
                dtype=float
            )
            return X

        if self.tree_ is None:
            X[self.count_col_] = 0
            X[self.velocity_col_] = 0.0
            return X

        reference_dates = pd.to_datetime(
            self._get_column_series(
                X,
                self.valid_from_col,
            ),
            errors="coerce",
        ).to_numpy(dtype="datetime64[ns]")

        window_delta = np.timedelta64(
            self.window_days,
            "D",
        )

        counts = np.zeros(len(X), dtype=int)
        valid_coords_mask = (
            self._get_column_series(X, self.lat_col).notna()
            & self._get_column_series(X, self.lon_col).notna()
        ).to_numpy()

        if not valid_coords_mask.any():
            X[self.count_col_] = counts
            X[self.velocity_col_] = (
                counts / max(self.window_days, 1)
            )
            return X

        valid_coords = np.vstack([
            self._get_column_series(
                X,
                self.lat_col,
            ).loc[valid_coords_mask],
            self._get_column_series(
                X,
                self.lon_col,
            ).loc[valid_coords_mask],
        ]).T
        valid_coords_rad = np.radians(valid_coords)

        nearby_idx = self.tree_.query_radius(
            valid_coords_rad,
            r=self.radius_rad_,
        )
        valid_row_indices = np.flatnonzero(
            valid_coords_mask
        )

        for position, neighbor_idx in enumerate(nearby_idx):
            row_idx = valid_row_indices[position]
            ref_date = reference_dates[row_idx]

            if (
                np.isnat(ref_date)
                or len(neighbor_idx) == 0
            ):
                continue

            neighbor_dates = self.sale_dates_[neighbor_idx]
            valid_window = (
                (neighbor_dates <= ref_date)
                & (neighbor_dates >= ref_date - window_delta)
            )

            counts[row_idx] = int(valid_window.sum())

        X[self.count_col_] = counts
        X[self.velocity_col_] = (
            counts / max(self.window_days, 1)
        )

        return X

    def _build_sales_events(self, X):

        id_column = self._resolve_id_column(X)
        internal_id_col = "__sales_velocity_id__"
        internal_valid_to_col = "__sales_velocity_valid_to__"

        required_columns = [
            id_column,
            self.lon_col,
            self.lat_col,
            self.valid_to_col,
        ]
        missing_columns = [
            column
            for column in required_columns
            if column not in X.columns
        ]

        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(
                "Faltan columnas para calcular "
                f"SalesVelocityTransformer: {missing}"
            )

        working_df = X.copy()
        working_df[internal_id_col] = self._get_column_series(
            working_df,
            id_column,
        )
        working_df[internal_valid_to_col] = pd.to_datetime(
            self._get_column_series(
                working_df,
                self.valid_to_col,
            ),
            errors="coerce",
        )
        working_df = self._assign_column(
            working_df,
            self.valid_to_col,
            working_df[internal_valid_to_col],
        )

        if self.valid_from_col in working_df.columns:
            valid_from_series = pd.to_datetime(
                self._get_column_series(
                    working_df,
                    self.valid_from_col,
                ),
                errors="coerce",
            )
            working_df = self._assign_column(
                working_df,
                self.valid_from_col,
                valid_from_series,
            )

        closed_df = working_df.loc[
            working_df[internal_valid_to_col].notna()
        ].copy()

        if closed_df.empty:
            return closed_df

        active_ids = set()
        if self.valid_to_col in working_df.columns:
            active_ids = set(
                self._get_column_series(
                    working_df.loc[
                        working_df[internal_valid_to_col].isna()
                    ],
                    internal_id_col,
                ).dropna()
            )

        # El ultimo cierre sin version activa posterior
        # aproxima una venta efectiva, no una actualizacion.
        closed_df = closed_df.sort_values(
            [internal_id_col, internal_valid_to_col]
        )
        sales_events = (
            closed_df
            .groupby(internal_id_col, as_index=False)
            .tail(1)
        )

        if active_ids:
            sales_events = sales_events.loc[
                ~sales_events[internal_id_col].isin(active_ids)
            ]

        sales_events = sales_events.dropna(
            subset=[
                self.lon_col,
                self.lat_col,
                self.valid_to_col,
            ]
        )

        return sales_events.reset_index(drop=True)

    def _resolve_id_column(self, X):

        if self.id_col in X.columns:
            return self.id_col

        fallback_columns = [
            "id_left",
            "id",
        ]

        for column in fallback_columns:
            if column in X.columns:
                return column

        return self.id_col

    def _get_column_series(self, X, column_name):

        column = X.loc[:, column_name]

        if isinstance(column, pd.DataFrame):
            return column.iloc[:, 0]

        return column

    def _assign_column(self, X, column_name, values):

        column_positions = np.flatnonzero(
            X.columns == column_name
        )

        if len(column_positions) <= 1:
            X[column_name] = values
            return X

        X = X.copy()
        X.iloc[:, column_positions[0]] = values.to_numpy()
        return X
