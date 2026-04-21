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

    def __init__(self, columns):

        self.columns = columns
        self.scaler = StandardScaler()

    def fit(self, X, y=None):

        self.scaler.fit(
            X[self.columns]
        )
        self.fitted_ = True

        return self

    def transform(self, X):

        X = X.copy()

        X[self.columns] = (
            self.scaler.transform(
                X[self.columns]
            )
        )

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