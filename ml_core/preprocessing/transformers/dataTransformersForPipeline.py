from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import geopandas as gpd



class NormalizeStringsTransformer(
    BaseEstimator,
    TransformerMixin
):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        for col in self.columns:
            if col in X.columns:
                X[col] = (
                    X[col]
                    .astype(str)
                    .str.strip()
                )

        return X
    
class FilterSmallBarriosTransformer(
    BaseEstimator,
    TransformerMixin
):

    def __init__(self, min_size=5):
        self.min_size = min_size

    def fit(self, X, y=None):

        counts = X['barrio'].value_counts()

        self.valid_barrios_ = counts[
            counts >= self.min_size
        ].index

        return self

    def transform(self, X):

        X = X.copy()

        return X[
            X['barrio'].isin(
                self.valid_barrios_
            )
        ]
class AntiguedadImputer(
    BaseEstimator,
    TransformerMixin
):

    def fit(self, X, y=None):

        X = X.copy()

        # 🔧 Forzar antiguedad a numérico
        X['antiguedad'] = pd.to_numeric(
            X['antiguedad'],
            errors="coerce"
        )

        self.mean_by_barrio_ = (
            X.groupby('barrio')['antiguedad']
            .mean()
        )

        self.global_mode_ = (
            X['antiguedad']
            .mode()
            .iloc[0]
        )

        return self

    def transform(self, X):

        X = X.copy()

        # 🔧 Forzar antiguedad a numérico también aquí
        X['antiguedad'] = pd.to_numeric(
            X['antiguedad'],
            errors="coerce"
        )

        # Imputar por barrio
        X['antiguedad'] = X['antiguedad'].fillna(
            X['barrio'].map(
                self.mean_by_barrio_
            )
        )

        # Imputar global
        X['antiguedad'] = X['antiguedad'].fillna(
            self.global_mode_
        )

        return X

class EstadoImputer(
    BaseEstimator,
    TransformerMixin
):

    def fit(self, X, y=None):

        X = X.copy()

        X['antiguedad_cat'] = (
            X['antiguedad']
            .round()
            .astype('Int64')
        )

        self.estado_lookup_ = (
            X.dropna(subset=['estado'])
            .groupby(
                ['barrio','antiguedad_cat']
            )['estado']
            .agg(lambda x: x.mode().iloc[0])
        )

        self.global_mode_ = (
            X['estado']
            .mode()
            .iloc[0]
        )

        return self

    def transform(self, X):

        X = X.copy()

        X['antiguedad_cat'] = (
            X['antiguedad']
            .round()
            .astype('Int64')
        )

        mask = X['estado'].isna()

        X.loc[mask, 'estado'] = (
            X.loc[mask]
            .set_index(
                ['barrio','antiguedad_cat']
            )
            .index
            .map(self.estado_lookup_)
        )

        X['estado'] = X['estado'].fillna(
            self.global_mode_
        )

        return X

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

class ExpensasImputer(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        column="expensas",
        fill_value=0
    ):
        self.column = column
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        if self.column in X.columns:

            X[self.column] = pd.to_numeric(
                X[self.column],
                errors="coerce"
            )

            X[self.column] = (
                X[self.column]
                .fillna(self.fill_value)
            )

        return X    
    
class BanosImputer(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        column="banos",
        fallback_value=1
    ):
        self.column = column
        self.fallback_value = fallback_value

    def fit(self, X, y=None):

        X = X.copy()

        # Convertir a numérico
        X[self.column] = pd.to_numeric(
            X[self.column],
            errors="coerce"
        )

        if X[self.column].dropna().empty:

            self.mode_ = self.fallback_value

        else:

            self.mode_ = (
                X[self.column]
                .mode()
                .iloc[0]
            )

        return self

    def transform(self, X):

        X = X.copy()

        X[self.column] = pd.to_numeric(
            X[self.column],
            errors="coerce"
        )

        X[self.column] = (
            X[self.column]
            .fillna(self.mode_)
        )

        return X
    
class DisposicionImputer(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        column="disposicion"
    ):

        self.column = column

        self.valid_categories = [
            'Frente',
            'Contrafrente',
            'Lateral',
            'Interno'
        ]

    def fit(self, X, y=None):

        X = X.copy()

        # Normalizar
        X[self.column] = (
            X[self.column]
            .astype(str)
            .str.strip()
        )

        # Convertir strings raros
        X[self.column] = (
            X[self.column]
            .replace(["nan","None",""], np.nan)
        )

        self.mode_ = (
            X[self.column]
            .mode()
            .iloc[0]
        )

        return self

    def transform(self, X):

        X = X.copy()

        X[self.column] = (
            X[self.column]
            .astype(str)
            .str.strip()
        )

        X[self.column] = (
            X[self.column]
            .replace(["nan","None",""], np.nan)
        )

        # Fill NaN
        X[self.column] = (
            X[self.column]
            .fillna(self.mode_)
        )

        # Reemplazar inválidos
        X.loc[
            ~X[self.column].isin(self.valid_categories),
            self.column
        ] = self.mode_

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

        # 🔧 CLAVE: usar categorías fijas
        dummies = pd.get_dummies(
            pd.Categorical(
                X['disposicion'],
                categories=self.categories
            ),
            prefix='disposicion',
            dtype=int   # 🔧 CLAVE
        )

        return pd.concat([X, dummies], axis=1)
    


class RemoveOutliersTransformer(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        columns,
        lower=0.01,
        upper=0.99
    ):
        self.columns = columns
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):

        self.bounds_ = {}

        for col in self.columns:

            col_values = pd.to_numeric(X[col], errors="coerce")
            q_low = col_values.quantile(self.lower)
            q_high = col_values.quantile(self.upper)

            self.bounds_[col] = (
                q_low,
                q_high
            )

        return self

    def transform(self, X):

        X = X.copy()

        mask = np.ones(len(X), dtype=bool)

        for col in self.columns:

            low, high = self.bounds_[col]
            col_values = pd.to_numeric(X[col], errors="coerce")

            mask &= (
                (col_values >= low) &
                (col_values <= high)
            )

        return X[mask].reset_index(drop=True)
    
class SpatialJoinBarriosTransformer(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        barrios_path,
        lon_col="longitud",
        lat_col="latitud",
        barrio_source_col="nombre",  # ← nombre real en geojson
        barrio_target_col="barrio",  # ← nombre estándar del pipeline
        drop_outside=True,
        verbose=True
    ):

        self.barrios_path = barrios_path
        self.lon_col = lon_col
        self.lat_col = lat_col

        self.barrio_source_col = barrio_source_col
        self.barrio_target_col = barrio_target_col

        self.drop_outside = drop_outside
        self.verbose = verbose

    def fit(self, X, y=None):

        self.barrios_ = gpd.read_file(
            self.barrios_path
        )

        # Validar que exista la columna fuente

        if self.barrio_source_col not in self.barrios_.columns:

            raise ValueError(
                f"La columna '{self.barrio_source_col}' "
                f"no existe en barrios.geojson.\n"
                f"Columnas disponibles: "
                f"{self.barrios_.columns.tolist()}"
            )

        if self.barrios_.crs is None:

            self.barrios_.set_crs(
                "EPSG:4326",
                inplace=True
            )

        return self

    def transform(self, X):

        X = X.copy()

        n_before = len(X)

        gdf = gpd.GeoDataFrame(
            X,
            geometry=gpd.points_from_xy(
                X[self.lon_col],
                X[self.lat_col]
            ),
            crs="EPSG:4326"
        )

        gdf = gpd.sjoin(
            gdf,
            self.barrios_,
            how="left",
            predicate="within"
        )

        gdf = gdf.rename(
            columns={
                self.barrio_source_col:
                self.barrio_target_col
            }
        )

        if self.barrio_target_col not in gdf.columns:

            raise ValueError(
                f"No se pudo crear columna "
                f"{self.barrio_target_col}"
            )

        outside_mask = gdf[
            self.barrio_target_col
        ].isna()

        n_outside = outside_mask.sum()

        if self.verbose:

            print(
                f"[SpatialJoin] "
                f"Puntos fuera barrios: "
                f"{n_outside}/{n_before}"
            )

        if self.drop_outside:

            gdf = gdf[~outside_mask]

            if self.verbose:

                print(
                    f"[SpatialJoin] "
                    f"Filas restantes: "
                    f"{len(gdf)}"
                )

        gdf = gdf.drop(
            columns=[
                "geometry",
                "index_right"
            ],
            errors="ignore"
        )

        return pd.DataFrame(gdf)

class FillZeroImputer(
    BaseEstimator,
    TransformerMixin
):

    def __init__(self, columns=None):

        # Si columns=None → se detectan automáticamente
        self.columns = columns

    def fit(self, X, y=None):

        if self.columns is None:

            self.columns_ = (
                X.select_dtypes(
                    include=[np.number]
                ).columns
            )

        else:

            self.columns_ = self.columns

        return self

    def transform(self, X):

        X = X.copy()

        X[self.columns_] = (
            X[self.columns_]
            .fillna(0)
        )

        return X

class NumericClipper(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        column,
        max_value
    ):

        self.column = column
        self.max_value = max_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X[self.column] = np.clip(
            X[self.column],
            0,
            self.max_value
        )

        return X    

class AntiguedadCleaner(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        max_valid=200
    ):
        """
        Limpia valores absurdos de antigüedad.

        Parámetros
        ----------
        max_valid : int
            Máximo valor válido permitido para antigüedad.
            Valores mayores se reemplazan por NaN.
        """

        self.max_valid = max_valid

    def fit(self, X, y=None):

        # No aprende nada, solo guarda parámetro
        return self

    def transform(self, X):

        X = X.copy()

        # Asegurar tipo numérico
        X["antiguedad"] = pd.to_numeric(
            X["antiguedad"],
            errors="coerce"
        )

        # Valores absurdos → NaN
        X.loc[
            X["antiguedad"] > self.max_valid,
            "antiguedad"
        ] = np.nan

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

        return self

    def transform(self, X):

        X = X.copy()

        X[self.columns] = (
            self.scaler.transform(
                X[self.columns]
            )
        )

        return X