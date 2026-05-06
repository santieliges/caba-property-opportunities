from sklearn.pipeline import Pipeline

from ml_core.preprocessing.transformers import (
    LogPrecioTransformer,
)
from ml_core.preprocessing.transformers import (DisposicionEncoder, 
                                                EstadoOrdinalEncoder,
                                                FeatureScaler,
                                                DistanceToPOITransformer,
                                                DistanceToPolygonTransformer,
                                                CountNearbyPOITransformer,
                                                SalesVelocityTransformer)



def build_feature_engineering_pipeline():


    return Pipeline([
            (
            "dist_subte",
            DistanceToPOITransformer(
                poi_geojson_path="GeoData/estaciones_de_subte.geojson",
                prefix="subte",
                poi_name_col="estacion"
            )
        ),

        (
            "dist_Universidad",
            DistanceToPOITransformer(
                poi_geojson_path="GeoData/universidades.geojson",
                prefix="universidad",
                poi_name_col="nombre"
            )
        ),

        (
            "dist_hospital",
            DistanceToPOITransformer(
                poi_geojson_path="GeoData/hospitales.geojson",
                prefix="hospital",
                poi_name_col="fna"
            )
        ),

        (
            "dist_est_educativo",
            DistanceToPOITransformer(
                poi_geojson_path="GeoData/establecimientos_educativos.geojson",
                prefix="est_educativo",
                poi_name_col="fna"
            )
        ),
        (
            "dist_espacio_verde",
            DistanceToPolygonTransformer(
                polygon_path="GeoData/espacio_verde_publico.geojson",
                prefix="espacio_verde"
            )   
        ),
        (
            "dist_areas_programaticas",
            DistanceToPolygonTransformer(
                polygon_path="GeoData/areas_programaticas.geojson",
                prefix="areas_programaticas"
            )   
        ),
        (
            "dist_avenida_rivadavia",
            DistanceToPolygonTransformer(
                polygon_path="GeoData/avenida_rivadavia.geojson",
                prefix="avenida_rivadavia"
            )   
        ),
        (
            "universidades_1km",
            CountNearbyPOITransformer(
                poi_geojson_path="GeoData/universidades.geojson",
                radius_meters=1000,
                prefix="universidades"
            )
        ),
        (
            "delitos_1km",
            CountNearbyPOITransformer(
                poi_geojson_path="GeoData/Delitos_Diciembre_2024.geojson",
                radius_meters=1000,
                prefix="robos"
            )
        ),
        (
            "velocidad_ventas_1km_90d",
            SalesVelocityTransformer(
                radius_meters=1000,
                window_days=90,
                prefix="ventas"
            )
        ),
        (
            "encode_estado_ordinal",
            EstadoOrdinalEncoder()
        ),
        (
            "encode_disposicion",
            DisposicionEncoder()
        ),
        (
            "log_precio",
            LogPrecioTransformer()
        ),
        ("scale_features", FeatureScaler(
            columns=[
                "area_m2_total",
                "area_m2_descubierta",
                "ambientes",
                "banos",
                "cocheras",
                "antiguedad",
                "expensas",
                "estado_num",
                'dist_subte',
                'dist_universidad',
                'dist_hospital',
                'dist_est_educativo',
                'dist_espacio_verde',
                'dist_areas_programaticas',
                'dist_avenida_rivadavia',
                "n_robos_1000m",
                "n_universidades_1000m",
                "n_ventas_1000m_90d",
                "velocidad_ventas_1000m_90d",
            ]
        )),
    ])
