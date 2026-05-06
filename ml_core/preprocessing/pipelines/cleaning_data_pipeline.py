from pathlib import Path

from sklearn.pipeline import Pipeline

from ml_core.preprocessing.transformers import (
    AntiguedadImputer,
    BanosImputer,
    DisposicionImputer,
    EstadoImputer,
    ExpensasImputer,
    FillZeroImputer,
    FilterRequiredPositiveTransformer,
    FilterSmallBarriosTransformer,
    NormalizeStringsTransformer,
    NumericClipper,
    RemoveOutliersTransformer,
    SpatialJoinBarriosTransformer,
)


def build_cleaning_pipeline():

    barrios_path = Path("GeoData/barrios.geojson")

    return Pipeline([
        (
            "spatial_join_barrios",
            SpatialJoinBarriosTransformer(
                barrios_path=barrios_path
            )
        ),
        (
            "normalize_strings",
            NormalizeStringsTransformer(
                columns=[
                    "estado",
                    "disposicion",
                ]
            )
        ),
        (
            "clip_antiguedad",
            NumericClipper(
                column="antiguedad",
                min_value=0,
                max_value=200,
                out_of_bounds_strategy="nan"
            )
        ),
        (
            "clip_expensas",
            NumericClipper(
                column="expensas",
                max_value=1000000
            )
        ),
        (
            "clip_banos",
            NumericClipper(
                column="banos",
                max_value=5
            )
        ),
        (
            "clip_cocheras",
            NumericClipper(
                column="cocheras",
                max_value=4
            )
        ),
        (
            "clip_area_desc",
            NumericClipper(
                column="area_m2_descubierta",
                max_value=200
            )
        ),
        (
            "impute_disposicion",
            DisposicionImputer()
        ),
        (
            "filter_required_surface",
            FilterRequiredPositiveTransformer(
                columns=[
                    "area_m2_cubierta",
                ]
            )
        ),
        (
            "filter_barrios",
            FilterSmallBarriosTransformer(
                min_size=5
            )
        ),
        (
            "remove_outliers",
            RemoveOutliersTransformer(
                columns=[
                    "precio",
                    "area_m2_total",
                ]
            )
        ),
        (
            "impute_antiguedad",
            AntiguedadImputer()
        ),
        (
            "impute_banos",
            BanosImputer()
        ),
        (
            "impute_expensas",
            ExpensasImputer()
        ),
        (
            "impute_estado",
            EstadoImputer()
        ),
        (
            "fill_remaining_nans",
            FillZeroImputer()
        ),
        
    ])
