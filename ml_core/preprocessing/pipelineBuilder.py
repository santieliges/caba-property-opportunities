# ml_core/preprocessing/pipeline_builder.py

from sklearn.pipeline import Pipeline
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from ml_core.preprocessing.transformers import LogPrecioTransformer

from ml_core.preprocessing.transformers import (
    NormalizeStringsTransformer,
    FilterSmallBarriosTransformer,
    AntiguedadImputer,
    EstadoImputer,
    DisposicionEncoder,
    RemoveOutliersTransformer,
    SpatialJoinBarriosTransformer,
    BanosImputer,
    DisposicionImputer,
    ExpensasImputer,
    EstadoOrdinalEncoder,
    FillZeroImputer,
    AntiguedadCleaner,
    FeatureScaler, 
    NumericClipper
)
def build_cleaning_pipeline():

    barrios_path = (
        Path("barrios.geojson")
    )

    pipeline = Pipeline([

        # =========================================================
        # 1️⃣ SPATIAL FEATURE
        # =========================================================

        (
            "spatial_join_barrios",
            SpatialJoinBarriosTransformer(
                barrios_path=barrios_path
            )
        ),

        # =========================================================
        # 2️⃣ LIMPIEZA STRINGS
        # =========================================================

        (
            "normalize_strings",
            NormalizeStringsTransformer(
                columns=[
                    "estado",
                    "disposicion"
                ]
            )
        ),

        # =========================================================
        # 🆕 3️⃣ LIMPIEZA NUMÉRICA CRÍTICA
        # =========================================================

        (
            "clean_antiguedad",
            AntiguedadCleaner(
                max_valid=200
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

        # =========================================================
        # 4️⃣ IMPUTACIONES SIMPLES
        # =========================================================

        (
            "impute_disposicion",
            DisposicionImputer()
        ),

        # =========================================================
        # 5️⃣ FILTRADO BARRIOS
        # =========================================================

        (
            "filter_barrios",
            FilterSmallBarriosTransformer(
                min_size=5
            )
        ),

        # =========================================================
        # 6️⃣ REMOCIÓN OUTLIERS BÁSICOS
        # =========================================================

        (
            "remove_outliers",
            RemoveOutliersTransformer(
                columns=[
                    "precio",
                    "area_m2_total"
                ]
            )
        ),

        # =========================================================
        # 7️⃣ IMPUTACIONES DEPENDIENTES
        # =========================================================

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

        # =========================================================
        # 8️⃣ ENCODING
        # =========================================================

        (
            "encode_estado_ordinal",
            EstadoOrdinalEncoder()
        ),

        (
            "encode_disposicion",
            DisposicionEncoder()
        ),

        # =========================================================
        # 9️⃣ RELLENAR NaNs RESTANTES
        # =========================================================

        (
            "fill_remaining_nans",
            FillZeroImputer()
        ),

        (
            "scale_features",
            FeatureScaler(
                columns=[
                    "area_m2_total",
                    "area_m2_descubierta",
                    "ambientes",
                    "banos",
                    "cocheras",
                    "antiguedad",
                    "expensas",
                ]
            )
        ),
        # =========================================================
        # 🔟 FEATURE ENGINEERING
        # =========================================================

        (
            "add_log_precio",
            LogPrecioTransformer()
        ),

        # =========================================================
        # 11️⃣ FILTRADO FINAL
        # =========================================================

        (
            "remove_outliers_log_precio",
            RemoveOutliersTransformer(
                columns=[
                    "log_precio"
                ]
            )
        ),


    ])

    return pipeline