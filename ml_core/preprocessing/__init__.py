# Pipelines

from .pipeline_builder import (
    build_cleaning_pipeline,
    build_feature_engineering_pipeline,
    build_preprocessing_pipeline,
)


# Splitting

from .splitting import (
    build_venta_splits,
    build_alquiler_splits,
)



# Transformers (opcional re-export)

from .transformers import (
    NormalizeStringsTransformer,
    FilterSmallBarriosTransformer,
    AntiguedadImputer,
    EstadoImputer,
    DisposicionEncoder,
)
from .knhs import KNHSSchema, KNHSWeightSpec, PreparedKNHS


__all__ = [

    # Pipelines
    "build_cleaning_pipeline",
    "build_feature_engineering_pipeline",
    "build_preprocessing_pipeline",

    # Splits
    "build_venta_splits",
    "build_alquiler_splits",


    # Transformers
    "NormalizeStringsTransformer",
    "FilterSmallBarriosTransformer",
    "AntiguedadImputer",
    "EstadoImputer",
    "DisposicionEncoder",
    "KNHSSchema",
    "KNHSWeightSpec",
    "PreparedKNHS",

]
