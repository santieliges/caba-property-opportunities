from sklearn.pipeline import Pipeline

from ml_core.preprocessing.pipelines.cleaningDataPipeline import (
    build_cleaning_pipeline,
)
from ml_core.preprocessing.pipelines.featureEngeneeringPipeline import (
    build_feature_engineering_pipeline,
)


def build_preprocessing_pipeline():

    return Pipeline([
        (
            "cleaning",
            build_cleaning_pipeline()
        ),
        (
            "feature_engineering",
            build_feature_engineering_pipeline()
        ),
    ])
