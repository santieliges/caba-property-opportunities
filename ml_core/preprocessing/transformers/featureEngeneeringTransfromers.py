from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


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