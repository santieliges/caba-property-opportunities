import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    residuals = y_true - y_pred

    mask = y_true != 0

    if mask.sum() > 0:
        mape = np.mean(
            np.abs(residuals[mask] / y_true[mask])
        ) * 100
    else:
        mape = np.nan

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float(residuals.mean()),
        "median_abs_error": float(np.median(np.abs(residuals))),
        "mape": float(mape),
    }