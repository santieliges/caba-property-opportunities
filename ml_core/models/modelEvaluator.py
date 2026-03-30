import numpy as np


class ModelEvaluator:

    def evaluate(self, y_true, y_pred):
        metrics = {}

        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)

        residuals = y_true - y_pred


        metrics["rmse"] = np.sqrt(np.mean(residuals**2))
        metrics["mae"] = np.mean(np.abs(residuals))
        metrics["r2"] = 1 - np.sum(residuals**2) / np.sum((y_true - y_true.mean())**2)


        return metrics
