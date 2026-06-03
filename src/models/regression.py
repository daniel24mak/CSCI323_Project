import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def _evaluate_model(model, X, y):
    """
    Evaluate a trained regression model.
    """
    predictions = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "predictions": predictions,
    }


def train_regression(X_train, y_train, X_val, y_val):
    """
    Train Linear Regression, Ridge and Lasso models.

    Returns the required M3 contract dictionary.
    """

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=1.0),
    }

    comparison_rows = []

    best_model = None
    best_model_name = None
    best_r2 = float("-inf")

    best_train_metrics = None
    best_val_metrics = None

    for model_name, model in models.items():

        model.fit(X_train, y_train)

        train_results = _evaluate_model(
            model,
            X_train,
            y_train,
        )

        val_results = _evaluate_model(
            model,
            X_val,
            y_val,
        )

        comparison_rows.append({
            "Model": model_name,
            "Train RMSE": train_results["rmse"],
            "Train MAE": train_results["mae"],
            "Train R²": train_results["r2"],
            "Val RMSE": val_results["rmse"],
            "Val MAE": val_results["mae"],
            "Val R²": val_results["r2"],
        })

        if val_results["r2"] > best_r2:
            best_r2 = val_results["r2"]

            best_model = model
            best_model_name = model_name

            best_train_metrics = train_results
            best_val_metrics = val_results

    comparison_table = pd.DataFrame(comparison_rows)

    return {
        "best_model": best_model,
        "best_model_name": best_model_name,

        "train_metrics": {
            "rmse": best_train_metrics["rmse"],
            "mae": best_train_metrics["mae"],
            "r2": best_train_metrics["r2"],
        },

        "val_metrics": {
            "rmse": best_val_metrics["rmse"],
            "mae": best_val_metrics["mae"],
            "r2": best_val_metrics["r2"],
        },

        "predictions": best_val_metrics["predictions"],

        "comparison_table": comparison_table,
    }


def predict_expense(model, X_new):
    """
    Predict healthcare expense for new customer(s).
    """

    prediction = model.predict(X_new)[0]

    return float(max(prediction, 0))
