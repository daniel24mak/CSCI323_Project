import numpy as np
import pandas as pd

from src.models.regression import (
    train_regression,
    predict_expense,
)

def create_mock_data():

    X_train = pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 45, 50, 55],
        "bmi": [22, 24, 26, 28, 30, 32, 34, 36],
        "children": [0, 1, 0, 2, 1, 3, 2, 1],
        "gender": [0, 1, 0, 1, 0, 1, 0, 1],
        "discount_eligibility": [1, 0, 1, 0, 1, 0, 1, 0],
    })

    y_train = pd.Series([
        2000,
        3000,
        5000,
        7000,
        9000,
        12000,
        15000,
        18000,
    ])

    X_val = pd.DataFrame({
        "age": [28, 38],
        "bmi": [25, 31],
        "children": [1, 2],
        "gender": [0, 1],
        "discount_eligibility": [1, 0],
    })

    y_val = pd.Series([
        4500,
        10000,
    ])

    return X_train, y_train, X_val, y_val


def test_train_regression_contract_keys():

    X_train, y_train, X_val, y_val = create_mock_data()

    result = train_regression(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    expected_keys = {
        "best_model",
        "best_model_name",
        "train_metrics",
        "val_metrics",
        "predictions",
        "comparison_table",
    }

    assert set(result.keys()) == expected_keys


def test_predictions_length_matches_validation():

    X_train, y_train, X_val, y_val = create_mock_data()

    result = train_regression(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    assert len(result["predictions"]) == len(X_val)


def test_metrics_are_finite():

    X_train, y_train, X_val, y_val = create_mock_data()

    result = train_regression(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    metrics = result["val_metrics"]

    assert np.isfinite(metrics["rmse"])
    assert np.isfinite(metrics["mae"])
    assert np.isfinite(metrics["r2"])


def test_predict_expense_single_row():

    X_train, y_train, X_val, y_val = create_mock_data()

    result = train_regression(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    model = result["best_model"]

    single_customer = X_val.iloc[[0]]

    prediction = predict_expense(
        model,
        single_customer,
    )

    assert isinstance(prediction, float)
    assert prediction > 0
