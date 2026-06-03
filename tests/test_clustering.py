import numpy as np
import pandas as pd

from src.models.clustering import run_clustering, get_cluster_model


def create_mock_data():
    return pd.DataFrame({
        "age": [20, 22, 25, 35, 40, 45, 55, 60, 65, 70],
        "bmi": [21, 22, 23, 27, 28, 29, 32, 34, 35, 37],
    })


def test_run_clustering_contract_keys():
    X = create_mock_data()

    result = run_clustering(X, k=3)

    expected_keys = {
        "labels",
        "risk_tier",
        "k",
        "inertia",
        "silhouette_score",
        "cluster_centers",
    }

    assert set(result.keys()) == expected_keys


def test_labels_length_matches_input():
    X = create_mock_data()

    result = run_clustering(X, k=3)

    assert len(result["labels"]) == len(X)


def test_risk_tier_values_are_valid():
    X = create_mock_data()

    result = run_clustering(X, k=3)

    allowed_labels = {"LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"}

    assert all(label in allowed_labels for label in result["risk_tier"])


def test_silhouette_score_range():
    X = create_mock_data()

    result = run_clustering(X, k=3)

    assert isinstance(result["silhouette_score"], float)
    assert -1 <= result["silhouette_score"] <= 1


def test_cluster_centers_shape():
    X = create_mock_data()

    result = run_clustering(X, k=3)

    assert result["cluster_centers"].shape == (3, 2)


def test_get_cluster_model_predicts_single_row():
    X = create_mock_data()

    run_clustering(X, k=3)
    model = get_cluster_model()

    new_customer = pd.DataFrame({
        "age": [30],
        "bmi": [25],
    })

    prediction = model.predict(new_customer)

    assert len(prediction) == 1
    assert isinstance(prediction[0], (np.integer, int))