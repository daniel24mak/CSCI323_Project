from pathlib import Path
import numpy as np

from src.data.load import load_raw_data
from src.data.preprocess import get_preprocessed_data


CSV_PATH = Path("data/raw/medical_insurance.csv")


def test_raw_dataset_exists():
    assert CSV_PATH.exists(), f"Dataset not found at {CSV_PATH}"


def test_raw_dataset_shape_and_columns():
    df = load_raw_data(CSV_PATH)

    assert df.shape == (1338, 8)

    expected_columns = [
        "age",
        "gender",
        "bmi",
        "children",
        "discount_eligibility",
        "region",
        "expenses",
        "premium",
    ]

    assert list(df.columns) == expected_columns


def test_preprocess_output_keys():
    data = get_preprocessed_data(CSV_PATH)

    expected_keys = {
        "X_train",
        "X_val",
        "X_test",
        "y_train",
        "y_val",
        "y_test",
        "feature_names",
    }

    assert set(data.keys()) == expected_keys


def test_no_missing_values_after_preprocessing():
    data = get_preprocessed_data(CSV_PATH)

    for key in ["X_train", "X_val", "X_test"]:
        assert data[key].isna().sum().sum() == 0

    for key in ["y_train", "y_val", "y_test"]:
        assert data[key].isna().sum() == 0


def test_split_lengths_sum_to_1338():
    data = get_preprocessed_data(CSV_PATH)

    total_rows = (
        len(data["X_train"])
        + len(data["X_val"])
        + len(data["X_test"])
    )

    assert total_rows == 1338


def test_target_and_premium_not_inside_features():
    data = get_preprocessed_data(CSV_PATH)

    for key in ["X_train", "X_val", "X_test"]:
        assert "expenses" not in data[key].columns
        assert "premium" not in data[key].columns


def test_required_model_features_exist():
    data = get_preprocessed_data(CSV_PATH)

    required_features = [
        "age",
        "bmi",
        "children",
        "gender",
        "discount_eligibility",
    ]

    for feature in required_features:
        assert feature in data["X_train"].columns


def test_region_was_one_hot_encoded():
    data = get_preprocessed_data(CSV_PATH)

    region_columns = [
        col for col in data["X_train"].columns
        if col.startswith("region_")
    ]

    assert len(region_columns) > 0


def test_feature_names_match_x_train_columns():
    data = get_preprocessed_data(CSV_PATH)

    assert data["feature_names"] == list(data["X_train"].columns)


def test_feature_values_are_finite():
    data = get_preprocessed_data(CSV_PATH)

    for key in ["X_train", "X_val", "X_test"]:
        assert np.isfinite(data[key].values).all()

    for key in ["y_train", "y_val", "y_test"]:
        assert np.isfinite(data[key].values).all()


def test_m2_can_access_clustering_features():
    data = get_preprocessed_data(CSV_PATH)

    X_cluster = data["X_train"][["age", "bmi"]]

    assert list(X_cluster.columns) == ["age", "bmi"]
    assert len(X_cluster) == len(data["X_train"])


def test_m3_can_access_regression_data():
    data = get_preprocessed_data(CSV_PATH)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)