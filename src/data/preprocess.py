from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.load import load_raw_data
from src.utils.split import train_val_test_split


RANDOM_STATE = 42
TARGET_COLUMN = "expenses"

NUMERIC_FEATURES = ["age", "bmi", "children"]
BINARY_FEATURES = ["gender", "discount_eligibility"]
CATEGORICAL_FEATURES = ["region"]

# expenses is the target, so it must not be inside X.
# premium is also excluded because M4 will recommend a premium later.
EXCLUDED_FROM_FEATURES = ["expenses", "premium"]


def _clean_text_value(value) -> str:
    """
    Cleans text values for consistent encoding.
    """
    return str(value).strip().lower().replace(" ", "_")


def _encode_binary_column(series: pd.Series, column_name: str) -> pd.Series:
    """
    Encodes binary categorical columns.
    """
    cleaned = series.apply(_clean_text_value)

    if column_name == "gender":
        mapping = {
            "male": 1,
            "m": 1,
            "man": 1,
            "1": 1,
            "female": 0,
            "f": 0,
            "woman": 0,
            "0": 0,
        }

    elif column_name == "discount_eligibility":
        mapping = {
            "yes": 1,
            "y": 1,
            "true": 1,
            "eligible": 1,
            "1": 1,
            "no": 0,
            "n": 0,
            "false": 0,
            "not_eligible": 0,
            "0": 0,
        }

    else:
        raise ValueError(f"No binary encoding rule defined for column: {column_name}")

    encoded = cleaned.map(mapping)

    if encoded.isna().any():
        bad_values = sorted(cleaned[encoded.isna()].unique())
        raise ValueError(f"Unexpected values in {column_name}: {bad_values}")

    return encoded.astype(int)


def _clean_region_column(series: pd.Series) -> pd.Series:
    """
    Cleans region values before one-hot encoding.
    """
    return series.apply(_clean_text_value)


def _fit_preprocessing_artifacts(train_df: pd.DataFrame) -> dict:

    numeric_medians = train_df[NUMERIC_FEATURES].median()

    category_modes = {}

    for col in BINARY_FEATURES + CATEGORICAL_FEATURES:
        mode_values = train_df[col].dropna().mode()

        if mode_values.empty:
            raise ValueError(f"Column {col} has no valid values to calculate mode.")

        category_modes[col] = mode_values.iloc[0]

    temp_train = train_df.copy()

    for col in NUMERIC_FEATURES:
        temp_train[col] = temp_train[col].fillna(numeric_medians[col])

    for col in BINARY_FEATURES + CATEGORICAL_FEATURES:
        temp_train[col] = temp_train[col].fillna(category_modes[col])

    temp_train["region"] = _clean_region_column(temp_train["region"])

    region_dummies = pd.get_dummies(temp_train["region"], prefix="region")
    region_columns = list(region_dummies.columns)

    scaler = StandardScaler()
    scaler.fit(temp_train[NUMERIC_FEATURES])

    return {
        "numeric_medians": numeric_medians,
        "category_modes": category_modes,
        "region_columns": region_columns,
        "scaler": scaler,
    }


def _transform_features(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Applies the fitted preprocessing artifacts to a dataset split.
    """
    df = df.copy()

    for col in NUMERIC_FEATURES:
        df[col] = df[col].fillna(artifacts["numeric_medians"][col])

    for col in BINARY_FEATURES + CATEGORICAL_FEATURES:
        df[col] = df[col].fillna(artifacts["category_modes"][col])

    df["region"] = _clean_region_column(df["region"])

    X_numeric = pd.DataFrame(
        artifacts["scaler"].transform(df[NUMERIC_FEATURES]),
        columns=NUMERIC_FEATURES,
        index=df.index,
    )

    X_binary = pd.DataFrame(index=df.index)

    for col in BINARY_FEATURES:
        X_binary[col] = _encode_binary_column(df[col], col)

    X_region = pd.get_dummies(df["region"], prefix="region")
    X_region = X_region.reindex(
        columns=artifacts["region_columns"],
        fill_value=0,
    )

    X = pd.concat([X_numeric, X_binary, X_region], axis=1)
    X = X.astype(float)

    return X.reset_index(drop=True)


def get_preprocessed_data(
    csv_path: str | Path,
    save_artifacts: bool = True,
    artifacts_path: str | Path = "models/preprocessor_artifacts.pkl",
) -> dict:

    df = load_raw_data(csv_path)

    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    train_df, val_df, test_df = train_val_test_split(
        df,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        random_state=RANDOM_STATE,
    )

    artifacts = _fit_preprocessing_artifacts(train_df)

    X_train = _transform_features(train_df, artifacts)
    X_val = _transform_features(val_df, artifacts)
    X_test = _transform_features(test_df, artifacts)

    y_train = train_df[TARGET_COLUMN].astype(float).reset_index(drop=True)
    y_val = val_df[TARGET_COLUMN].astype(float).reset_index(drop=True)
    y_test = test_df[TARGET_COLUMN].astype(float).reset_index(drop=True)

    data = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": list(X_train.columns),
    }

    if save_artifacts:
        artifacts_path = Path(artifacts_path)
        artifacts_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifacts, artifacts_path)

    return data


def save_preprocessed_outputs(
    data: dict,
    output_dir: str | Path = "data/processed",
) -> None:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data["X_train"].to_csv(output_dir / "X_train.csv", index=False)
    data["X_val"].to_csv(output_dir / "X_val.csv", index=False)
    data["X_test"].to_csv(output_dir / "X_test.csv", index=False)

    data["y_train"].to_frame(name=TARGET_COLUMN).to_csv(
        output_dir / "y_train.csv",
        index=False,
    )

    data["y_val"].to_frame(name=TARGET_COLUMN).to_csv(
        output_dir / "y_val.csv",
        index=False,
    )

    data["y_test"].to_frame(name=TARGET_COLUMN).to_csv(
        output_dir / "y_test.csv",
        index=False,
    )

    pd.DataFrame({"feature_names": data["feature_names"]}).to_csv(
        output_dir / "feature_names.csv",
        index=False,
    )

    print(f"Preprocessed CSV files saved to: {output_dir}")


if __name__ == "__main__":
    csv_path = "data/raw/medical_insurance.csv"

    data = get_preprocessed_data(csv_path)
    save_preprocessed_outputs(data, "data/processed")

    print("\nPreprocessing completed successfully.")
    print("Output keys:", list(data.keys()))
    print("X_train shape:", data["X_train"].shape)
    print("X_val shape:", data["X_val"].shape)
    print("X_test shape:", data["X_test"].shape)
    print("y_train shape:", data["y_train"].shape)
    print("y_val shape:", data["y_val"].shape)
    print("y_test shape:", data["y_test"].shape)
    print("Feature names:", data["feature_names"])

    print("\nM2 clustering input example:")
    print(data["X_train"][["age", "bmi"]].head())

    print("\nM3 regression input example:")
    print(data["X_train"].head())
    print(data["y_train"].head())