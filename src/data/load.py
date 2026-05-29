from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = [
    "age",
    "gender",
    "bmi",
    "children",
    "discount_eligibility",
    "region",
    "expenses",
    "premium",
]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def load_raw_data(csv_path: str | Path) -> pd.DataFrame:

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    df = normalize_column_names(df)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[REQUIRED_COLUMNS].copy()

    numeric_columns = ["age", "bmi", "children", "expenses", "premium"]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df