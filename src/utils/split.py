import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_test_split(
    df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
):

    total = train_size + val_size + test_size

    if round(total, 5) != 1.0:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        random_state=random_state,
        shuffle=True,
    )

    relative_test_size = test_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=random_state,
        shuffle=True,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )