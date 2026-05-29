from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.load import load_raw_data


def save_numeric_distribution_plots(df: pd.DataFrame, output_dir: Path) -> None:

    numeric_columns = ["age", "bmi", "expenses", "premium"]

    for col in numeric_columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / f"{col}_distribution.png", dpi=300)
        plt.close()


def save_categorical_count_plots(df: pd.DataFrame, output_dir: Path) -> None:

    categorical_columns = ["gender", "children", "discount_eligibility", "region"]

    for col in categorical_columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=col)
        plt.title(f"Count Plot of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(output_dir / f"{col}_countplot.png", dpi=300)
        plt.close()


def save_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:

    numeric_df = df[["age", "bmi", "children", "expenses", "premium"]]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=300)
    plt.close()


def save_pairplot(df: pd.DataFrame, output_dir: Path) -> None:

    pairplot = sns.pairplot(
        df,
        vars=["age", "bmi", "expenses", "premium"],
        hue="region",
        diag_kind="hist",
    )

    pairplot.fig.suptitle("Pairplot Colored by Region", y=1.02)
    pairplot.savefig(output_dir / "pairplot_by_region.png", dpi=300)
    plt.close("all")


def save_eda_summary(df: pd.DataFrame, output_tables_dir: Path) -> None:

    output_tables_dir.mkdir(parents=True, exist_ok=True)

    summary = df.describe(include="all")
    missing_values = df.isna().sum().to_frame(name="missing_count")
    dtypes = df.dtypes.astype(str).to_frame(name="dtype")

    summary.to_csv(output_tables_dir / "eda_summary_statistics.csv")
    missing_values.to_csv(output_tables_dir / "missing_values_report.csv")
    dtypes.to_csv(output_tables_dir / "dtypes_report.csv")


def run_eda(
    csv_path: str = "data/raw/medical_insurance.csv",
    output_figures_dir: str = "outputs/figures/eda",
    output_tables_dir: str = "outputs/tables",
) -> None:

    output_figures_dir = Path(output_figures_dir)
    output_tables_dir = Path(output_tables_dir)

    output_figures_dir.mkdir(parents=True, exist_ok=True)
    output_tables_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(csv_path)

    print("Dataset loaded successfully.")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nSummary statistics:")
    print(df.describe(include="all"))

    save_numeric_distribution_plots(df, output_figures_dir)
    save_categorical_count_plots(df, output_figures_dir)
    save_correlation_heatmap(df, output_figures_dir)
    save_pairplot(df, output_figures_dir)
    save_eda_summary(df, output_tables_dir)

    print("\nEDA completed successfully.")
    print(f"Figures saved to: {output_figures_dir}")
    print(f"Tables saved to: {output_tables_dir}")


if __name__ == "__main__":
    run_eda()