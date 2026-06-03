"""
End-to-end pipeline: preprocess → cluster → regress → price.

Typical usage
-------------
    from src.pipeline import run_full_pipeline

    result = run_full_pipeline("data/raw/medical_insurance.csv")
    print(result["pricing_results"].head())
"""

import numpy as np
import pandas as pd

from src.data.preprocess import get_preprocessed_data
from src.models.clustering import run_clustering, get_cluster_model
from src.models.regression import train_regression, predict_expense
from src.pricing.strategy import price_customers


def _derive_risk_mapping(labels, risk_tiers) -> dict:
    """
    Build a {cluster_id: risk_tier} dict from the arrays returned by
    run_clustering.  Works regardless of how many clusters were chosen.
    """
    return dict(zip(labels, risk_tiers))


def _assign_risk_tiers(X: pd.DataFrame, cluster_model, risk_mapping: dict):
    """
    Assign risk tiers to a new set of customers using the fitted cluster model.
    """
    labels = cluster_model.predict(X[["age", "bmi"]])
    return [risk_mapping[label] for label in labels]


def run_full_pipeline(csv_path: str, save_results: bool = False) -> dict:
    """
    Run the complete M1-M4 pipeline on a raw CSV.

    Steps
    -----
    1. Preprocess  — clean, encode, scale; produce train / val / test splits.
    2. Cluster     — K-Means on (age, bmi) to assign LOW / MEDIUM / HIGH risk tiers.
    3. Regress     — compare Linear, Ridge, Lasso; keep best model by val R².
    4. Price       — apply pricing engine to every customer in the test split.

    Parameters
    ----------
    csv_path     : path to raw medical_insurance.csv
    save_results : if True, write pricing_results.csv to outputs/demo_results/

    Returns
    -------
    dict with keys:
        data             – preprocessed splits (X_train, X_val, X_test, y_*)
        clustering       – M2 contract dict
        regression       – M3 contract dict  (best_model, metrics, …)
        pricing_results  – pd.DataFrame, one row per test customer
    """
    # ── 1. Preprocess ────────────────────────────────────────────────────────
    data = get_preprocessed_data(csv_path)

    X_train = data["X_train"]
    X_val   = data["X_val"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_val   = data["y_val"]

    # ── 2. Cluster ────────────────────────────────────────────────────────────
    clustering_result = run_clustering(X_train)
    cluster_model = get_cluster_model()
    risk_mapping = _derive_risk_mapping(
        clustering_result["labels"],
        clustering_result["risk_tier"],
    )

    # ── 3. Regress ────────────────────────────────────────────────────────────
    regression_result = train_regression(X_train, y_train, X_val, y_val)
    best_model = regression_result["best_model"]

    # ── 4. Price test split ───────────────────────────────────────────────────
    predicted_expenses = np.array([
        predict_expense(best_model, X_test.iloc[[i]])
        for i in range(len(X_test))
    ])

    test_risk_tiers = _assign_risk_tiers(X_test, cluster_model, risk_mapping)
    discount_flags  = X_test["discount_eligibility"].values

    pricing_df = price_customers(predicted_expenses, test_risk_tiers, discount_flags)

    if save_results:
        from pathlib import Path
        out_dir = Path("outputs/demo_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        pricing_df.to_csv(out_dir / "pricing_results.csv", index=False)

    return {
        "data": data,
        "clustering": clustering_result,
        "regression": regression_result,
        "pricing_results": pricing_df,
    }
