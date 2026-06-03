import numpy as np
import pandas as pd
import pytest

from src.pricing.strategy import compute_premium, price_customers
from src.pipeline import run_full_pipeline, _derive_risk_mapping, _assign_risk_tiers


# ── Unit tests: pricing strategy ────────────────────────────────────────────

def test_compute_premium_low_risk_no_discount():
    result = compute_premium(10_000.0, "LOW_RISK", discount_eligible=False)
    # base = 1000, risk_adj = 1000 * 0.85 = 850, no discount
    assert result["base_premium"] == 1000.0
    assert result["risk_multiplier"] == 0.85
    assert result["discount_applied"] == 0.0
    assert result["recommended_premium"] == pytest.approx(850.0)


def test_compute_premium_high_risk_with_discount():
    result = compute_premium(10_000.0, "HIGH_RISK", discount_eligible=True)
    # base = 1000, risk_adj = 1000 * 1.25 = 1250, discount = 125
    assert result["recommended_premium"] == pytest.approx(1125.0)
    assert result["discount_applied"] == pytest.approx(125.0)


def test_compute_premium_unknown_tier_raises():
    with pytest.raises(ValueError, match="Unknown risk tier"):
        compute_premium(5000.0, "EXTREME_RISK")


def test_compute_premium_zero_expense():
    result = compute_premium(0.0, "MEDIUM_RISK")
    assert result["recommended_premium"] == 0.0


def test_price_customers_returns_dataframe():
    expenses = [5000.0, 12000.0, 20000.0]
    tiers = ["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"]
    discounts = [1, 0, 1]

    df = price_customers(expenses, tiers, discounts)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "recommended_premium" in df.columns
    assert "risk_tier" in df.columns
    assert "discount_applied" in df.columns


def test_price_customers_all_premiums_positive():
    expenses = [1000.0, 5000.0, 30000.0]
    tiers = ["LOW_RISK", "LOW_RISK", "HIGH_RISK"]
    discounts = [0, 0, 0]

    df = price_customers(expenses, tiers, discounts)

    assert (df["recommended_premium"] > 0).all()


# ── Unit tests: pipeline helpers ─────────────────────────────────────────────

def test_derive_risk_mapping():
    labels = np.array([0, 1, 2, 0, 1])
    tiers = ["LOW_RISK", "HIGH_RISK", "MEDIUM_RISK", "LOW_RISK", "HIGH_RISK"]
    mapping = _derive_risk_mapping(labels, tiers)

    assert mapping[0] == "LOW_RISK"
    assert mapping[1] == "HIGH_RISK"
    assert mapping[2] == "MEDIUM_RISK"


def test_assign_risk_tiers_shape():
    """_assign_risk_tiers must return one tier per row."""
    from sklearn.cluster import KMeans

    X = pd.DataFrame({
        "age": [25, 40, 55, 30, 60],
        "bmi": [22, 28, 35, 24, 32],
    })
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    model.fit(X[["age", "bmi"]])

    labels = model.predict(X[["age", "bmi"]])
    risk_mapping = {0: "LOW_RISK", 1: "MEDIUM_RISK", 2: "HIGH_RISK"}

    tiers = _assign_risk_tiers(X, model, risk_mapping)

    assert len(tiers) == len(X)
    assert all(t in {"LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"} for t in tiers)


# ── Integration test: full pipeline contract ─────────────────────────────────

def test_full_pipeline_contract(tmp_path):
    result = run_full_pipeline("data/raw/medical_insurance.csv", save_results=False)

    # Top-level keys
    assert {"data", "clustering", "regression", "pricing_results"} == set(result.keys())

    # Clustering contract (M2)
    clustering = result["clustering"]
    for key in ("labels", "risk_tier", "k", "inertia", "silhouette_score"):
        assert key in clustering

    # Regression contract (M3)
    regression = result["regression"]
    for key in ("best_model", "best_model_name", "train_metrics", "val_metrics"):
        assert key in regression

    assert regression["val_metrics"]["r2"] > 0.5, "Model R² should exceed 0.5 on this dataset"

    # Pricing results
    pricing = result["pricing_results"]
    assert isinstance(pricing, pd.DataFrame)
    assert len(pricing) > 0

    required_cols = {
        "predicted_expense",
        "risk_tier",
        "base_premium",
        "risk_multiplier",
        "risk_adjusted_premium",
        "discount_applied",
        "recommended_premium",
    }
    assert required_cols.issubset(set(pricing.columns))

    assert (pricing["recommended_premium"] >= 0).all()
    assert pricing["risk_tier"].isin(["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"]).all()
