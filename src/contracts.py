"""
Data contracts — documented shapes of each module's return dictionary.

These are reference definitions, not enforced at runtime.
Each module's public function is expected to return a dict matching
the TypedDict defined here so that the pipeline can rely on consistent keys.
"""

from typing import TypedDict, Any
import numpy as np
import pandas as pd


# ── M2 — Clustering ──────────────────────────────────────────────────────────

class ClusteringResult(TypedDict):
    labels: np.ndarray          # shape (n_train,), int cluster IDs
    risk_tier: list             # shape (n_train,), each "LOW_RISK" | "MEDIUM_RISK" | "HIGH_RISK"
    k: int                      # number of clusters chosen
    inertia: float
    silhouette_score: float
    cluster_centers: np.ndarray # shape (k, 2) — [age, bmi] centroids


# ── M3 — Regression ──────────────────────────────────────────────────────────

class RegressionMetrics(TypedDict):
    rmse: float
    mae: float
    r2: float


class RegressionResult(TypedDict):
    best_model: Any             # fitted sklearn estimator
    best_model_name: str        # "Linear Regression" | "Ridge" | "Lasso"
    train_metrics: RegressionMetrics
    val_metrics: RegressionMetrics
    predictions: np.ndarray     # val-set predictions from best model
    comparison_table: pd.DataFrame


# ── M4 — Pricing ─────────────────────────────────────────────────────────────

class PremiumBreakdown(TypedDict):
    predicted_expense: float
    risk_tier: str              # "LOW_RISK" | "MEDIUM_RISK" | "HIGH_RISK"
    base_premium: float         # predicted_expense * BASE_LOADING_RATE
    risk_multiplier: float      # tier-specific multiplier (0.85 / 1.00 / 1.25)
    risk_adjusted_premium: float
    discount_applied: float     # 0.0 if not eligible
    recommended_premium: float  # final output


# ── M4 — Pipeline ─────────────────────────────────────────────────────────────

class PipelineResult(TypedDict):
    data: dict                  # preprocessed splits (X_train, X_val, X_test, y_*)
    clustering: ClusteringResult
    regression: RegressionResult
    pricing_results: pd.DataFrame  # one PremiumBreakdown row per test customer
