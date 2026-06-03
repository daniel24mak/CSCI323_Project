"""
M4 Demo — run the full pipeline and display a formatted summary.

Usage
-----
    python demo.py
    python demo.py --csv data/raw/medical_insurance.csv
"""

import argparse
import textwrap

import pandas as pd

from src.pipeline import run_full_pipeline

CSV_PATH = "data/raw/medical_insurance.csv"


def _divider(title: str = "", width: int = 70) -> str:
    if title:
        pad = (width - len(title) - 2) // 2
        return "=" * pad + f" {title} " + "=" * (width - pad - len(title) - 2)
    return "=" * width


def _print_regression_summary(regression_result: dict) -> None:
    print(_divider("Regression"))
    print(f"  Best model : {regression_result['best_model_name']}")
    vm = regression_result["val_metrics"]
    print(f"  Val R²     : {vm['r2']:.4f}")
    print(f"  Val RMSE   : ${vm['rmse']:,.2f}")
    print(f"  Val MAE    : ${vm['mae']:,.2f}")

    print("\n  Model comparison (validation set):")
    comparison = regression_result["comparison_table"]
    r2_cols = [c for c in comparison.columns if "R²" in c]
    rmse_cols = [c for c in comparison.columns if "RMSE" in c]
    print(
        comparison[["Model"] + rmse_cols + r2_cols].to_string(index=False)
    )


def _print_clustering_summary(clustering_result: dict) -> None:
    print(_divider("Clustering"))
    print(f"  Clusters (k)      : {clustering_result['k']}")
    print(f"  Silhouette score  : {clustering_result['silhouette_score']:.4f}")

    from collections import Counter
    tier_counts = Counter(clustering_result["risk_tier"])
    for tier in ["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"]:
        if tier in tier_counts:
            print(f"  {tier:<14} : {tier_counts[tier]} training customers")


def _print_pricing_summary(pricing_df: pd.DataFrame) -> None:
    print(_divider("Pricing Engine — Test Set"))
    print(f"  Customers priced  : {len(pricing_df)}")

    by_tier = pricing_df.groupby("risk_tier")["recommended_premium"]
    print("\n  Mean recommended premium by risk tier:")
    for tier, group in by_tier:
        print(f"    {tier:<14} : ${group.mean():>10,.2f}")

    discount_pct = (pricing_df["discount_applied"] > 0).mean() * 100
    print(f"\n  Customers receiving discount : {discount_pct:.1f}%")

    print("\n  Sample of 8 priced customers:")
    sample = pricing_df[[
        "predicted_expense",
        "risk_tier",
        "risk_multiplier",
        "discount_applied",
        "recommended_premium",
    ]].head(8)
    sample = sample.rename(columns={
        "predicted_expense": "pred_expense",
        "risk_multiplier": "mult",
        "discount_applied": "discount",
        "recommended_premium": "premium",
    })
    print(textwrap.indent(sample.to_string(index=False), "    "))


def main(csv_path: str = CSV_PATH) -> None:
    print(_divider())
    print("  CSCI323 — Medical Insurance Pricing Pipeline")
    print(_divider())
    print(f"  Data : {csv_path}\n")

    result = run_full_pipeline(csv_path, save_results=True)

    _print_clustering_summary(result["clustering"])
    print()
    _print_regression_summary(result["regression"])
    print()
    _print_pricing_summary(result["pricing_results"])
    print()
    print(_divider())
    print("  Results saved to outputs/demo_results/pricing_results.csv")
    print(_divider())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV_PATH)
    args = parser.parse_args()
    main(args.csv)
