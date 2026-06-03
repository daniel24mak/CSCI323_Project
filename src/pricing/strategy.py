import pandas as pd

# Premium = predicted_expense * BASE_LOADING_RATE * risk_multiplier * (1 - discount)
BASE_LOADING_RATE = 0.10

RISK_MULTIPLIERS = {
    "LOW_RISK": 0.85,
    "MEDIUM_RISK": 1.00,
    "HIGH_RISK": 1.25,
}

DISCOUNT_RATE = 0.10


def compute_premium(
    predicted_expense: float,
    risk_tier: str,
    discount_eligible: bool = False,
) -> dict:
    """
    Compute a recommended insurance premium for one customer.

    Returns a breakdown dict with all intermediate values so results
    are auditable end-to-end.
    """
    if risk_tier not in RISK_MULTIPLIERS:
        raise ValueError(
            f"Unknown risk tier '{risk_tier}'. "
            f"Expected one of: {list(RISK_MULTIPLIERS)}"
        )

    multiplier = RISK_MULTIPLIERS[risk_tier]
    base_premium = predicted_expense * BASE_LOADING_RATE
    risk_adjusted = base_premium * multiplier
    discount = risk_adjusted * DISCOUNT_RATE if discount_eligible else 0.0
    final_premium = risk_adjusted - discount

    return {
        "predicted_expense": round(predicted_expense, 2),
        "risk_tier": risk_tier,
        "base_premium": round(base_premium, 2),
        "risk_multiplier": multiplier,
        "risk_adjusted_premium": round(risk_adjusted, 2),
        "discount_applied": round(discount, 2),
        "recommended_premium": round(final_premium, 2),
    }


def price_customers(
    predicted_expenses,
    risk_tiers,
    discount_flags,
) -> pd.DataFrame:
    """
    Compute recommended premiums for a batch of customers.

    Parameters
    ----------
    predicted_expenses : array-like of float
    risk_tiers         : array-like of str  ("LOW_RISK" | "MEDIUM_RISK" | "HIGH_RISK")
    discount_flags     : array-like of int/bool  (1 = eligible, 0 = not)

    Returns
    -------
    pd.DataFrame with one row per customer and full pricing breakdown.
    """
    rows = [
        compute_premium(float(exp), tier, bool(disc))
        for exp, tier, disc in zip(predicted_expenses, risk_tiers, discount_flags)
    ]
    return pd.DataFrame(rows)
