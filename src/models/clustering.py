import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


_cluster_model = None

RISK_LABELS = ["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"]


def _validate_input(X):
    """
    Validate input data and keep only age and bmi.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")

    required_columns = ["age", "bmi"]

    for col in required_columns:
        if col not in X.columns:
            raise ValueError(f"Missing required column: {col}")

    X_cluster = X[required_columns].copy()

    if X_cluster.isnull().sum().sum() > 0:
        raise ValueError("Input contains missing values.")

    return X_cluster


def _select_best_k(X_cluster, k_min=2, k_max=8):
    """
    Test k values from 2 to 8 and select best k using silhouette score.
    """
    inertia_values = {}
    silhouette_values = {}

    max_k = min(k_max, len(X_cluster) - 1)

    for k in range(k_min, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_cluster)

        inertia_values[k] = model.inertia_
        silhouette_values[k] = silhouette_score(X_cluster, labels)

    best_k = max(silhouette_values, key=silhouette_values.get)

    return best_k, inertia_values, silhouette_values


def _map_clusters_to_risk_tiers(X_cluster, labels):
    """
    Map K-Means cluster numbers into LOW_RISK, MEDIUM_RISK, HIGH_RISK.

    Lower average age + BMI = lower risk.
    Higher average age + BMI = higher risk.
    """
    temp_df = X_cluster.copy()
    temp_df["cluster"] = labels

    cluster_profiles = temp_df.groupby("cluster")[["age", "bmi"]].mean()
    cluster_profiles["risk_score"] = cluster_profiles["age"] + cluster_profiles["bmi"]

    sorted_clusters = cluster_profiles.sort_values("risk_score").index.tolist()

    risk_mapping = {}

    if len(sorted_clusters) == 2:
        risk_mapping[sorted_clusters[0]] = "LOW_RISK"
        risk_mapping[sorted_clusters[1]] = "HIGH_RISK"

    elif len(sorted_clusters) == 3:
        risk_mapping[sorted_clusters[0]] = "LOW_RISK"
        risk_mapping[sorted_clusters[1]] = "MEDIUM_RISK"
        risk_mapping[sorted_clusters[2]] = "HIGH_RISK"

    else:
        for i, cluster_id in enumerate(sorted_clusters):
            if i < len(sorted_clusters) / 3:
                risk_mapping[cluster_id] = "LOW_RISK"
            elif i < 2 * len(sorted_clusters) / 3:
                risk_mapping[cluster_id] = "MEDIUM_RISK"
            else:
                risk_mapping[cluster_id] = "HIGH_RISK"

    risk_tier = [risk_mapping[label] for label in labels]

    return risk_tier, risk_mapping


def run_clustering(X, k=None):
    """
    Run K-Means clustering using age and bmi only.

    Returns the required M2 contract dictionary.
    """
    global _cluster_model

    X_cluster = _validate_input(X)

    if k is None:
        k, _, _ = _select_best_k(X_cluster)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_cluster)

    silhouette = silhouette_score(X_cluster, labels)
    risk_tier, risk_mapping = _map_clusters_to_risk_tiers(X_cluster, labels)

    _cluster_model = model

    return {
        "labels": labels,
        "risk_tier": risk_tier,
        "k": k,
        "inertia": float(model.inertia_),
        "silhouette_score": float(silhouette),
        "cluster_centers": model.cluster_centers_,
    }


def get_cluster_model():
    """
    Return the fitted KMeans model.
    """
    if _cluster_model is None:
        raise ValueError("No clustering model has been fitted yet. Run run_clustering() first.")

    return _cluster_model


def plot_elbow_and_silhouette(X, save_path=None):
    """
    Create and save:
    1. Elbow curve
    2. Silhouette score curve
    """
    X_cluster = _validate_input(X)
    _, inertia_values, silhouette_values = _select_best_k(X_cluster)

    k_values = list(inertia_values.keys())

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, list(inertia_values.values()), marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Curve for K-Means Clustering")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, list(silhouette_values.values()), marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Scores for K-Means Clustering")

    if save_path:
        silhouette_path = save_path.replace(".png", "_silhouette.png")
        plt.savefig(silhouette_path, bbox_inches="tight")

    plt.show()

    return inertia_values, silhouette_values


def plot_clusters(X, clustering_result, save_path=None):
    """
    Create and save age vs BMI cluster scatter plot.
    """
    X_cluster = _validate_input(X)

    plot_df = X_cluster.copy()
    plot_df["risk_tier"] = clustering_result["risk_tier"]

    plt.figure(figsize=(8, 5))

    for tier in sorted(plot_df["risk_tier"].unique()):
        subset = plot_df[plot_df["risk_tier"] == tier]
        plt.scatter(subset["age"], subset["bmi"], label=tier, alpha=0.7)

    plt.xlabel("Age")
    plt.ylabel("BMI")
    plt.title("Customer Risk Clusters Based on Age and BMI")
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def create_cluster_profile(X, clustering_result, expenses=None):
    """
    Create cluster profile table with mean age, BMI, and optional expenses.
    """
    X_cluster = _validate_input(X)

    profile_df = X_cluster.copy()
    profile_df["cluster"] = clustering_result["labels"]
    profile_df["risk_tier"] = clustering_result["risk_tier"]

    if expenses is not None:
        profile_df["expenses"] = expenses

    group_cols = {
        "age": "mean",
        "bmi": "mean",
        "risk_tier": lambda x: x.mode()[0],
    }

    if expenses is not None:
        group_cols["expenses"] = "mean"

    profile = profile_df.groupby("cluster").agg(group_cols).reset_index()

    return profile

