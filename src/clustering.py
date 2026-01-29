"""
Clustering Module
Clustering with k-Means algorithm, optimal k selection (Elbow + Silhouette)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def elbow_method(X, k_range=range(2, 11)):
    """
    Determines the optimal number of clusters using the Elbow method.
    
    Calculates WCSS (Within-Cluster Sum of Squares) values.
    
    Parameters:
    -----------
    X : np.ndarray
        Standardized feature matrix
    k_range : range
        k values to try (default: 2-10)
    
    Returns:
    --------
    dict
        WCSS value for each k
    """
    print("\n=== Elbow Method ===")
    print("Trying k values:")
    
    wcss_values = {}
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss = kmeans.inertia_
        wcss_values[k] = wcss
        print(f"k={k:2d} → WCSS = {wcss:12.2f}")
    
    return wcss_values


def silhouette_analysis(X, k_range=range(2, 11)):
    """
    Evaluates cluster quality using Silhouette analysis.
    
    Silhouette Score:
    - Close to 1: Well-separated clusters
    - Close to 0: Overlapping clusters
    - Negative: Incorrect clustering
    
    Parameters:
    -----------
    X : np.ndarray
        Standardized feature matrix
    k_range : range
        k values to try
    
    Returns:
    --------
    dict
        Silhouette Score for each k
    """
    print("\n=== Silhouette Analysis ===")
    print("Trying k values:")
    
    silhouette_scores = {}
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores[k] = score
        print(f"k={k:2d} → Silhouette Score = {score:.4f}")
    
    return silhouette_scores


def plot_elbow_silhouette(wcss_values, silhouette_scores):
    """
    Plots Elbow and Silhouette graphs side by side.
    
    Parameters:
    -----------
    wcss_values : dict
        WCSS values for each k
    silhouette_scores : dict
        Silhouette scores for each k
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Determining Optimal Number of Clusters', fontsize=14, fontweight='bold')
    
    k_values = list(wcss_values.keys())
    wcss_vals = list(wcss_values.values())
    
    axes[0].plot(k_values, wcss_vals, marker='o', linestyle='-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[0].set_ylabel('WCSS', fontsize=11)
    axes[0].set_title('Elbow Method')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(k_values)
    
    silhouette_vals = list(silhouette_scores.values())
    
    axes[1].plot(k_values, silhouette_vals, marker='s', linestyle='-', 
                 linewidth=2, markersize=8, color='coral')
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[1].set_ylabel('Silhouette Score', fontsize=11)
    axes[1].set_title('Silhouette Analysis')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(k_values)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('outputs/figures/elbow_silhouette.png', dpi=300, bbox_inches='tight')
    print("\nElbow-Silhouette graph saved: outputs/figures/elbow_silhouette.png")
    plt.close()


def select_optimal_k(wcss_values, silhouette_scores):
    """
    Suggests optimal k by evaluating Elbow and Silhouette results.
    
    Parameters:
    -----------
    wcss_values : dict
        WCSS values
    silhouette_scores : dict
        Silhouette scores
    
    Returns:
    --------
    int
        Recommended optimal k value
    """
    print("\n=== Optimal k Selection ===")
    
    best_silhouette_k = max(silhouette_scores, key=silhouette_scores.get)
    best_silhouette_score = silhouette_scores[best_silhouette_k]
    
    print(f"Highest Silhouette Score: k={best_silhouette_k}, score={best_silhouette_score:.4f}")
    
    k_values = sorted(wcss_values.keys())
    wcss_diffs = []
    
    for i in range(1, len(k_values)):
        diff = wcss_values[k_values[i-1]] - wcss_values[k_values[i]]
        wcss_diffs.append(diff)
    
    max_diff_idx = wcss_diffs.index(max(wcss_diffs))
    elbow_k = k_values[max_diff_idx + 1]
    
    print(f"Elbow method suggestion: k={elbow_k}")
    
    if best_silhouette_score > 0.3:
        optimal_k = best_silhouette_k
        print(f"\n✓ Selected optimal k: {optimal_k} (Silhouette-based)")
    else:
        optimal_k = elbow_k
        print(f"\n✓ Selected optimal k: {optimal_k} (Elbow-based)")
    
    return optimal_k


def final_clustering(X, k):
    """
    Performs final clustering with the determined k value.
    
    Parameters:
    -----------
    X : np.ndarray
        Standardized feature matrix
    k : int
        Number of clusters
    
    Returns:
    --------
    KMeans, np.ndarray
        Trained model and cluster labels
    """
    print(f"\n=== Final Clustering (k={k}) ===")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    print(f"Clustering completed")
    print(f"Cluster labels: {np.unique(labels)}")
    print(f"Number of customers in each cluster:")
    
    for cluster_id in range(k):
        count = np.sum(labels == cluster_id)
        percentage = (count / len(labels)) * 100
        print(f"  Cluster {cluster_id}: {count:5d} customers ({percentage:5.2f}%)")
    
    return kmeans, labels


if __name__ == "__main__":
    from data_loading import load_online_retail_data, clean_data
    from feature_engineering import calculate_rfm_features
    from preprocessing import apply_log_transformation, standardize_features
    
    df = load_online_retail_data()
    df_clean = clean_data(df)
    rfm_df = calculate_rfm_features(df_clean)
    rfm_transformed = apply_log_transformation(rfm_df)
    X_scaled, scaler = standardize_features(rfm_transformed)
    
    wcss_values = elbow_method(X_scaled)
    silhouette_scores = silhouette_analysis(X_scaled)
    plot_elbow_silhouette(wcss_values, silhouette_scores)
    optimal_k = select_optimal_k(wcss_values, silhouette_scores)
    kmeans, labels = final_clustering(X_scaled, optimal_k)
