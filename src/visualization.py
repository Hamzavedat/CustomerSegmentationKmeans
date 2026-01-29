"""
Visualization and Analysis Module
Dimensionality reduction with PCA, cluster visualization and characteristic analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def apply_pca(X, n_components=2):
    """
    Performs dimensionality reduction with PCA.
    
    3D RFM data → reduced to 2D
    
    Parameters:
    -----------
    X : np.ndarray
        Standardized feature matrix
    n_components : int
        Target number of dimensions
    
    Returns:
    --------
    np.ndarray, PCA
        Reduced data and PCA object
    """
    print(f"\n=== Dimensionality Reduction with PCA (3D → {n_components}D) ===")
    
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    
    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    return X_pca, pca


def visualize_clusters(X_pca, labels, k):
    """
    Visualizes clusters in PCA coordinates.
    
    Parameters:
    -----------
    X_pca : np.ndarray
        2D data reduced with PCA
    labels : np.ndarray
        Cluster labels
    k : int
        Number of clusters
    """
    print("\n=== Cluster Visualization ===")
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, k))
    
    for cluster_id in range(k):
        cluster_mask = labels == cluster_id
        plt.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1],
                   c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                   alpha=0.6, edgecolors='black', linewidth=0.5, s=50)
    
    plt.xlabel('PCA Component 1', fontsize=12)
    plt.ylabel('PCA Component 2', fontsize=12)
    plt.title(f'Customer Segmentation (k={k})', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/figures/cluster_visualization.png', dpi=300, bbox_inches='tight')
    print("Cluster visualization saved: outputs/figures/cluster_visualization.png")
    plt.close()


def analyze_cluster_characteristics(rfm_df, labels):
    """
    Analyzes RFM characteristics of each cluster.
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        Original RFM values (not log-transformed)
    labels : np.ndarray
        Cluster labels
    
    Returns:
    --------
    pd.DataFrame
        Cluster characteristics table
    """
    print("\n=== Cluster Characteristics Analysis ===")
    
    rfm_with_clusters = rfm_df.copy()
    rfm_with_clusters['Cluster'] = labels
    
    cluster_summary = rfm_with_clusters.groupby('Cluster').agg({
        'Recency': ['mean', 'median'],
        'Frequency': ['mean', 'median'],
        'Monetary': ['mean', 'median'],
        'Customer ID': 'count'
    }).round(2)
    
    cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
    cluster_summary.rename(columns={'Customer ID_count': 'Customer_Count'}, inplace=True)
    
    print("\nCluster Statistics:")
    print(cluster_summary)
    
    print("\n=== Cluster Interpretations ===")
    
    for cluster_id in range(len(cluster_summary)):
        recency_mean = cluster_summary.iloc[cluster_id]['Recency_mean']
        frequency_mean = cluster_summary.iloc[cluster_id]['Frequency_mean']
        monetary_mean = cluster_summary.iloc[cluster_id]['Monetary_mean']
        count = cluster_summary.iloc[cluster_id]['Customer_Count']
        
        print(f"\nCluster {cluster_id} ({count} customers):")
        print(f"  Average Recency: {recency_mean:.1f} days")
        print(f"  Average Frequency: {frequency_mean:.1f} purchases")
        print(f"  Average Monetary: ${monetary_mean:.2f}")
        
        if recency_mean < 50:
            recency_label = "Recently active"
        elif recency_mean < 100:
            recency_label = "Moderately active"
        else:
            recency_label = "Inactive for a long time"
        
        if frequency_mean > 10:
            frequency_label = "Frequent shopper"
        elif frequency_mean > 5:
            frequency_label = "Moderate shopper"
        else:
            frequency_label = "Infrequent shopper"
        
        if monetary_mean > 5000:
            monetary_label = "High spending"
        elif monetary_mean > 2000:
            monetary_label = "Medium spending"
        else:
            monetary_label = "Low spending"
        
        print(f"  → {recency_label}, {frequency_label}, {monetary_label}")
    
    return cluster_summary


def plot_cluster_characteristics(rfm_df, labels):
    """
    Visualizes RFM features of clusters with box plots.
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        Original RFM values
    labels : np.ndarray
        Cluster labels
    """
    rfm_with_clusters = rfm_df.copy()
    rfm_with_clusters['Cluster'] = labels
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('RFM Features of Clusters', fontsize=14, fontweight='bold')
    
    features = ['Recency', 'Frequency', 'Monetary']
    
    for i, feature in enumerate(features):
        sns.boxplot(x='Cluster', y=feature, data=rfm_with_clusters, ax=axes[i])
        axes[i].set_title(f'{feature} Distribution')
        axes[i].set_xlabel('Cluster')
        axes[i].set_ylabel(feature)
        axes[i].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('outputs/figures/cluster_characteristics.png', dpi=300, bbox_inches='tight')
    print("\nCluster characteristics plot saved: outputs/figures/cluster_characteristics.png")
    plt.close()


def generate_business_insights(cluster_summary):
    """
    Generates business insights and action recommendations.
    
    Parameters:
    -----------
    cluster_summary : pd.DataFrame
        Cluster statistics
    """
    print("\n" + "="*70)
    print("BUSINESS ANALYSIS AND ACTION RECOMMENDATIONS")
    print("="*70)
    
    print("\nThis segmentation study enables behavioral separation of")
    print("e-commerce customers. Recommendations for each segment:")
    
    n_clusters = len(cluster_summary)
    
    for cluster_id in range(n_clusters):
        recency = cluster_summary.iloc[cluster_id]['Recency_mean']
        frequency = cluster_summary.iloc[cluster_id]['Frequency_mean']
        monetary = cluster_summary.iloc[cluster_id]['Monetary_mean']
        
        print(f"\n{'─'*70}")
        print(f"CLUSTER {cluster_id}:")
        
        if recency < 50 and frequency > 8 and monetary > 3000:
            print("  TYPE: Champion Customers")
            print("  STRATEGY: VIP programs, early access, exclusive discounts")
        elif recency < 50 and frequency < 5:
            print("  TYPE: New Customers")
            print("  STRATEGY: Onboarding campaigns, loyalty program introduction")
        elif recency > 100:
            print("  TYPE: Lost Customers")
            print("  STRATEGY: Win-back campaigns, special offers")
        elif frequency > 8:
            print("  TYPE: Loyal Customers")
            print("  STRATEGY: Cross-sell, up-sell, referral programs")
        else:
            print("  TYPE: Potential Loyalists")
            print("  STRATEGY: Increase engagement, personalized recommendations")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    from data_loading import load_online_retail_data, clean_data
    from feature_engineering import calculate_rfm_features
    from preprocessing import apply_log_transformation, standardize_features
    from clustering import final_clustering
    
    df = load_online_retail_data()
    df_clean = clean_data(df)
    rfm_df = calculate_rfm_features(df_clean)
    rfm_transformed = apply_log_transformation(rfm_df)
    X_scaled, scaler = standardize_features(rfm_transformed)
    
    kmeans, labels = final_clustering(X_scaled, k=4)
    X_pca, pca = apply_pca(X_scaled)
    visualize_clusters(X_pca, labels, k=4)
    cluster_summary = analyze_cluster_characteristics(rfm_df, labels)
    plot_cluster_characteristics(rfm_df, labels)
    generate_business_insights(cluster_summary)
