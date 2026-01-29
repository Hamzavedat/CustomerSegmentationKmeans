"""
Main Execution File
Runs the entire pipeline sequentially
"""

import sys
import os

sys.path.append('src')

from data_loading import load_online_retail_data, clean_data
from feature_engineering import calculate_rfm_features, analyze_rfm_distributions
from preprocessing import (calculate_skewness, apply_log_transformation, 
                          compare_distributions_before_after, standardize_features)
from clustering import (elbow_method, silhouette_analysis, plot_elbow_silhouette,
                       select_optimal_k, final_clustering)
from visualization import (apply_pca, visualize_clusters, analyze_cluster_characteristics,
                          plot_cluster_characteristics, generate_business_insights)


def main():
    """
    Machine Learning Project - Main Pipeline
    Customer Segmentation with k-Means
    """
    print("="*80)
    print("MACHINE LEARNING PROJECT - CUSTOMER SEGMENTATION")
    print("Unsupervised Learning with k-Means")
    print("Student ID: 24501077")
    print("="*80)
    
    print("\n" + "─"*80)
    print("STEP 1: DATA LOADING AND CLEANING")
    print("─"*80)
    
    try:
        df = load_online_retail_data('data/online_retail_II.xlsx')
        df_clean = clean_data(df)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease download the dataset and continue.")
        return
    
    print("\n" + "─"*80)
    print("STEP 2: FEATURE ENGINEERING (RFM)")
    print("─"*80)
    
    rfm_df = calculate_rfm_features(df_clean)
    analyze_rfm_distributions(rfm_df)
    
    print("\n" + "─"*80)
    print("STEP 3: DISTRIBUTION ANALYSIS AND TRANSFORMATIONS")
    print("─"*80)
    
    calculate_skewness(rfm_df)
    rfm_transformed = apply_log_transformation(rfm_df)
    compare_distributions_before_after(rfm_df, rfm_transformed)
    X_scaled, scaler = standardize_features(rfm_transformed)
    
    print("\n" + "─"*80)
    print("STEP 4: DETERMINING OPTIMAL NUMBER OF CLUSTERS")
    print("─"*80)
    
    k_range = range(2, 11)
    wcss_values = elbow_method(X_scaled, k_range)
    silhouette_scores = silhouette_analysis(X_scaled, k_range)
    plot_elbow_silhouette(wcss_values, silhouette_scores)
    optimal_k = select_optimal_k(wcss_values, silhouette_scores)
    
    print("\n" + "─"*80)
    print("STEP 5: FINAL CLUSTERING")
    print("─"*80)
    
    kmeans, labels = final_clustering(X_scaled, optimal_k)
    
    print("\n" + "─"*80)
    print("STEP 6: VISUALIZATION AND ANALYSIS")
    print("─"*80)
    
    X_pca, pca = apply_pca(X_scaled, n_components=2)
    visualize_clusters(X_pca, labels, optimal_k)
    cluster_summary = analyze_cluster_characteristics(rfm_df, labels)
    plot_cluster_characteristics(rfm_df, labels)
    generate_business_insights(cluster_summary)
    

    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
