"""
Preprocessing and Transformation Module
Performs skewness analysis, logarithmic transformation, and standardization operations
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def calculate_skewness(rfm_df):
    """
    Calculates the Fisher-Pearson skewness coefficient (g1) for each RFM feature.
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        DataFrame containing RFM features
    
    Returns:
    --------
    dict
        Skewness value for each feature
    """
    print("\n=== Skewness Analysis ===")
    
    features = ['Recency', 'Frequency', 'Monetary']
    skewness_values = {}
    
    for feature in features:
        skew_val = stats.skew(rfm_df[feature])
        skewness_values[feature] = skew_val
        
        if abs(skew_val) < 0.5:
            interpretation = "Approximately symmetric"
        elif skew_val > 0:
            interpretation = "Right-skewed (positive)"
        else:
            interpretation = "Left-skewed (negative)"
        
        print(f"{feature:12s}: g₁ = {skew_val:7.3f}  ({interpretation})")
    
    return skewness_values


def apply_log_transformation(rfm_df):
    """
    Applies log(1+x) transformation to RFM features.
    
    PDF condition: Transformation is applied if at least two features have g₁ values different from 0.
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        Original RFM features
    
    Returns:
    --------
    pd.DataFrame
        Log-transformed RFM features
    """
    print("\n=== Logarithmic Transformation ===")
    
    skewness_values = calculate_skewness(rfm_df)
    non_zero_skew = sum([1 for val in skewness_values.values() if abs(val) > 0.01])
    
    print(f"\nNumber of non-zero skewness values: {non_zero_skew}")
    
    if non_zero_skew >= 2:
        print("✓ Condition met: Applying log transformation...")
        
        rfm_transformed = rfm_df.copy()
        features = ['Recency', 'Frequency', 'Monetary']
        
        for feature in features:
            rfm_transformed[feature] = np.log1p(rfm_df[feature])
        
        print("\nSkewness values after transformation:")
        skewness_after = calculate_skewness(rfm_transformed)
        
        print("\nBefore → After comparison:")
        for feature in features:
            before = skewness_values[feature]
            after = skewness_after[feature]
            improvement = abs(after) < abs(before)
            status = "✓ Improved" if improvement else "✗ No change"
            print(f"{feature:12s}: {before:7.3f} → {after:7.3f}  {status}")
        
        return rfm_transformed
    else:
        print("✗ Condition not met: Log transformation not applied")
        return rfm_df.copy()


def compare_distributions_before_after(rfm_original, rfm_transformed):
    """
    Compares distributions before and after transformation.
    
    Parameters:
    -----------
    rfm_original : pd.DataFrame
        Original RFM values
    rfm_transformed : pd.DataFrame
        Log-transformed RFM values
    """
    features = ['Recency', 'Frequency', 'Monetary']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('RFM Transformation Comparison', fontsize=14, fontweight='bold')
    
    for i, feature in enumerate(features):
        axes[0, i].hist(rfm_original[feature], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, i].set_title(f'{feature} - Original')
        axes[0, i].set_xlabel(feature)
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].grid(True, alpha=0.3)
        
        axes[1, i].hist(rfm_transformed[feature], bins=50, edgecolor='black', alpha=0.7, color='coral')
        axes[1, i].set_title(f'{feature} - Log(1+x)')
        axes[1, i].set_xlabel(f'Log({feature})')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/figures/distribution_comparison.png', dpi=300, bbox_inches='tight')
    print("\nDistribution comparison saved: outputs/figures/distribution_comparison.png")
    plt.close()


def standardize_features(rfm_df):
    """
    Standardizes RFM features using Z-score normalization.
    
    Z = (X - μ) / σ
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        Log-transformed RFM features
    
    Returns:
    --------
    np.ndarray, StandardScaler
        Standardized features and scaler object
    """
    print("\n=== Standardization (Z-Score Normalization) ===")
    
    features = ['Recency', 'Frequency', 'Monetary']
    X = rfm_df[features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Features standardized")
    print(f"Previous mean: {X.mean(axis=0)}")
    print(f"New mean: {X_scaled.mean(axis=0)}")
    print(f"Previous std: {X.std(axis=0)}")
    print(f"New std: {X_scaled.std(axis=0)}")
    
    return X_scaled, scaler


if __name__ == "__main__":
    from data_loading import load_online_retail_data, clean_data
    from feature_engineering import calculate_rfm_features
    
    df = load_online_retail_data()
    df_clean = clean_data(df)
    rfm_df = calculate_rfm_features(df_clean)
    
    skewness = calculate_skewness(rfm_df)
    rfm_transformed = apply_log_transformation(rfm_df)
    compare_distributions_before_after(rfm_df, rfm_transformed)
    X_scaled, scaler = standardize_features(rfm_transformed)
    
    print("\nProcess completed!")
