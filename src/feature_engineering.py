"""
Feature Engineering Module
Calculates RFM (Recency, Frequency, Monetary) features
"""

import pandas as pd
import numpy as np


def calculate_rfm_features(df):
    """
    Calculates RFM features for each customer.
    
    RFM Explanation:
    - Recency: Number of days since last purchase
    - Frequency: Total number of unique invoices
    - Monetary: Total spending amount
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned transaction data
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing RFM features (Customer ID, Recency, Frequency, Monetary)
    """
    print("\n=== RFM Feature Calculation ===")
    
    df['TotalAmount'] = df['Quantity'] * df['Price']
    reference_date = df['InvoiceDate'].max()
    print(f"Reference date (max date): {reference_date}")
    
    rfm_df = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()
    
    rfm_df.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
    
    print(f"\nRFM features calculated for {len(rfm_df)} customers")
    print("\nRFM Statistics:")
    print(rfm_df[['Recency', 'Frequency', 'Monetary']].describe())
    
    return rfm_df


def analyze_rfm_distributions(rfm_df):
    """
    Analyzes the distributions of RFM features.
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        DataFrame containing RFM features
    """
    import matplotlib.pyplot as plt
    
    print("\n=== RFM Distribution Analysis ===")
    
    features = ['Recency', 'Frequency', 'Monetary']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Distribution of RFM Features', fontsize=14, fontweight='bold')
    
    for i, feature in enumerate(features):
        axes[i].hist(rfm_df[feature], bins=50, edgecolor='black', alpha=0.7)
        axes[i].set_xlabel(feature, fontsize=11)
        axes[i].set_ylabel('Frequency', fontsize=11)
        axes[i].set_title(f'{feature} Distribution')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/figures/rfm_distributions.png', dpi=300, bbox_inches='tight')
    print("Histogram saved: outputs/figures/rfm_distributions.png")
    plt.close()


if __name__ == "__main__":
    from data_loading import load_online_retail_data, clean_data
    
    df = load_online_retail_data()
    df_clean = clean_data(df)
    rfm_df = calculate_rfm_features(df_clean)
    
    print("\nFirst 10 customers:")
    print(rfm_df.head(10))
    
    analyze_rfm_distributions(rfm_df)
