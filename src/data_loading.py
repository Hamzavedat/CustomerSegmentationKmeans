"""
Data Loading Module
Loads the Online Retail II dataset and performs initial cleaning
"""

import pandas as pd
import os


def load_online_retail_data(filepath='data/online_retail_II.xlsx'):
    """
    Loads the Online Retail II dataset.
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            "Please download the Online Retail II dataset from UCI ML Repository:\n"
            "https://archive.ics.uci.edu/ml/datasets/Online+Retail+II\n"
            "And save it to the data/ folder as 'online_retail_II.xlsx'."
        )
    
    print(f"Loading data: {filepath}")
    
    all_sheets = pd.read_excel(filepath, sheet_name=None)
    
    print(f"Found sheets: {list(all_sheets.keys())}")
    
    dfs = []
    for sheet_name, df in all_sheets.items():
        print(f"  - {sheet_name}: {df.shape}")
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    
    print(f"\nCombined data size: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df


def clean_data(df):
    """
    Performs basic cleaning operations on the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataset
    """
    print("\n=== Data Cleaning ===")
    print(f"Initial row count: {len(df)}")
    
    df_clean = df[df['Customer ID'].notna()].copy()
    print(f"Removed missing Customer IDs: {len(df_clean)} rows remaining")
    
    df_clean = df_clean[df_clean['Quantity'] > 0]
    print(f"Removed negative/zero Quantity: {len(df_clean)} rows remaining")
    
    df_clean = df_clean[df_clean['Price'] > 0]
    print(f"Removed negative/zero Price: {len(df_clean)} rows remaining")
    
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    df_clean['Customer ID'] = df_clean['Customer ID'].astype(int)
    
    print(f"\nCleaned data size: {df_clean.shape}")
    print(f"Unique customers: {df_clean['Customer ID'].nunique()}")
    
    return df_clean


if __name__ == "__main__":
    df = load_online_retail_data()
    df_clean = clean_data(df)
    print("\nFirst 5 rows:")
    print(df_clean.head())
