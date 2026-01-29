# Customer Segmentation with k-Means Clustering 🛍️

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Customer Segmentation - Unsupervised Learning with k-Means**

A machine learning project that performs customer segmentation of e-commerce customers using RFM (Recency, Frequency, Monetary) analysis.

## 📊 About the Project

This project analyzes customer behavior using **ALL sheets** of the [Online Retail II](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) dataset (Year 2009-2010 and Year 2010-2011) and segments them using the k-Means clustering algorithm.

### Key Features

- ✅ Complete dataset utilization (all Excel sheets)
- ✅ RFM (Recency, Frequency, Monetary) feature engineering
- ✅ Data normalization with log transformation
- ✅ Optimal cluster number detection with Elbow method and Silhouette analysis
- ✅ Dimensionality reduction and visualization with PCA
- ✅ Detailed cluster characteristics analysis


## 📁 Project Structure

```
customer-segmentation-kmeans/
├── data/                           # Data directory
│   └── online_retail_II.xlsx      # (To be added by user)
├── src/                            # Source code modules
│   ├── data_loading.py            # Data loading and cleaning
│   ├── feature_engineering.py     # RFM feature extraction
│   ├── preprocessing.py           # Data preprocessing and transformations
│   ├── clustering.py              # k-Means algorithm implementation
│   └── visualization.py           # Visualization and analysis
├── outputs/                        # Output files
│   └── figures/                   # Generated plots
├── main.py                         # Main execution script
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```



## 🚀 Installation and Usage

### Requirements

- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository

```bash
git clone https://github.com/Hamzavedat/CustomerSegmentationKmeans.git
cd CustomerSegmentationKmeans
```

### 2. Download the Dataset

Download the dataset from [here](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) and place it in the `data/` folder:

```
data/online_retail_II.xlsx
```

> **⚠️ IMPORTANT:** This project automatically loads and combines **ALL sheets** from the Excel file (using `sheet_name=None` parameter).

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Project

```bash
python main.py
```



## 📈 Results

### Dataset Statistics
- **Total Transactions:** 1,067,371 records
- **After Cleaning:** 805,549 valid transactions
- **Unique Customers:** 5,878 customers
- **Date Range:** 2009-2010 and 2010-2011

### Clustering Results
- **Optimal Clusters (k):** 2 (Silhouette Score: 0.4399)
- **Total Explained Variance (PCA):** 95.14%

### Customer Segments

#### 🔴 Cluster 0: Lost Customers (61.38% - 3,608 customers)
- **Average Recency:** 295.4 days
- **Average Frequency:** 2.1 purchases
- **Average Monetary:** $636.11
- **Strategy:** Win-back campaigns, special offers, re-engagement programs

#### 🟢 Cluster 1: Champion Customers (38.62% - 2,270 customers)
- **Average Recency:** 49.2 days
- **Average Frequency:** 12.9 purchases
- **Average Monetary:** $6,805.44
- **Strategy:** VIP programs, early access, exclusive discounts, loyalty rewards

### Visualization Outputs

All plots are saved to the `outputs/figures/` folder:

| Plot | Description |
|------|-------------|
| `rfm_distributions.png` | Distribution of RFM features |
| `distribution_comparison.png` | Before/after log transformation comparison |
| `elbow_silhouette.png` | Optimal cluster number analysis (Elbow + Silhouette) |
| `cluster_visualization.png` | 2D cluster visualization with PCA |
| `cluster_characteristics.png` | Cluster characteristics comparison |

## 🔬 Methodology

1. **Data Loading**: Combining all Excel sheets
2. **Data Cleaning**: Handling missing values and outliers
3. **RFM Analysis**: 
   - Recency (Days since last purchase)
   - Frequency (Total number of orders)
   - Monetary (Total spending)
4. **Preprocessing**: Log transformation and standardization
5. **Clustering**: k-Means algorithm
6. **Optimization**: Elbow method and Silhouette analysis
7. **Visualization**: Dimensionality reduction with PCA

## 🛠️ Technologies Used

- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning (k-Means, PCA, StandardScaler)
- **matplotlib & seaborn**: Data visualization

## 📝 License

This project is for educational purposes.

---

⭐ If you found this project useful, don't forget to give it a star!

