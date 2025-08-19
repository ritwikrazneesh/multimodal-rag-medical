import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def etl_eda(data_path):
    """ETL and EDA with PCA for feature reduction."""
    # ETL: Load and clean data
    df = pd.read_csv(data_path)  # Assume CSV with features and 'retention' label
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)

    # EDA: Basic stats
    print(df.describe())

    # Feature scaling
    X = df.drop('retention', axis=1)
    y = df['retention']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA: Reduce features by 70%
    n_components = max(1, int(X.shape[1] * 0.3))
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Explained variance: {pca.explained_variance_ratio_.sum()}")

    return X_pca, y, X.columns