from etl_eda import etl_eda
from feature_selection import select_features
from models import build_models

def train_pipeline(data_path):
    """Full pipeline: ETL/EDA, Feature Selection, Model Building."""
    X_pca, y, original_columns = etl_eda(data_path)
    X_selected, selected_features = select_features(X_pca, y, original_columns, k=5)
    models = build_models(X_selected, y)
    return models