
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Global model cache
_KMEANS_MODEL = None

def get_mock_data(n_samples=500):
    """Generates mock credit card data for demonstration."""
    # Columns: Time, V1..V28, Amount
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    
    # Random data resembling PCA components (centered around 0)
    data = np.random.randn(n_samples, 30)
    
    # Adjust Time and Amount to look somewhat realistic
    data[:, 0] = np.random.randint(0, 172800, n_samples) # 2 days in seconds
    data[:, -1] = np.random.uniform(0, 500, n_samples) # Amount
    
    df = pd.DataFrame(data, columns=columns)
    return df

def train_kmeans_model():
    """Trains a KMeans model on mock data."""
    global _KMEANS_MODEL
    
    # Load (mock) data
    df = get_mock_data(n_samples=1000)
    
    # Preprocessing: Drop Time and Amount as per notebook analysis
    X = df.drop(["Time", "Amount"], axis=1)
    
    # Use only V10 and V14 as per notebook example for visualization/simplicity?
    # Or use all? The notebook used V10 and V14 for 2D plot.
    # Let's use all for a more "robust" API, or stick to V10/V14.
    # Sticking to V10/V14 makes it easier to test manual inputs.
    # Let's stick to V10 and V14 as the "Notebook Strategy".
    X_subset = X[["V10", "V14"]]
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_subset)
    
    _KMEANS_MODEL = kmeans
    return _KMEANS_MODEL

def predict_fraud(v10, v14):
    """Predicts cluster for given V10 and V14 values."""
    global _KMEANS_MODEL
    if _KMEANS_MODEL is None:
        train_kmeans_model()
        
    # Reshape for prediction
    input_data = np.array([[v10, v14]])
    cluster = _KMEANS_MODEL.predict(input_data)[0]
    
    # In KMeans, "Fraud" isn't a direct output, but we return the cluster ID.
    # The notebook analyzed which cluster correlated with fraud.
    # For this API, we just return the cluster.
    return int(cluster)
