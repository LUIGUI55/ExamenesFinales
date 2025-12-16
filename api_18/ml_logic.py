
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Global model cache (DBSCAN isn't really "predictive" on new single samples easily without KNN, 
# but we can simulate "fitting" or check neighbor distance. 
# For this demo, we'll retrain or use a trick: 
# DBSCAN clusters existing points. To classify a NEW point, usually you see which core point it's close to.
# Or we just run DBSCAN on Data + NewPoint.
# For simplicity/performance in this demo context, we'll run it on the batch including the new point, or just return random result if slow? 
# Running on 500 points is fast. Let's append and re-run.)

def get_mock_data(n_samples=500):
    """Generates mock credit card data for demonstration."""
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    data = np.random.randn(n_samples, 30)
    data[:, 0] = np.random.randint(0, 172800, n_samples)
    data[:, -1] = np.random.uniform(0, 500, n_samples)
    df = pd.DataFrame(data, columns=columns)
    return df

def predict_fraud_dbscan(v10, v14):
    """
    DBSCAN doesn't have a simple .predict() method for new points.
    We'll create a small dataset, add our point, and see where it lands.
    """
    # 1. Generate background mock data
    df = get_mock_data(n_samples=200) # Small batch for speed
    X = df[["V10", "V14"]].values
    
    # 2. Add our new point
    new_point = np.array([[v10, v14]])
    X_combined = np.vstack([X, new_point])
    
    # 3. Scale data (DBSCAN is sensitive to scale)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # 4. Run DBSCAN
    # eps=0.5, min_samples=5 (standard defaults or tweaked)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(X_scaled)
    
    # 5. Get cluster of the last point (our input)
    my_cluster = clusters[-1]
    
    return int(my_cluster)
