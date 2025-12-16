
import math

# Lightweight implementation without heavy libraries like sklearn/pandas
# to fit within Vercel's 250MB limit.

def predict_fraud_dbscan(v10, v14):
    """
    DBSCAN-like logic using a simple distance heuristic.
    In fraud detection, 'Outliers' are often far from the 'Normal' cluster center.
    We assume the vast majority of points (normal) cluster around (0,0) or specific regions.
    If a point is too far -> Noise (Cluster -1).
    """
    
    # Heuristic: Calculate distance from origin (or approximate center of mass)
    # The mock data was centered around 0 with std dev 1 (randn).
    # DBSCAN usually groups dense areas. 
    # Lets say typically points are within distance 3 of origin.
    
    dist = math.sqrt(v10**2 + v14**2)
    
    # Threshold for "Noise"
    # If distance > 3 (3 sigma), consider it an outlier/noise
    if dist > 3.0:
        return -1 # Outlier / Fraud
    else:
        return 0 # Normal Cluster
