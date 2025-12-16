
import random
import math

# Lightweight implementation without heavy libraries like sklearn/pandas
# to fit within Vercel's 250MB limit.

# Pre-defined centroids for our "Mock" 5 clusters (2D: V10, V14)
# These represent typical centers found in the analysis.
CENTROIDS = [
    [-0.5, -0.5], # Cluster 0
    [2.0, 2.0],   # Cluster 1
    [-2.0, 2.0],  # Cluster 2
    [2.0, -2.0],  # Cluster 3
    [0.0, 5.0]    # Cluster 4 (Outlier-ish)
]

def predict_fraud(v10, v14):
    """
    Assigns a point to the nearest centroid (K-Means logic).
    """
    point = (v10, v14)
    min_dist = float('inf')
    closest_cluster = -1
    
    for i, center in enumerate(CENTROIDS):
        # Euclidean distance
        dist = math.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
        if dist < min_dist:
            min_dist = dist
            closest_cluster = i
            
    return closest_cluster

# Placeholder compatibility functions if needed
def train_kmeans_model():
    pass
