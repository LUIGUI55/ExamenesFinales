
import math
import os
import csv
from django.conf import settings

# Lightweight implementation reading from a real CSV file
# This loads "datasets/creditcard_sample.csv" to determine centroids dynamically.

CENTROIDS = []

def load_data_and_train():
    """Reads CSV and calculates simple centroids for 2 clusters (Normal vs Fraud)."""
    global CENTROIDS
    
    csv_path = os.path.join(settings.BASE_DIR, 'datasets', 'creditcard_sample.csv')
    
    points_normal = []
    points_fraud = []
    
    if not os.path.exists(csv_path):
        # Fallback if file missing
        return [[0,0], [-10,-10]] 

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                v10 = float(row['V10'])
                v14 = float(row['V14'])
                cls = int(row['Class'])
                
                if cls == 0:
                    points_normal.append([v10, v14])
                else:
                    points_fraud.append([v10, v14])
            except ValueError:
                continue
                
    # Calculate means (Centroids)
    def get_mean(points):
        if not points: return [0,0]
        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points)
        return [sum_x/len(points), sum_y/len(points)]
    
    centroid_normal = get_mean(points_normal)
    centroid_fraud = get_mean(points_fraud)
    
    # We can fake "5 clusters" or just use these 2 logical ones for this dataset sample
    # Let's stick to 2 distinct behaviors for this robust sample
    CENTROIDS = [centroid_normal, centroid_fraud]

def predict_fraud(v10, v14):
    """
    Assigns a point to the nearest centroid (K-Means logic) trained on CSV.
    """
    if not CENTROIDS:
        load_data_and_train()
        
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
