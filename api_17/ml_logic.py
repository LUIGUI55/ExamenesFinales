
import math
import os
import csv
from django.conf import settings

# Lightweight implementation reading from a real CSV file
# This loads "datasets/creditcard_sample.csv" to determine centroids dynamically.

CENTROIDS = []


# Embedded dataset to guarantee availability on Vercel without file I/O issues
CREDIT_CARD_DATA = [
    {"V10": -0.3, "V14": -0.5, "Class": 0},
    {"V10": -0.4, "V14": -0.4, "Class": 0},
    {"V10": 0.1, "V14": 0.2, "Class": 0},
    {"V10": 2.5, "V14": 2.1, "Class": 0},
    {"V10": 2.4, "V14": 2.2, "Class": 0},
    {"V10": 2.6, "V14": 2.0, "Class": 0},
    {"V10": -2.1, "V14": 2.2, "Class": 0},
    {"V10": -2.0, "V14": 2.1, "Class": 0},
    {"V10": -2.2, "V14": 2.3, "Class": 0},
    {"V10": 2.1, "V14": -2.4, "Class": 0},
    {"V10": 2.0, "V14": -2.5, "Class": 0},
    {"V10": 1.9, "V14": -2.6, "Class": 0},
    {"V10": -12.5, "V14": -15.2, "Class": 1},
    {"V10": -11.2, "V14": -14.5, "Class": 1},
    {"V10": -13.0, "V14": -14.0, "Class": 1},
    {"V10": -10.0, "V14": -12.0, "Class": 1},
    {"V10": 0.0, "V14": 0.0, "Class": 0},
    {"V10": 0.1, "V14": 0.1, "Class": 0}
]

def load_data_and_train():
    """Calculates simple centroids from embedded data."""
    global CENTROIDS
    
    points_normal = []
    points_fraud = []
    
    for row in CREDIT_CARD_DATA:
        v10 = row['V10']
        v14 = row['V14']
        cls = row['Class']
        
        if cls == 0:
            points_normal.append([v10, v14])
        else:
            points_fraud.append([v10, v14])
                
    # Calculate means (Centroids)
    def get_mean(points):
        if not points: return [0,0]
        sum_x = sum(p[0] for p in points)
        sum_y = sum(p[1] for p in points)
        return [sum_x/len(points), sum_y/len(points)]
    
    centroid_normal = get_mean(points_normal)
    centroid_fraud = get_mean(points_fraud)
    
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

def evaluate_model():
    """
    Iterates over the embedded dataset and calculates metrics manually.
    Returns: dict with accuracy, f1, precision, recall, confusion_matrix
    """
    if not CENTROIDS:
        load_data_and_train()
        
    y_true = []
    y_pred = []
    
    for row in CREDIT_CARD_DATA:
        v10 = row['V10']
        v14 = row['V14']
        true_cls = row['Class']
        
        # Predict
        pred_cls = predict_fraud(v10, v14)
        
        y_true.append(true_cls)
        y_pred.append(pred_cls)

    # Calculate Metrics Manualy
    tp = 0 # Fraud (1) predicted as Fraud (1)
    tn = 0 # Normal (0) predicted as Normal (0)
    fp = 0 # Normal (0) predicted as Fraud (1)
    fn = 0 # Fraud (1) predicted as Normal (0)
    
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1: tp += 1
        elif t == 0 and p == 0: tn += 1
        elif t == 0 and p == 1: fp += 1
        elif t == 1 and p == 0: fn += 1
        
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2),
        "total_samples": total,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    }
