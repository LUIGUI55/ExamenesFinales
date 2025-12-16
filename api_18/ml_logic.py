
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

import os
import csv
from django.conf import settings
from api_17.ml_logic import CREDIT_CARD_DATA

def evaluate_model():
    """
    Evaluates DBSCAN heuristic against embedded CREDIT_CARD_DATA from api_17.
    Checks if Fraud(1) is detected as Noise(-1) and Normal(0) as Cluster(0).
    """
    y_true = []
    y_pred = []
    
    for row in CREDIT_CARD_DATA:
        v10 = row['V10']
        v14 = row['V14']
        true_cls = row['Class']
        
        # Predict (returns -1 for Outlier/Fraud, 0 for Normal)
        pred_cluster = predict_fraud_dbscan(v10, v14)
        
        # Map Cluster to Binary for Metrics
        # Cluster -1 (Noise) => Fraud (1)
        # Cluster 0 => Normal (0)
        pred_cls = 1 if pred_cluster == -1 else 0
        
        y_true.append(true_cls)
        y_pred.append(pred_cls)

    # Calculate Metrics Manualy
    tp = 0 
    tn = 0 
    fp = 0 
    fn = 0 
    
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
