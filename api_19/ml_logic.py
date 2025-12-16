
import string
import math
from collections import defaultdict

# Lightweight implementation without heavy libraries like sklearn/pandas
# to fit within Vercel's 250MB limit.

class SimpleNaiveBayes:
    def __init__(self):
        self.spam_counts = defaultdict(int)
        self.ham_counts = defaultdict(int)
        self.spam_total = 0
        self.ham_total = 0
        self.vocab = set()

    def train(self, data):
        for text, label in data:
            words = self._tokenize(text)
            for word in words:
                self.vocab.add(word)
                if label == 1:
                    self.spam_counts[word] += 1
                    self.spam_total += 1
                else:
                    self.ham_counts[word] += 1
                    self.ham_total += 1

    def predict(self, text):
        words = self._tokenize(text)
        
        # P(Spam) and P(Ham) priors (assume equal for now or based on counts)
        log_prob_spam = 0.0
        log_prob_ham = 0.0
        
        # Additive smoothing (Laplace)
        vocab_size = len(self.vocab)
        
        for word in words:
            # P(Word | Spam)
            p_w_spam = (self.spam_counts[word] + 1) / (self.spam_total + vocab_size)
            log_prob_spam += math.log(p_w_spam)
            
            # P(Word | Ham)
            p_w_ham = (self.ham_counts[word] + 1) / (self.ham_total + vocab_size)
            log_prob_ham += math.log(p_w_ham)
        
        # Compare
        if log_prob_spam > log_prob_ham:
            return 1, 1.0 # High probability
        else:
            return 0, 0.0 # Low probability

    def _tokenize(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        return text.split()

import os
import csv
from django.conf import settings

# Lightweight implementation that trains on "datasets/spam_sample.csv"

# [SimpleNaiveBayes Class Definition remains same, just replacing data loading part]
# ... (Redefining class for completeness in replacement if needed, 
# but effectively we just need to change how data is loaded)

# Let's rewrite the file content efficiently or just the bottom part if possible.
# Since the class is large, I'll rewrite the TRAIN_DATA part.


# Embedded dataset for Spam Detection
# (1 = Spam, 0 = Ham)
SPAM_DATA = [
    ("Win a free iPhone now! Click here to claim your prize.", 1),
    ("Congratulations! You've been selected for a cash award.", 1),
    ("Viagra for sale, cheap prices, fast delivery.", 1),
    ("Urgent: Your account security is at risk. Verify now.", 1),
    ("Hot singles in your area waiting to chat.", 1),
    ("Meeting reminded for tomorrow at 10 AM regarding the project.", 0),
    ("Hey, can you send me the report when you're done?", 0),
    ("Lunch plans? Let's meet at the usual place.", 0),
    ("Attached is the invoice for last month's services.", 0),
    ("Happy Birthday! Hope you have a great day.", 0),
    ("Please review the attached document.", 0),
    ("What time are we meeting later?", 0),
    ("Exclusive offer: 50% off all products today only!", 1)
]

TRAIN_DATA = []

def load_spam_data():
    global TRAIN_DATA
    # Just load from embedded list
    for text, label in SPAM_DATA:
        TRAIN_DATA.append((text, label))

_NB_MODEL = None

def get_model():
    global _NB_MODEL
    if not TRAIN_DATA:
        load_spam_data()
        
    if _NB_MODEL is None:
        _NB_MODEL = model
    return _NB_MODEL

def evaluate_model():
    """
    Evaluates Naive Bayes model against embedded SPAM_DATA.
    """
    model = get_model() # Ensure trained
    
    y_true = []
    y_pred = []
    
    # Validation loop
    for text, label in SPAM_DATA:
        pred, prob = model.predict(text)
        y_true.append(label)
        y_pred.append(pred)
        
    # Calculate Metrics Manualy
    tp = 0 # Spam (1) as Spam
    tn = 0 # Ham (0) as Ham
    fp = 0 # Ham (0) as Spam
    fn = 0 # Spam (1) as Ham
    
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

def predict_spam(text):
    model = get_model()
    prediction, prob = model.predict(text)
    return prediction, prob
