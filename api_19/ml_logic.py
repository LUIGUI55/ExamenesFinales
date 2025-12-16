
import numpy as np
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Ensure NLTK data (if possible, or skip stemming for simple demo)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    # nltk.download('punkt', quiet=True) 
    # nltk.download('stopwords', quiet=True)
    pass 
    # We might skip complex stemming if internet/nltk is broken in this env
    # For a robust demo without external deps, we'll do simple split.

class SimpleParser:
    """ simplified parser replacing the complex one from notebook for demo purposes """
    def parse(self, text):
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.lower()

# Mock training data (since trec07p is missing)
TRAIN_DATA = [
    ("Win money now free cash prize", 1), # Spam
    ("Buy cheap meds viagra pills", 1),
    ("Click here to claim your reward", 1),
    ("Meeting tomorrow at 10am", 0), # Ham
    ("Hello friend how are you", 0),
    ("Project report attached", 0),
]

_NB_MODEL = None
_VECTORIZER = None

def train_nb_model():
    global _NB_MODEL, _VECTORIZER
    
    # Prepare data
    corpus = [text for text, label in TRAIN_DATA]
    labels = [label for text, label in TRAIN_DATA]
    
    # Vectorize
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)
    
    # Train
    model = MultinomialNB()
    model.fit(X, labels)
    
    _NB_MODEL = model
    _VECTORIZER = vectorizer
    return model

def predict_spam(text):
    global _NB_MODEL, _VECTORIZER
    if _NB_MODEL is None:
        train_nb_model()
        
    try:
        # Transform input
        X_test = _VECTORIZER.transform([text])
        # Predict
        prediction = _NB_MODEL.predict(X_test)[0]
        # Probabilities
        proba = _NB_MODEL.predict_proba(X_test)[0]
        spam_prob = proba[1]
        
        return int(prediction), float(spam_prob)
    except Exception as e:
        print(f"Error predicting: {e}")
        return 0, 0.0
