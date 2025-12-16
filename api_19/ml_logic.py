
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

# Values from previous mock training
TRAIN_DATA = [
    ("Win money now free cash prize", 1), # Spam
    ("Buy cheap meds viagra pills", 1),
    ("Click here to claim your reward", 1),
    ("Meeting tomorrow at 10am", 0), # Ham
    ("Hello friend how are you", 0),
    ("Project report attached", 0),
]

_NB_MODEL = None

def get_model():
    global _NB_MODEL
    if _NB_MODEL is None:
        model = SimpleNaiveBayes()
        model.train(TRAIN_DATA)
        _NB_MODEL = model
    return _NB_MODEL

def predict_spam(text):
    model = get_model()
    prediction, prob = model.predict(text)
    return prediction, prob
