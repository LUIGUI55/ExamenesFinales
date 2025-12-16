
# ML Exams Django Project

This project exposes 3 Machine Learning models as APIs with a Graphical Interface.

## Apps

1. **API 17 (K-Means)**: Fraud Detection.
   - Uses Mock Data (mimicking `creditcard.csv`).
   - Inputs: V10, V14.
   - Output: Cluster ID.
   - URL: `/17/`

2. **API 18 (DBSCAN)**: Fraud Detection.
   - Uses Mock Data (mimicking `creditcard.csv`).
   - Inputs: V10, V14.
   - Output: Cluster ID (Cluster -1 indicates Noise/Potential Fraud).
   - URL: `/18/`

3. **API 19 (Naive Bayes)**: Spam Detection.
   - Uses a tiny demonstration corpus (since `trec07p` was missing).
   - Inputs: Email Text.
   - Output: SPAM or HAM.
   - URL: `/19/`

## How to Run

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   *(Note: `scikit-learn`, `pandas`, `numpy`, `django` are required).*

2. **Run Server**:

   ```bash
   cd ML_Exams
   python manage.py runserver
   ```

3. **Access APIs**:
   - Go to `http://127.0.0.1:8000/17/` for K-Means.
   - Go to `http://127.0.0.1:8000/18/` for DBSCAN.
   - Go to `http://127.0.0.1:8000/19/` for Naive Bayes.

## Deployment (Vercel)

This project is configured for Vercel.

1. Install Vercel CLI (optional) or connect GitHub to Vercel.
2. If using CLI:

   ```bash
   vercel
   ```

3. Should deploy automatically as a Python Serverless Function.
