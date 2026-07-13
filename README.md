# 🛍️ Data Driven Customer Churn Prediction & Personalized Product Recommendation System

A professional Streamlit dashboard that loads your **already-trained** models and
pre-computed artifacts — no retraining happens inside the app.

## 📁 Folder Contents

Place `app.py` in the **same folder** as all of these files (already included here):

```
app.py
requirements.txt
best_model.joblib
scaler.joblib
selected_features.joblib
user_item_matrix.pkl
item_similarity.pkl
product_lookup.pkl
customer_features.csv
customer_predictions.csv
predicted_churn_customers.csv
recommendations.csv
feature_importance.csv
model_results.csv
clean_data.csv
```

## ⚙️ Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 🧭 Pages

| Page | Description |
|---|---|
| 🏠 Home | Project overview, objectives, tech stack, key statistics |
| 📊 Dashboard | Churn distribution, RFM histograms, top products, feature importance, model accuracy |
| 🤖 Churn Prediction | Select a customer → predict Active/Churn with probability gauge |
| 🎁 Product Recommendation | Personalized top-15 product recommendations for at-risk customers |
| 📈 Model Performance | Compare Accuracy, Precision, Recall, F1, ROC-AUC across models |
| 📁 Customer Data | Search, filter, and download the customer dataset |
| ℹ️ About | Project details, methodology, developer information |

## ✏️ Personalize the About Page

Open `app.py` and edit these constants near the top:

```python
DEVELOPER_NAME = "Your Name Here"
UNIVERSITY_NAME = "Your University Name Here"
DEPARTMENT_NAME = "Department of Computer Science"
SUPERVISOR_NAME = "Your Supervisor's Name"
PROJECT_YEAR = "2025 - 2026"
```

## ℹ️ Notes

- The app **only loads** pre-trained/pre-computed files. It never retrains any model.
- If a required file is missing, the relevant page shows a warning instead of crashing.
- You may see a harmless scikit-learn version warning when loading `best_model.joblib` /
  `scaler.joblib` if your installed scikit-learn version differs slightly from the one
  used to save them. This does not affect predictions, but installing the exact same
  scikit-learn version used during training will remove the warning.
# fyp-project
