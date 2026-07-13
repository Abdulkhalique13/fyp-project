# ==========================================================================
# DATA DRIVEN CUSTOMER CHURN PREDICTION AND PERSONALIZED PRODUCT
# RECOMMENDATION SYSTEM
# Final Year Project - Streamlit Web Application
#
# This app ONLY loads pre-trained/pre-computed artifacts. No training,
# fitting, or model building happens inside this application.
# ==========================================================================

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# EDIT THIS SECTION WITH YOUR OWN DETAILS FOR THE "ABOUT" PAGE
# --------------------------------------------------------------------------
DEVELOPER_NAME = "Group-14"
UNIVERSITY_NAME = "Quest Univerity"
DEPARTMENT_NAME = "Department of Software Engineering"
SUPERVISOR_NAME = "Dr. Ali Raza"
PROJECT_YEAR = "2025 - 2026"

# --------------------------------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Prediction & Recommendation System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# BASE DIRECTORY & FILE PATHS
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "best_model": os.path.join(BASE_DIR, "best_model.joblib"),
    "scaler": os.path.join(BASE_DIR, "scaler.joblib"),
    "selected_features": os.path.join(BASE_DIR, "selected_features.joblib"),
    "user_item_matrix": os.path.join(BASE_DIR, "user_item_matrix.pkl"),
    "item_similarity": os.path.join(BASE_DIR, "item_similarity.pkl"),
    "product_lookup": os.path.join(BASE_DIR, "product_lookup.pkl"),
    "customer_features": os.path.join(BASE_DIR, "customer_features.csv"),
    "customer_predictions": os.path.join(BASE_DIR, "customer_predictions.csv"),
    "predicted_churn_customers": os.path.join(BASE_DIR, "predicted_churn_customers.csv"),
    "recommendations": os.path.join(BASE_DIR, "recommendations.csv"),
    "feature_importance": os.path.join(BASE_DIR, "feature_importance.csv"),
    "model_results": os.path.join(BASE_DIR, "model_results.csv"),
    "clean_data": os.path.join(BASE_DIR, "clean_data.csv"),
}

# --------------------------------------------------------------------------
# CUSTOM CSS - PROFESSIONAL BLUE THEME
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    }

    /* App background */
    .stApp {
        background: linear-gradient(180deg, #f4f7fb 0%, #eef2f9 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b2447 0%, #1b3b6f 60%, #2563eb 130%);
    }
    section[data-testid="stSidebar"] * {
        color: #f0f4fa !important;
    }

    /* Titles */
    h1, h2, h3 {
        color: #0b2447;
        font-weight: 700;
    }

    /* Card component */
    .custom-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 4px 18px rgba(15, 40, 90, 0.08);
        border: 1px solid #e6ecf5;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 18px;
    }
    .custom-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 26px rgba(15, 40, 90, 0.16);
    }

    /* Metric box */
    .metric-box {
        background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
        border-radius: 16px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.30);
        transition: transform 0.2s ease;
    }
    .metric-box:hover {
        transform: translateY(-4px);
    }
    .metric-box .metric-value {
        font-size: 32px;
        font-weight: 800;
        margin: 4px 0;
    }
    .metric-box .metric-label {
        font-size: 14px;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Tech badge */
    .tech-badge {
        display: inline-block;
        background: #e8f0fe;
        color: #1e40af;
        border: 1px solid #c3d7fb;
        border-radius: 999px;
        padding: 7px 16px;
        margin: 5px;
        font-weight: 600;
        font-size: 14px;
    }

    /* Prediction result boxes */
    .result-active {
        background: linear-gradient(135deg, #16a34a 0%, #22c55e 100%);
        color: white;
        border-radius: 16px;
        padding: 26px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        box-shadow: 0 8px 22px rgba(34, 197, 94, 0.30);
    }
    .result-churn {
        background: linear-gradient(135deg, #b91c1c 0%, #ef4444 100%);
        color: white;
        border-radius: 16px;
        padding: 26px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        box-shadow: 0 8px 22px rgba(239, 68, 68, 0.30);
    }

    /* Section divider */
    .section-title {
        border-left: 6px solid #2563eb;
        padding-left: 12px;
        margin: 18px 0 10px 0;
        color: #0b2447;
        font-weight: 700;
    }

    /* Dataframe container */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Buttons */
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 600;
        transition: transform 0.15s ease;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-2px);
        color: white;
    }

    /* Badge pill for best model */
    .best-badge {
        background: #fde68a;
        color: #7c4a03;
        padding: 3px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 12px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# SAFE LOADING HELPERS (cached, with error handling)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_joblib_file(path):
    """Load a joblib artifact (model / scaler / list). Returns None on failure."""
    try:
        if not os.path.exists(path):
            return None
        return joblib.load(path)
    except Exception as e:
        st.error(f"⚠️ Could not load `{os.path.basename(path)}`: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_pickle_joblib(path):
    """Load pickle-style artifacts saved via joblib (matrices / lookups)."""
    try:
        if not os.path.exists(path):
            return None
        return joblib.load(path)
    except Exception as e:
        st.error(f"⚠️ Could not load `{os.path.basename(path)}`: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_csv_file(path):
    """Load a CSV file into a DataFrame. Returns empty DataFrame on failure."""
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"⚠️ Could not load `{os.path.basename(path)}`: {e}")
        return pd.DataFrame()


# --------------------------------------------------------------------------
# LOAD ALL ARTIFACTS ONCE
# --------------------------------------------------------------------------
best_model = load_joblib_file(PATHS["best_model"])
scaler = load_joblib_file(PATHS["scaler"])
selected_features = load_joblib_file(PATHS["selected_features"])
user_item_matrix = load_pickle_joblib(PATHS["user_item_matrix"])
item_similarity = load_pickle_joblib(PATHS["item_similarity"])
product_lookup = load_pickle_joblib(PATHS["product_lookup"])

customer_features_df = load_csv_file(PATHS["customer_features"])
customer_predictions_df = load_csv_file(PATHS["customer_predictions"])
predicted_churn_customers_df = load_csv_file(PATHS["predicted_churn_customers"])
recommendations_df = load_csv_file(PATHS["recommendations"])
feature_importance_df = load_csv_file(PATHS["feature_importance"])
model_results_df = load_csv_file(PATHS["model_results"])
clean_data_df = load_csv_file(PATHS["clean_data"])


# --------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------
def metric_box(label, value, icon=""):
    st.markdown(
        f"""
        <div class="metric-box">
            <div style="font-size:26px;">{icon}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text):
    st.markdown(f'<h3 class="section-title">{text}</h3>', unsafe_allow_html=True)


def predict_customer_churn(customer_id):
    """
    Predict churn for a given CustomerID using saved model + scaler.
    Returns dict with prediction, probability, and customer row -- or None
    if the customer / required artifacts are unavailable.
    """
    if customer_features_df.empty or best_model is None or scaler is None or not selected_features:
        return None

    row = customer_features_df[customer_features_df["CustomerID"] == customer_id]
    if row.empty:
        return None

    try:
        X = row[selected_features].copy()
        X_scaled = scaler.transform(X)
        pred = best_model.predict(X_scaled)[0]
        try:
            proba = best_model.predict_proba(X_scaled)[0]
            churn_proba = float(proba[1]) if len(proba) > 1 else float(pred)
        except Exception:
            churn_proba = float(pred)

        return {
            "prediction": int(pred),
            "probability": churn_proba,
            "row": row.iloc[0],
        }
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        return None


def file_missing_warning(name):
    st.warning(f"⚠️ Required file `{name}` was not found or could not be loaded. This section is unavailable.")


# --------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 20px 0;">
            <div style="font-size:40px;">🛍️</div>
            <div style="font-size:18px; font-weight:800; letter-spacing:0.5px;">
                CHURN & RECOMMENDATION
            </div>
            <div style="font-size:12px; opacity:0.8;">Final Year Project System</div>
        </div>
        <hr style="border-color: rgba(255,255,255,0.2);">
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📊 Dashboard",
            "🤖 Churn Prediction",
            "🎁 Product Recommendation",
            "📈 Model Performance",
            "📁 Customer Data",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color: rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size:12px; text-align:center; opacity:0.75; padding-top:10px;">
            Built with Streamlit 🚀<br>Powered by Machine Learning
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================================
# PAGE 1: HOME
# ==========================================================================
def render_home():
    st.markdown(
        """
        <div class="custom-card" style="text-align:center; background: linear-gradient(135deg, #0b2447 0%, #1e40af 100%); color:white;">
            <div style="font-size:42px;">🛍️ 📉</div>
            <h1 style="color:white; margin-bottom:4px;">Data Driven Customer Churn Prediction</h1>
            <h2 style="color:#dbe8ff; font-weight:500; margin-top:0;">and Personalized Product Recommendation System</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        section_title("📌 Project Description")
        st.write(
            """
            This system analyzes historical transaction data to identify customers who are
            likely to **churn** (stop purchasing) and provides **personalized product
            recommendations** to encourage re-engagement. It combines supervised machine
            learning for churn classification with an item-based collaborative filtering
            engine for recommendations, wrapped in an interactive business-intelligence
            dashboard for decision-makers.
            """
        )
        section_title("🎯 Objectives")
        st.markdown(
            """
            - Predict whether a customer is likely to **churn** or remain **active**.
            - Identify key behavioral drivers of churn using feature importance analysis.
            - Recommend **personalized products** to customers at risk of churning.
            - Compare multiple ML algorithms and select the best-performing model.
            - Present insights through a clean, interactive, and professional dashboard.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        section_title("🛠️ Technologies Used")
        techs = [
            "Python", "Machine Learning", "Random Forest", "XGBoost",
            "Logistic Regression", "KNN", "Pandas", "NumPy",
            "Scikit-Learn", "Streamlit",
        ]
        badges_html = "".join([f'<span class="tech-badge">{t}</span>' for t in techs])
        st.markdown(badges_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Statistics ----
    section_title("📊 Key Statistics")

    if not customer_features_df.empty:
        total_customers = len(customer_features_df)
        churn_col = "PredictedChurn" if "PredictedChurn" in customer_features_df.columns else "Churn"
        predicted_churn = int(customer_features_df[churn_col].sum()) if churn_col in customer_features_df.columns else 0
        active_customers = total_customers - predicted_churn
    else:
        total_customers = predicted_churn = active_customers = 0

    if product_lookup is not None:
        total_products = len(product_lookup)
    elif not clean_data_df.empty and "StockCode" in clean_data_df.columns:
        total_products = clean_data_df["StockCode"].nunique()
    else:
        total_products = 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_box("Total Customers", f"{total_customers:,}", "👥")
    with c2:
        metric_box("Predicted Churn", f"{predicted_churn:,}", "⚠️")
    with c3:
        metric_box("Active Customers", f"{active_customers:,}", "✅")
    with c4:
        metric_box("Total Products", f"{total_products:,}", "📦")


# ==========================================================================
# PAGE 2: DASHBOARD
# ==========================================================================
def render_dashboard():
    st.title("📊 Analytics Dashboard")

    if customer_features_df.empty:
        file_missing_warning("customer_features.csv")
        return

    churn_col = "PredictedChurn" if "PredictedChurn" in customer_features_df.columns else "Churn"

    # ---- Row 1: Churn distribution + Pie chart ----
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        section_title("Churn Distribution")
        if churn_col in customer_features_df.columns:
            counts = customer_features_df[churn_col].value_counts().rename(
                index={0: "Active", 1: "Churn"}
            )
            fig = px.bar(
                x=counts.index, y=counts.values,
                labels={"x": "Customer Status", "y": "Number of Customers"},
                color=counts.index,
                color_discrete_map={"Active": "#22c55e", "Churn": "#ef4444"},
                text=counts.values,
            )
            fig.update_layout(showlegend=False, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Churn column not available.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        section_title("Churn vs Active (Proportion)")
        if churn_col in customer_features_df.columns:
            counts = customer_features_df[churn_col].value_counts().rename(
                index={0: "Active", 1: "Churn"}
            )
            fig = px.pie(
                names=counts.index, values=counts.values,
                color=counts.index,
                color_discrete_map={"Active": "#22c55e", "Churn": "#ef4444"},
                hole=0.45,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Churn column not available.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Row 2: RFM Histograms ----
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    section_title("📈 Customer Behavior Distributions (RFM)")
    h1, h2, h3 = st.columns(3)
    with h1:
        if "Recency" in customer_features_df.columns:
            fig = px.histogram(customer_features_df, x="Recency", nbins=30, color_discrete_sequence=["#2563eb"])
            fig.update_layout(title="Recency (days)", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
    with h2:
        if "Monetary" in customer_features_df.columns:
            fig = px.histogram(customer_features_df, x="Monetary", nbins=30, color_discrete_sequence=["#1e40af"])
            fig.update_layout(title="Monetary Value", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
    with h3:
        if "Frequency" in customer_features_df.columns:
            fig = px.histogram(customer_features_df, x="Frequency", nbins=30, color_discrete_sequence=["#0b2447"])
            fig.update_layout(title="Purchase Frequency", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Row 3: Top products + Feature importance ----
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        section_title("🏆 Top 10 Most Purchased Products")
        if not clean_data_df.empty and {"Description", "Quantity"}.issubset(clean_data_df.columns):
            top_products = (
                clean_data_df.groupby("Description")["Quantity"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            fig = px.bar(
                top_products, x="Quantity", y="Description",
                orientation="h", color="Quantity",
                color_continuous_scale="Blues",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            file_missing_warning("clean_data.csv")
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        section_title("🔑 Feature Importance")
        if not feature_importance_df.empty:
            fi_sorted = feature_importance_df.sort_values("Importance", ascending=True)
            fig = px.bar(
                fi_sorted, x="Importance", y="Feature",
                orientation="h", color="Importance",
                color_continuous_scale="Blues",
            )
            fig.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            file_missing_warning("feature_importance.csv")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Row 4: Model accuracy comparison ----
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    section_title("🧪 Model Accuracy Comparison")
    if not model_results_df.empty:
        fig = px.bar(
            model_results_df, x="Model", y="Accuracy",
            color="Model", text="Accuracy",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        fig.update_layout(showlegend=False, plot_bgcolor="white", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        file_missing_warning("model_results.csv")
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================================
# PAGE 3: CHURN PREDICTION
# ==========================================================================
def render_churn_prediction():
    st.title("🤖 Churn Prediction")

    if customer_features_df.empty:
        file_missing_warning("customer_features.csv")
        return
    if best_model is None:
        file_missing_warning("best_model.joblib")
        return
    if scaler is None:
        file_missing_warning("scaler.joblib")
        return
    if not selected_features:
        file_missing_warning("selected_features.joblib")
        return

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    section_title("Select a Customer")
    customer_ids = sorted(customer_features_df["CustomerID"].unique().tolist())
    selected_customer = st.selectbox("Customer ID", customer_ids)
    st.markdown("</div>", unsafe_allow_html=True)

    if selected_customer is None:
        return

    result = predict_customer_churn(selected_customer)

    if result is None:
        st.error("⚠️ Invalid Customer ID or prediction could not be computed.")
        return

    pred = result["prediction"]
    proba = result["probability"]
    cust_row = result["row"]

    col1, col2 = st.columns([1, 1])

    with col1:
        if pred == 1:
            st.markdown(
                f'<div class="result-churn">⚠️ Prediction: CHURN<br>'
                f'<span style="font-size:16px; font-weight:400;">Churn Probability: {proba:.1%}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-active">✅ Prediction: ACTIVE<br>'
                f'<span style="font-size:16px; font-weight:400;">Churn Probability: {proba:.1%}</span></div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        section_title("Prediction Confidence")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ef4444" if pred == 1 else "#22c55e"},
                "steps": [
                    {"range": [0, 50], "color": "#dcfce7"},
                    {"range": [50, 100], "color": "#fee2e2"},
                ],
            },
        ))
        fig.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    section_title("👤 Customer Information")
    info_df = cust_row.to_frame().reset_index()
    info_df.columns = ["Feature", "Value"]
    st.dataframe(info_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================================
# PAGE 4: PRODUCT RECOMMENDATION
# ==========================================================================
def render_product_recommendation():
    st.title("🎁 Product Recommendation")

    if customer_features_df.empty:
        file_missing_warning("customer_features.csv")
        return

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    section_title("Select a Customer")
    customer_ids = sorted(customer_features_df["CustomerID"].unique().tolist())
    selected_customer = st.selectbox("Customer ID", customer_ids, key="rec_customer")
    st.markdown("</div>", unsafe_allow_html=True)

    if selected_customer is None:
        return

    result = predict_customer_churn(selected_customer)

    if result is None:
        st.error("⚠️ Invalid Customer ID or prediction could not be computed.")
        return

    pred = result["prediction"]

    if pred == 0:
        st.markdown(
            '<div class="result-active">✅ This customer is not predicted to churn.<br>'
            '<span style="font-size:15px; font-weight:400;">Product recommendations are not required.</span></div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        '<div class="result-churn">⚠️ This customer is predicted to CHURN.<br>'
        '<span style="font-size:15px; font-weight:400;">Showing personalized recommendations to encourage re-engagement.</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    section_title("🎁 Top 15 Recommended Products")

    if recommendations_df.empty:
        file_missing_warning("recommendations.csv")
    else:
        cust_recs = recommendations_df[recommendations_df["CustomerID"] == selected_customer]
        if cust_recs.empty:
            st.info("No recommendations found for this customer.")
        else:
            cust_recs = cust_recs.sort_values("Score", ascending=False).head(15)
            display_df = cust_recs[["StockCode", "Description", "Score"]].rename(
                columns={
                    "StockCode": "Product Code",
                    "Description": "Product Description",
                    "Score": "Recommendation Score",
                }
            ).reset_index(drop=True)
            display_df["Recommendation Score"] = display_df["Recommendation Score"].round(4)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            fig = px.bar(
                display_df.sort_values("Recommendation Score"),
                x="Recommendation Score", y="Product Description",
                orientation="h", color="Recommendation Score",
                color_continuous_scale="Blues",
            )
            fig.update_layout(plot_bgcolor="white", height=500)
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================================
# PAGE 5: MODEL PERFORMANCE
# ==========================================================================
def render_model_performance():
    st.title("📈 Model Performance")

    if model_results_df.empty:
        file_missing_warning("model_results.csv")
        return

    metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"] if c in model_results_df.columns]
    best_model_name = None
    if "Accuracy" in model_results_df.columns:
        best_idx = model_results_df["Accuracy"].idxmax()
        best_model_name = model_results_df.loc[best_idx, "Model"]

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    section_title("📋 Model Comparison Table")
    if best_model_name:
        st.markdown(
            f'Best performing model: <span class="best-badge">🏆 {best_model_name}</span>',
            unsafe_allow_html=True,
        )

    def highlight_best(row):
        if best_model_name and row["Model"] == best_model_name:
            return ["background-color: #dbeafe; font-weight: 700;"] * len(row)
        return [""] * len(row)

    styled = model_results_df.style.apply(highlight_best, axis=1)
    format_dict = {c: "{:.2%}" for c in metric_cols}
    styled = styled.format(format_dict)
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    section_title("📊 Metric Comparison Across Models")
    if metric_cols:
        melted = model_results_df.melt(id_vars="Model", value_vars=metric_cols,
                                        var_name="Metric", value_name="Score")
        fig = px.bar(
            melted, x="Model", y="Score", color="Metric",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(plot_bgcolor="white", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Individual metric charts
    cols = st.columns(len(metric_cols)) if metric_cols else []
    for col, metric in zip(cols, metric_cols):
        with col:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            fig = px.bar(
                model_results_df, x="Model", y=metric, color="Model",
                text=metric, color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
            fig.update_layout(showlegend=False, title=metric, plot_bgcolor="white", yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================================
# PAGE 6: CUSTOMER DATA
# ==========================================================================
def render_customer_data():
    st.title("📁 Customer Data Explorer")

    if customer_features_df.empty:
        file_missing_warning("customer_features.csv")
        return

    churn_col = "PredictedChurn" if "PredictedChurn" in customer_features_df.columns else "Churn"

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    section_title("🔍 Search & Filter")
    c1, c2 = st.columns([1, 1])
    with c1:
        search_id = st.text_input("Search by Customer ID (leave blank to show all)")
    with c2:
        filter_option = st.radio(
            "Filter", ["All Customers", "Only Churn Customers", "Only Active Customers"],
            horizontal=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    filtered_df = customer_features_df.copy()

    if churn_col in filtered_df.columns:
        if filter_option == "Only Churn Customers":
            filtered_df = filtered_df[filtered_df[churn_col] == 1]
        elif filter_option == "Only Active Customers":
            filtered_df = filtered_df[filtered_df[churn_col] == 0]

    if search_id.strip():
        try:
            search_val = int(search_id.strip())
            filtered_df = filtered_df[filtered_df["CustomerID"] == search_val]
            if filtered_df.empty:
                st.warning(f"⚠️ No customer found with ID `{search_val}`.")
        except ValueError:
            st.error("⚠️ Please enter a valid numeric Customer ID.")
            filtered_df = filtered_df.iloc[0:0]

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    section_title(f"📄 Customer Records ({len(filtered_df):,} shown)")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Data as CSV",
        data=csv_data,
        file_name="customer_data_export.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================================
# PAGE 7: ABOUT
# ==========================================================================
def render_about():
    st.title("ℹ️ About This Project")

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    section_title("📌 Project Name")
    st.write("**Data Driven Customer Churn Prediction and Personalized Product Recommendation System**")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    section_title("❓ Problem Statement")
    st.write(
        """
        Retail and e-commerce businesses lose significant revenue when customers silently
        churn without any warning signs being acted upon in time. Identifying at-risk
        customers early and re-engaging them with relevant product offers can
        significantly reduce revenue loss and improve customer lifetime value. This
        project addresses that problem using historical transactional data.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    section_title("🧭 Methodology")
    st.markdown(
        """
        1. **Data Cleaning & Preprocessing** — Removing duplicates, handling missing
           values, and engineering RFM (Recency, Frequency, Monetary) based features.
        2. **Feature Engineering** — Deriving behavioral features such as purchase
           velocity, basket size, purchase consistency, and customer lifetime.
        3. **Model Training** — Training and evaluating multiple classification
           algorithms to predict churn.
        4. **Model Selection** — Selecting the best model based on Accuracy,
           Precision, Recall, F1-Score, and ROC-AUC.
        5. **Recommendation Engine** — Building an item-based collaborative
           filtering model to recommend products to at-risk customers.
        6. **Deployment** — Serving predictions and recommendations through this
           interactive Streamlit dashboard.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        section_title("🧠 Algorithms Used")
        st.markdown(
            """
            - Logistic Regression
            - Random Forest Classifier
            - XGBoost Classifier
            - K-Nearest Neighbors (KNN)
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        section_title("🎁 Recommendation System")
        st.write(
            """
            An **item-based collaborative filtering** approach is used, computing
            similarity between products from the customer-item purchase matrix and
            recommending the most similar, highly-scored products to at-risk customers.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    section_title("👨‍💻 Developer Information")
    st.markdown(
        f"""
        - **Developer:** {Group-14}
        - **University:** {Quest University}
        - **Department:** {Software Engineering}
        - **Supervisor:** {Dr Ali Raza}
        - **Academic Year:** {2025-2026}
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# PAGE ROUTER
# --------------------------------------------------------------------------
PAGE_ROUTER = {
    "🏠 Home": render_home,
    "📊 Dashboard": render_dashboard,
    "🤖 Churn Prediction": render_churn_prediction,
    "🎁 Product Recommendation": render_product_recommendation,
    "📈 Model Performance": render_model_performance,
    "📁 Customer Data": render_customer_data,
    "ℹ️ About": render_about,
}

PAGE_ROUTER[page]()
