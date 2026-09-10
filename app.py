import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Traffic Accident Hotspot Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.stApp {
background-color: #0E1117;
color: #F5F7FA;
}
section[data-testid="stSidebar"] {
background-color: #171C28;
}
section[data-testid="stSidebar"] * {
color: #E8EDF5;
}
.block-container {
padding-top: 2rem;
padding-bottom: 3rem;
max-width: 1400px;
}
.hero-container {
background: linear-gradient(135deg, rgba(22, 32, 52, 0.98), rgba(17, 25, 40, 0.98));
border: 1px solid #2C3B55;
border-radius: 22px;
padding: 55px 60px;
margin-bottom: 35px;
position: relative;
overflow: hidden;
}
.hero-container::before {
content: "";
position: absolute;
width: 400px;
height: 400px;
background: radial-gradient(circle, rgba(0, 210, 255, 0.12), transparent 70%);
top: -200px;
right: -100px;
}
.hero-badge {
display: inline-block;
padding: 9px 16px;
border-radius: 30px;
background-color: rgba(0, 210, 255, 0.10);
border: 1px solid rgba(0, 210, 255, 0.35);
color: #67D8FF;
font-size: 13px;
font-weight: 600;
letter-spacing: 1px;
margin-bottom: 25px;
}
.hero-title {
font-size: 52px;
font-weight: 800;
line-height: 1.15;
color: #F5F7FA;
margin-bottom: 22px;
}
.hero-highlight {
color: #38BDF8;
}
.hero-subtitle {
font-size: 18px;
line-height: 1.8;
color: #AAB7C7;
max-width: 780px;
}
.section-title {
font-size: 30px;
font-weight: 750;
margin-top: 25px;
margin-bottom: 25px;
color: #F5F7FA;
}
.section-subtitle {
color: #9AA8B8;
font-size: 16px;
margin-top: -10px;
margin-bottom: 25px;
}
.stat-card {
background-color: #171E2B;
border: 1px solid #2A364A;
border-radius: 18px;
padding: 28px 20px;
text-align: center;
height: 165px;
display: flex;
flex-direction: column;
justify-content: center;
transition: 0.2s;
}
.stat-number {
font-size: 32px;
font-weight: 800;
color: #4FC3F7;
margin-bottom: 10px;
}
.stat-label {
color: #AAB7C7;
font-size: 14px;
line-height: 1.5;
}
.feature-card {
background-color: #171E2B;
border: 1px solid #2A364A;
border-radius: 18px;
padding: 30px;
min-height: 260px;
}
.feature-icon {
font-size: 34px;
margin-bottom: 18px;
}
.feature-title {
font-size: 21px;
font-weight: 700;
color: #F5F7FA;
margin-bottom: 15px;
}
.feature-text {
color: #AAB7C7;
line-height: 1.7;
font-size: 15px;
}
.feature-tag {
margin-top: 18px;
color: #67D8FF;
font-size: 13px;
font-weight: 600;
}
.info-card {
background-color: #171E2B;
border: 1px solid #2A364A;
border-radius: 18px;
padding: 35px;
min-height: 290px;
}
.info-title {
font-size: 22px;
font-weight: 700;
margin-bottom: 20px;
color: #F5F7FA;
}
.info-text {
color: #AAB7C7;
font-size: 15px;
line-height: 1.8;
}
.pipeline-card {
background-color: #171E2B;
border: 1px solid #2A364A;
border-radius: 18px;
padding: 28px 15px;
text-align: center;
min-height: 190px;
}
.pipeline-icon {
font-size: 36px;
margin-bottom: 18px;
}
.pipeline-title {
font-size: 16px;
font-weight: 650;
color: #F5F7FA;
line-height: 1.5;
}
.model-card {
background: linear-gradient(135deg, rgba(22, 55, 45, 0.85), rgba(17, 30, 40, 0.95));
border: 1px solid #315D4D;
border-radius: 22px;
padding: 42px;
margin-top: 20px;
}
.model-title {
font-size: 28px;
font-weight: 750;
color: #8FE3B1;
margin-bottom: 15px;
}
.model-text {
color: #C5D5CC;
font-size: 16px;
line-height: 1.8;
max-width: 750px;
}
.model-metric {
background-color: rgba(255,255,255,0.05);
border-radius: 15px;
padding: 20px;
text-align: center;
}
.model-metric-number {
font-size: 27px;
font-weight: 800;
color: #8FE3B1;
}
.model-metric-label {
font-size: 13px;
color: #B7C7BE;
margin-top: 8px;
}
.home-footer {
text-align: center;
color: #7F8C9C;
margin-top: 50px;
padding: 30px;
border-top: 1px solid #273244;
}
div[data-testid="stMetric"] {
background-color: #171E2B;
border: 1px solid #2A364A;
padding: 18px;
border-radius: 14px;
}
div[data-testid="stDataFrame"] {
border-radius: 12px;
overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_PATH = BASE_DIR / "outputs" / "final_hotspot_predictions.csv"
VISUALIZATION_DIR = BASE_DIR / "outputs" / "visualizations"
FEATURE_IMPORTANCE_PATH = VISUALIZATION_DIR / "feature_importance.csv"
MODEL_COMPARISON_PATH = VISUALIZATION_DIR / "model_comparison.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_predictions():
    if PREDICTIONS_PATH.exists():
        return pd.read_csv(PREDICTIONS_PATH)
    return pd.DataFrame()


@st.cache_data
def load_feature_importance():
    if FEATURE_IMPORTANCE_PATH.exists():
        return pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return pd.DataFrame()


@st.cache_data
def load_model_comparison():
    if MODEL_COMPARISON_PATH.exists():
        return pd.read_csv(MODEL_COMPARISON_PATH)
    return pd.DataFrame()


predictions_df = load_predictions()
feature_importance_df = load_feature_importance()
model_comparison_df = load_model_comparison()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("# 🚦 Navigation")
    st.caption("Traffic Accident Hotspot Prediction")
    st.markdown("###")

    page = st.radio(
        "Select Dashboard Page",
        [
            "🏠 Home",
            "📍 Hotspot Predictions",
            "🗺️ Risk Map",
            "📊 Model Performance",
            "📈 Feature Importance"
        ]
    )

    st.markdown("---")
    st.markdown("### 🎯 Final Model")
    st.markdown("**Advanced Random Forest**")
    st.markdown("")
    st.markdown("**ROC-AUC:** 0.8508")
    st.markdown("")
    st.markdown("**F1 Score:** 0.2621")
    st.markdown("")
    st.markdown("**Threshold:** 0.90")
    st.markdown("---")
    st.caption("Spatio-Temporal Machine Learning System")


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    # --------------------------------------------------------
    # HERO SECTION
    # --------------------------------------------------------

    hero_html = """
<div class="hero-container">
<div class="hero-badge">🚦 SPATIO-TEMPORAL MACHINE LEARNING SYSTEM</div>
<div class="hero-title">Predict Accident Hotspots<br><span class="hero-highlight">Before They Happen</span></div>
<div class="hero-subtitle">An intelligent traffic accident prediction system that analyzes historical collision patterns, geographical locations, temporal behaviour, and advanced machine learning features to identify areas at risk of becoming future accident hotspots.</div>
</div>
"""
    st.markdown(hero_html, unsafe_allow_html=True)

    # --------------------------------------------------------
    # SYSTEM AT A GLANCE
    # --------------------------------------------------------

    st.markdown('<div class="section-title">📊 System at a Glance</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            '<div class="stat-card"><div class="stat-number">49,505</div>'
            '<div class="stat-label">Spatial Grids<br>Analyzed</div></div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            '<div class="stat-card"><div class="stat-number">5 Years</div>'
            '<div class="stat-label">Historical Collision<br>Data</div></div>',
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            '<div class="stat-card"><div class="stat-number">0.8508</div>'
            '<div class="stat-label">Final<br>ROC-AUC Score</div></div>',
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            '<div class="stat-card"><div class="stat-number">22</div>'
            '<div class="stat-label">Advanced Predictive<br>Features</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # WHAT DOES THE SYSTEM DO
    # --------------------------------------------------------

    st.markdown('<div class="section-title">⚡ What Does the System Do?</div>', unsafe_allow_html=True)

    feature_col1, feature_col2, feature_col3 = st.columns(3)

    with feature_col1:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">📍</div>'
            '<div class="feature-title">Analyze Location</div>'
            '<div class="feature-text">The geographical region is divided into spatial grids '
            'to identify locations where traffic accidents repeatedly occur.</div>'
            '<div class="feature-tag">Spatial Analysis → Grid-Based Accident Patterns</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with feature_col2:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">📅</div>'
            '<div class="feature-title">Learn Patterns</div>'
            '<div class="feature-text">Historical accident activity is analyzed over time using '
            'lag features, rolling patterns, collision frequency, and recency information.</div>'
            '<div class="feature-tag">Temporal Analysis → Monthly Prediction Patterns</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with feature_col3:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">🤖</div>'
            '<div class="feature-title">Predict Risk</div>'
            '<div class="feature-text">The Advanced Random Forest model estimates the probability '
            'that a location will become a traffic accident hotspot.</div>'
            '<div class="feature-tag">Decision Threshold → Probability ≥ 0.90</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # PROJECT PURPOSE
    # --------------------------------------------------------

    st.markdown('<div class="section-title">🎯 Why This Project Matters</div>', unsafe_allow_html=True)

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown(
            '<div class="info-card">'
            '<div class="info-title">🚨 The Challenge</div>'
            '<div class="info-text">'
            'Traffic accidents often occur repeatedly in particular geographical locations.<br><br>'
            'Identifying these high-risk areas from millions of historical records manually is '
            'difficult and time-consuming.<br><br>'
            'Accident patterns also change over time, making simple historical analysis '
            'insufficient for proactive planning.'
            '</div></div>',
            unsafe_allow_html=True
        )

    with info_col2:
        st.markdown(
            '<div class="info-card">'
            '<div class="info-title">🎯 Our Objective</div>'
            '<div class="info-text">'
            'Predict whether a spatial grid location is likely to become a traffic accident '
            'hotspot in the following month.<br><br>'
            'The goal is to support proactive road safety planning and help identify locations '
            'requiring greater attention.<br><br>'
            'The system combines spatial, temporal, and historical accident patterns for prediction.'
            '</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.markdown('<div class="section-title">🔄 How the System Works</div>', unsafe_allow_html=True)

    pipeline_col1, pipeline_col2, pipeline_col3, pipeline_col4 = st.columns(4)

    with pipeline_col1:
        st.markdown(
            '<div class="pipeline-card"><div class="pipeline-icon">📂</div>'
            '<div class="pipeline-title">Historical<br>Collision Data</div></div>',
            unsafe_allow_html=True
        )

    with pipeline_col2:
        st.markdown(
            '<div class="pipeline-card"><div class="pipeline-icon">⚙️</div>'
            '<div class="pipeline-title">Spatial + Temporal<br>Feature Engineering</div></div>',
            unsafe_allow_html=True
        )

    with pipeline_col3:
        st.markdown(
            '<div class="pipeline-card"><div class="pipeline-icon">🤖</div>'
            '<div class="pipeline-title">Advanced Random<br>Forest Model</div></div>',
            unsafe_allow_html=True
        )

    with pipeline_col4:
        st.markdown(
            '<div class="pipeline-card"><div class="pipeline-icon">🗺️</div>'
            '<div class="pipeline-title">Hotspot Risk<br>Prediction</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # FINAL MODEL
    # --------------------------------------------------------

    st.markdown('<div class="section-title">🏆 Final Selected Model</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="model-card">'
        '<div class="model-title">🤖 Advanced Random Forest</div>'
        '<div class="model-text">'
        'Selected as the final model after comparing Logistic Regression, Random Forest, and '
        'Advanced Random Forest.<br><br>'
        'The final model demonstrated the strongest overall predictive discrimination and achieved '
        'the highest F1 Score among the evaluated models.'
        '</div></div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.markdown(
            '<div class="model-metric"><div class="model-metric-number">0.8508</div>'
            '<div class="model-metric-label">ROC-AUC SCORE</div></div>',
            unsafe_allow_html=True
        )

    with metric2:
        st.markdown(
            '<div class="model-metric"><div class="model-metric-number">0.2621</div>'
            '<div class="model-metric-label">F1 SCORE</div></div>',
            unsafe_allow_html=True
        )

    with metric3:
        st.markdown(
            '<div class="model-metric"><div class="model-metric-number">0.90</div>'
            '<div class="model-metric-label">DECISION THRESHOLD</div></div>',
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.markdown(
        '<div class="home-footer">'
        '🚀 Explore the system using the navigation panel.<br><br>'
        'View predicted accident hotspots, explore geographical risk patterns, '
        'compare model performance, and analyze the most important predictive features.'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# HOTSPOT PREDICTIONS PAGE
# ============================================================

elif page == "📍 Hotspot Predictions":

    st.title("📍 Hotspot Predictions")
    st.write(
        "Explore locations predicted as traffic accident hotspots "
        "using the final Advanced Random Forest model."
    )

    if predictions_df.empty:
        st.error("Prediction file not found.")
    else:
        total_predictions = len(predictions_df)

        if "predicted_hotspot" in predictions_df.columns:
            predicted_hotspots = predictions_df[predictions_df["predicted_hotspot"] == 1].shape[0]
        else:
            predicted_hotspots = 6606

        hotspot_percentage = (predicted_hotspots / total_predictions) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", f"{total_predictions:,}")
        col2.metric("Predicted Hotspots", f"{predicted_hotspots:,}")
        col3.metric("Hotspot Percentage", f"{hotspot_percentage:.2f}%")

        st.markdown("---")

        if "risk_level" in predictions_df.columns:
            risk_options = sorted(predictions_df["risk_level"].dropna().unique())
            selected_risk = st.multiselect("Select Risk Levels", options=risk_options, default=risk_options)
            filtered_df = predictions_df[predictions_df["risk_level"].isin(selected_risk)]
        else:
            filtered_df = predictions_df.copy()

        if "year_month" in filtered_df.columns:
            months = sorted(filtered_df["year_month"].astype(str).unique())
            selected_month = st.selectbox("Select Prediction Month", ["All Months"] + months)
            if selected_month != "All Months":
                filtered_df = filtered_df[filtered_df["year_month"].astype(str) == selected_month]

        st.markdown("### 🔥 Highest Risk Predictions")

        if "prediction_probability" in filtered_df.columns:
            display_df = filtered_df.sort_values("prediction_probability", ascending=False)
        else:
            display_df = filtered_df.copy()

        preferred_columns = [
            "grid_id", "year_month", "grid_latitude", "grid_longitude",
            "prediction_probability", "risk_level", "predicted_hotspot"
        ]
        available_columns = [col for col in preferred_columns if col in display_df.columns]

        st.dataframe(display_df[available_columns], use_container_width=True, height=600)


# ============================================================
# RISK MAP PAGE
# ============================================================

elif page == "🗺️ Risk Map":

    st.title("🗺️ Accident Hotspot Risk Map")
    st.write("Geographical visualization of predicted accident risk locations.")

    if predictions_df.empty:
        st.error("Prediction file not found.")
    else:
        required_columns = ["grid_latitude", "grid_longitude"]

        if not all(col in predictions_df.columns for col in required_columns):
            st.error("Latitude and longitude columns were not found.")
        else:
            map_df = predictions_df.copy()

            if "year_month" in map_df.columns:
                months = sorted(map_df["year_month"].astype(str).unique())
                selected_month = st.selectbox("Select Month", months)
                map_df = map_df[map_df["year_month"].astype(str) == selected_month]

            if "risk_level" in map_df.columns:
                available_risks = sorted(map_df["risk_level"].dropna().unique())
                selected_risks = st.multiselect("Select Risk Levels", options=available_risks, default=available_risks)
                map_df = map_df[map_df["risk_level"].isin(selected_risks)]

            if "risk_level" in map_df.columns:
                color_map = {
                    "Low": [80, 180, 100],
                    "Medium": [255, 193, 7],
                    "High": [255, 140, 0],
                    "Critical": [220, 53, 69]
                }
                map_df["color"] = map_df["risk_level"].map(color_map)
                map_df["color"] = map_df["color"].apply(
                    lambda c: c if isinstance(c, list) else [150, 150, 150]
                )
            else:
                map_df["color"] = [[255, 140, 0]] * len(map_df)

            if len(map_df) > 50000:
                map_df = map_df.sample(50000, random_state=42)

            view_state = pdk.ViewState(
                latitude=float(map_df["grid_latitude"].mean()),
                longitude=float(map_df["grid_longitude"].mean()),
                zoom=8,
                pitch=0
            )

            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[grid_longitude, grid_latitude]",
                get_color="color",
                get_radius=500,
                pickable=True,
                opacity=0.7
            )

            deck = pdk.Deck(
                layers=[scatter_layer],
                initial_view_state=view_state,
                map_style=None
            )

            st.pydeck_chart(deck)

            st.info(
                "Risk Map Legend: "
                "🟢 Low Risk | 🟡 Medium Risk | 🟠 High Risk | 🔴 Critical Risk"
            )


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "📊 Model Performance":

    st.title("📊 Model Performance")
    st.write("Comparison of the machine learning models evaluated for accident hotspot prediction.")

    st.markdown("### 🏆 Final Model Performance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", "0.8508")
    col2.metric("Precision", "0.2667")
    col3.metric("Recall", "0.2577")
    col4.metric("F1 Score", "0.2621")

    st.markdown("---")

    st.markdown("### 🤖 Model Comparison")

    if not model_comparison_df.empty:
        st.dataframe(model_comparison_df, use_container_width=True)

        columns_lower = {col.lower(): col for col in model_comparison_df.columns}

        model_column = columns_lower.get("model")
        roc_column = columns_lower.get("roc-auc") or columns_lower.get("roc_auc")
        f1_column = columns_lower.get("f1 score") or columns_lower.get("f1_score")

        if model_column and roc_column:
            fig_roc = px.bar(
                model_comparison_df, x=model_column, y=roc_column,
                text=roc_column, title="ROC-AUC Comparison"
            )
            fig_roc.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig_roc, use_container_width=True)

        if model_column and f1_column:
            fig_f1 = px.bar(
                model_comparison_df, x=model_column, y=f1_column,
                text=f1_column, title="F1 Score Comparison"
            )
            fig_f1.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig_f1, use_container_width=True)

    else:
        fallback_models = pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest", "Advanced Random Forest"],
            "ROC-AUC": [0.7769, 0.7886, 0.8508],
            "F1 Score": [0.2187, 0.2298, 0.2621]
        })

        st.dataframe(fallback_models, use_container_width=True)

        fig = px.bar(fallback_models, x="Model", y="ROC-AUC", text="ROC-AUC", title="ROC-AUC Comparison")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("### 🎯 Final Model Selection")

    st.success(
        "Advanced Random Forest was selected as the final model because it achieved the "
        "highest overall F1 Score and strongest ROC-AUC performance among the evaluated models."
    )

    st.markdown(
        "**Final Configuration**\n\n"
        "- Model: Advanced Random Forest\n"
        "- ROC-AUC: 0.8508\n"
        "- Precision: 0.2667\n"
        "- Recall: 0.2577\n"
        "- F1 Score: 0.2621\n"
        "- Probability Threshold: 0.90"
    )


# ============================================================
# FEATURE IMPORTANCE PAGE
# ============================================================

elif page == "📈 Feature Importance":

    st.title("📈 Feature Importance")
    st.write("Analysis of the features contributing most to accident hotspot prediction.")

    if feature_importance_df.empty:
        st.warning("Feature importance file not found. Using known model results.")

        feature_importance_df = pd.DataFrame({
            "feature": [
                "historical_avg_collisions",
                "historical_collision_frequency",
                "grid_longitude",
                "collisions_last_12_months",
                "grid_latitude",
                "collisions_rolling_6",
                "historical_total_collisions",
                "historical_hotspot_frequency",
                "collisions_last_6_months",
                "historical_max_collisions"
            ],
            "importance": [
                0.171379, 0.149051, 0.094443, 0.094045, 0.089289,
                0.077320, 0.057468, 0.049213, 0.035407, 0.031825
            ]
        })

    feature_importance_df = feature_importance_df.sort_values("importance", ascending=True)

    fig = px.bar(
        feature_importance_df, x="importance", y="feature", orientation="h",
        title="Feature Importance - Advanced Random Forest"
    )
    fig.update_layout(template="plotly_dark", height=750, xaxis_title="Importance Score", yaxis_title="Feature")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("### 🔍 Most Important Predictive Features")

    top_features = feature_importance_df.sort_values("importance", ascending=False).head(5)
    st.dataframe(top_features, use_container_width=True)

    st.info(
        "The strongest predictive features are primarily related to historical collision "
        "activity, collision frequency, recent accident patterns, and spatial location. "
        "This confirms that both historical and spatio-temporal information contribute to "
        "hotspot prediction."
    )


# ============================================================
# END
# ============================================================