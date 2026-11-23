import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CREDIT DEFAULT RISK ANALYTICS — SINGLE-FILE STREAMLIT APP
# Navigation is implemented with st.sidebar.radio so you only
# need this one Python file. All reported metrics below are
# from the project's notebook/results supplied with the project.
# ============================================================

st.set_page_config(
    page_title="Credit Default Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"

# ---------------------------- Project constants ----------------------------
OPTIMAL_THRESHOLD_NOTEBOOK = 0.1834
OOF_PRECISION = 0.2451
OOF_RECALL = 0.7164
OOF_F2 = 0.5175
ENSEMBLE_WEIGHTS = "1 : 1"

FEATURES_DEFAULT = [
    "Age", "MonthlyIncome", "Dependents", "OpenCreditLines",
    "RealEstateLoans", "DebtRatio", "CreditUtil",
    "DPD_30_59", "DPD_60_89", "DPD_90plus"
]

MODEL_RESULTS = pd.DataFrame(
    [
        ["Logistic Regression", 0.8298, 0.1852, 0.7400, 0.2963, 0.3111],
        ["Tuned Random Forest", 0.8600, 0.4347, 0.4215, 0.4280, 0.3774],
        ["Untuned XGBoost", 0.8561, 0.5354, 0.1988, 0.2899, 0.3702],
        ["Tuned XGBoost", 0.8643, 0.5626, 0.2151, 0.3112, 0.3900],
        ["Weighted Ensemble (threshold 0.50)", 0.8644, 0.5227, 0.3059, 0.3859, 0.3910],
        ["Weighted Ensemble (threshold 0.1834)", 0.8644, 0.2473, 0.7211, 0.3683, 0.3910],
    ],
    columns=["Model", "ROC-AUC", "Precision", "Recall", "F1", "Average Precision"]
)

CHARTS = {
    "Model performance comparison": (
        "model_performance_comparison.png",
        "Test-set comparison of ROC-AUC, precision, recall and F1."
    ),
    "ROC curves": (
        "roc_curves.png",
        "ROC curves compare how well each model ranks positive cases above negative cases."
    ),
    "Precision–Recall curves": (
        "precision_recall_curves.png",
        "Precision–Recall curves are especially informative for this imbalanced target."
    ),
    "Multi-metric radar chart": (
        "multi_metric_radar.png",
        "A compact view of the precision, recall, F1 and ROC-AUC trade-offs."
    ),
    "Performance heatmap": (
        "performance_heatmap.png",
        "Heatmap of test metrics across the evaluated models."
    ),
}

# ---------------------------- Artifact loading -----------------------------
@st.cache_resource
def load_model_artifact():
    path = ROOT / "credit_risk_model.pkl"
    if not path.exists():
        return None, None, None, (
            "credit_risk_model.pkl is missing. Add the trained model artifact "
            "beside app.py to enable live predictions."
        )
    try:
        saved = joblib.load(path)
        model = saved["model"]
        threshold = float(saved.get("threshold", OPTIMAL_THRESHOLD_NOTEBOOK))
        features = list(saved.get("features", FEATURES_DEFAULT))
        if not hasattr(model, "predict_proba"):
            raise ValueError("The saved model does not expose predict_proba().")
        return model, threshold, features, None
    except Exception as exc:
        return None, None, None, f"Could not load credit_risk_model.pkl: {exc}"


@st.cache_resource
def load_shap_artifacts():
    background_path = ROOT / "shap_background.pkl"
    global_path = ROOT / "shap_global.pkl"

    background = joblib.load(background_path) if background_path.exists() else None
    global_data = joblib.load(global_path) if global_path.exists() else None

    if isinstance(global_data, dict):
        return (
            background,
            global_data.get("X_explain"),
            global_data.get("shap_values")
        )
    return background, None, None


model, threshold, model_features, model_error = load_model_artifact()
try:
    background, X_explain, global_shap = load_shap_artifacts()
except Exception as exc:
    background, X_explain, global_shap = None, None, None
    shap_error = str(exc)
else:
    shap_error = None

features = model_features or FEATURES_DEFAULT


def show_chart(filename, caption=None, width=None):
    """Display an image from assets/ when it exists."""
    path = ASSET_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(
            f"Chart file not found: assets/{filename}. "
            "Keep the supplied chart images in the assets folder."
        )


def show_shap_plot(plot_type, explanation, max_display=10):
    """Render a SHAP Explanation object safely in Streamlit."""
    try:
        plt.figure(figsize=(10, 6))
        if plot_type == "beeswarm":
            shap.plots.beeswarm(explanation, max_display=max_display, show=False)
        elif plot_type == "bar":
            shap.plots.bar(explanation, max_display=max_display, show=False)
        elif plot_type == "waterfall":
            shap.plots.waterfall(explanation, max_display=max_display, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
    finally:
        plt.close("all")


@st.cache_resource
def build_individual_explainer(_model, _background, feature_names):
    feature_names = list(feature_names)

    def predict_probability(data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=feature_names)
        return _model.predict_proba(data[feature_names])[:, 1]

    return shap.Explainer(
        predict_probability,
        shap.maskers.Independent(_background[feature_names]),
        algorithm="permutation",
        feature_names=feature_names
    )


# ---------------------------- Sidebar navigation ---------------------------
st.sidebar.title("💳 Credit Risk Intelligence")
st.sidebar.caption("Explainable machine learning portfolio project")
page = st.sidebar.radio(
    "Project navigation",
    [
        "Project Overview",
        "Project Workflow",
        "Model Findings",
        "Model Results",
        "SHAP Explainability",
        "Try the Model",
    ],
)
st.sidebar.divider()
st.sidebar.caption("Equal-weight XGBoost + Random Forest")
st.sidebar.caption("F2-optimized decision threshold")

# ================================ OVERVIEW ================================
if page == "Project Overview":
    st.title("Credit Default Risk Prediction")
    st.subheader(
        "An explainable machine learning system for identifying borrowers "
        "at risk of serious delinquency"
    )
    st.write(
        "This project estimates the probability that a borrower will experience "
        "serious delinquency within the next two years. It compares statistical "
        "and tree-based classifiers, combines tuned XGBoost and Random Forest "
        "probabilities, optimizes the classification threshold for recall, and "
        "uses SHAP to interpret model behavior."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original observations", "150,000")
    c2.metric("Predictor variables", "10")
    c3.metric("Held-out test set", "45,000")
    c4.metric("Ensemble ROC-AUC", "0.8644")

    st.divider()
    st.header("Problem definition")
    st.write(
        "The target variable indicates whether a borrower experienced serious "
        "delinquency (90 days or more past due) within two years. The original "
        "dataset contains 10,026 positive observations out of 150,000 records "
        "(approximately 6.68%), making this an imbalanced binary-classification problem."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Predictor groups")
        st.markdown(
            "- **Repayment history:** 30–59, 60–89 and 90+ days past due\n"
            "- **Credit behavior:** credit utilization and open credit lines\n"
            "- **Financial profile:** monthly income and debt ratio\n"
            "- **Borrower profile:** age and number of dependents\n"
            "- **Assets/liabilities:** real-estate loans"
        )
    with right:
        st.subheader("Application capabilities")
        st.markdown(
            "- Estimate a borrower's default probability\n"
            "- Apply the F2-optimized classification threshold\n"
            "- Compare model performance across several metrics\n"
            "- Explore global SHAP feature contributions\n"
            "- Generate an individual SHAP waterfall explanation"
        )

    st.info(
        "Final model: equal probability weights (1:1) for tuned XGBoost and "
        "Random Forest. The operating threshold is 0.1834, selected by "
        "maximizing out-of-fold F2-score to prioritize recall."
    )
    st.caption(
        "Dataset: Give Me Some Credit. This is a serious-delinquency prediction "
        "project, not a fraud-detection model."
    )

# ================================ WORKFLOW ================================
elif page == "Project Workflow":
    st.title("Project Workflow")
    st.write(
        "The workflow moves from data understanding and preprocessing to model "
        "selection, threshold optimization, explainability and deployment."
    )

    workflow = [
        ("01 · Exploratory data analysis",
         "Examined target imbalance, missing values, feature distributions, "
         "outliers, correlations and delinquency patterns."),
        ("02 · Data cleaning",
         "Removed observations with invalid Age = 0 and clipped numeric predictors "
         "at the 1st and 99th percentiles, following the notebook workflow."),
        ("03 · Train/test split",
         "Created a stratified 70/30 split with random_state=42, producing a "
         "45,000-observation held-out test set."),
        ("04 · Preprocessing comparison",
         "Compared alternative imputation and scaling strategies, including "
         "median/mode imputation, MinMax scaling, and IterativeImputer with "
         "RobustScaler and SMOTE."),
        ("05 · Model development",
         "Evaluated Logistic Regression, Random Forest and XGBoost. Imbalanced-learn "
         "pipelines were used where SMOTE was part of the modeling pipeline."),
        ("06 · Hyperparameter tuning",
         "Tuned Random Forest and XGBoost using cross-validation, with Average "
         "Precision used as the model-selection metric."),
        ("07 · Ensemble selection",
         "Combined tuned XGBoost and Random Forest predicted probabilities. "
         "Equal weights (1:1) achieved the best reported out-of-fold Average "
         "Precision among the tested weight combinations: 0.387872."),
        ("08 · Threshold optimization",
         "Selected a threshold of 0.1834 by maximizing out-of-fold F2-score. "
         "F2 places more emphasis on recall than precision."),
        ("09 · Explainability and deployment",
         "Used permutation-based SHAP explanations for global and individual "
         "interpretation, then packaged the fitted ensemble for Streamlit inference."),
    ]

    for heading, description in workflow:
        with st.container(border=True):
            st.markdown(f"**{heading}**")
            st.write(description)

    st.subheader("Why these evaluation choices?")
    st.markdown(
        "- **Average Precision:** evaluates positive-class ranking under class imbalance.\n"
        "- **F2-score:** gives recall greater weight than precision during threshold selection.\n"
        "- **Out-of-fold predictions:** provide training-side predictions for selecting "
        "the threshold without using the held-out test set.\n"
        "- **SHAP:** provides global and borrower-level explanations of model output."
    )

# ================================ FINDINGS ================================
elif page == "Model Findings":
    st.title("Key Model Findings")

    st.markdown("### 1 · Class imbalance makes accuracy insufficient")
    st.write(
        "Only about 6.68% of the original observations are positive cases. A model "
        "can therefore appear accurate while missing many borrowers who experience "
        "serious delinquency. The project emphasizes recall, Average Precision and "
        "Precision–Recall analysis rather than accuracy alone."
    )

    st.markdown("### 2 · The ensemble maintains strong ranking performance")
    st.write(
        "The equal-weight ensemble achieved a test ROC-AUC of 0.8644 and Average "
        "Precision of 0.3910. The ROC-AUC is marginally higher than the individual "
        "tuned models, while probability averaging combines the two fitted models."
    )

    st.markdown("### 3 · Threshold selection changes the operating point")
    st.write(
        "At a 0.50 threshold, the ensemble has precision 0.5227 and recall 0.3059. "
        "Using 0.1834 increases test recall to 0.7211, while precision decreases "
        "to 0.2473. This is an intentional trade-off driven by the F2 objective."
    )

    st.markdown("### 4 · Credit utilization and delinquency history dominate SHAP")
    st.write(
        "The supplied global SHAP plots rank CreditUtil as the largest average "
        "contributor. DPD_90plus, DPD_30_59 and DPD_60_89 are also influential. "
        "Higher delinquency values generally push the model output toward greater "
        "predicted default probability."
    )

    st.markdown("### 5 · Individual explanations are borrower-specific")
    st.write(
        "A SHAP waterfall plot starts from the model's expected output and adds "
        "or subtracts feature contributions until it reaches the selected "
        "borrower's predicted probability."
    )

    st.warning(
        "These are predictive associations, not causal effects. SHAP explains "
        "how the fitted model uses input features; it does not show that changing "
        "a feature would cause a change in a borrower's actual default risk."
    )

# ================================= RESULTS =================================
elif page == "Model Results":
    st.title("Model Performance and Comparison")
    st.write(
        "The table and charts summarize the reported held-out test performance. "
        "The optimized-threshold row changes the classification cutoff, not the "
        "underlying ensemble probability-ranking model."
    )

    st.dataframe(
        MODEL_RESULTS.style.format({
            "ROC-AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "Average Precision": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Model comparison visualizations")
    for title, (filename, caption) in CHARTS.items():
        st.subheader(title)
        show_chart(filename, caption)

    st.divider()
    st.subheader("F2-based threshold selection")
    a, b, c, d = st.columns(4)
    a.metric("Selected threshold", f"{OPTIMAL_THRESHOLD_NOTEBOOK:.4f}")
    b.metric("OOF precision", f"{OOF_PRECISION:.4f}")
    c.metric("OOF recall", f"{OOF_RECALL:.4f}")
    d.metric("OOF F2-score", f"{OOF_F2:.4f}")

    st.write(
        "The threshold was selected on out-of-fold training predictions by "
        "maximizing F2-score. Since F2 weights recall more heavily than precision, "
        "the selected threshold favors identifying more actual positive cases."
    )

    st.subheader("Held-out test confusion matrix at threshold 0.1834")
    cm = np.array([[35390, 6602], [839, 2169]])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual non-default", "Actual default"],
        columns=["Predicted non-default", "Predicted default"],
    )
    st.dataframe(cm_df, use_container_width=True)
    st.caption(
        "At the selected threshold, 2,169 of 3,008 actual defaults were identified "
        "(test recall 72.11%); 6,602 non-default observations were flagged."
    )

# ============================== EXPLAINABILITY =============================
elif page == "SHAP Explainability":
    st.title("SHAP Explainability")
    st.write(
        "SHAP (SHapley Additive exPlanations) attributes portions of a model "
        "prediction to individual features. The supplied plots summarize global "
        "feature behavior and one representative borrower."
    )

    st.subheader("Global feature impact · Beeswarm")
    show_chart(
        "shap_beeswarm.png",
        "Each dot represents one observation. Position shows the SHAP contribution; "
        "color indicates whether the feature value is relatively low or high."
    )
    st.write(
        "**Interpretation:** CreditUtil has a broad SHAP distribution, indicating "
        "substantial variation in its contribution across borrowers. High values "
        "often appear on the positive side, pushing predicted default probability upward. "
        "The delinquency variables also show positive contributions for some borrowers."
    )

    st.subheader("Global feature importance · Mean absolute SHAP")
    show_chart(
        "shap_global_importance.png",
        "Features are ranked by mean absolute SHAP value across the explained sample."
    )
    st.write(
        "**Interpretation:** CreditUtil ranks first (approximately 0.05 mean absolute "
        "SHAP value), followed by DPD_90plus and DPD_30_59 (approximately 0.03 each), "
        "then DPD_60_89 (approximately 0.02). The remaining features have smaller "
        "average absolute contributions in this sample."
    )

    st.subheader("Individual explanation · Waterfall")
    show_chart(
        "shap_waterfall.png",
        "Example borrower: the expected model output is moved by feature contributions "
        "to the final predicted probability."
    )
    st.write(
        "**Interpretation:** In the displayed example, zero delinquency counts and "
        "CreditUtil = 0.231 move the prediction downward relative to the model's "
        "baseline. Age = 37 and MonthlyIncome = 2,800 make small positive contributions. "
        "The final model output shown is approximately 0.063 (6.3%); the expected/baseline output is approximately 0.112 (11.2%)."
    )

    st.divider()
    st.subheader("Interactive SHAP for a new borrower")
    if background is None or model is None:
        st.info(
            "To calculate a new borrower-level SHAP waterfall, add both "
            "`credit_risk_model.pkl` and `shap_background.pkl` to the project root."
        )
    else:
        st.caption(
            "Use the Try the Model page to generate a prediction and its individual "
            "waterfall explanation."
        )

    st.caption(
        "SHAP values explain the model's predicted probability output. They are "
        "not the classification threshold and do not change when the threshold changes."
    )

# ================================ PREDICT ==================================
elif page == "Try the Model":
    st.title("Try the Credit Default Model")
    st.write(
        "Enter borrower characteristics to estimate the probability of serious "
        "delinquency within the next two years. The classification uses the "
        "threshold stored in the trained model artifact."
    )

    if model is None:
        st.error(model_error or "The trained model artifact is unavailable.")
        st.markdown(
            "Add `credit_risk_model.pkl` beside `app.py`. It must be a joblib "
            "dictionary containing `model`, `threshold` and `features`."
        )
    else:
        st.caption(f"Loaded model threshold: {threshold:.4f}")

        with st.form("borrower_input"):
            left, right = st.columns(2)
            with left:
                age = st.number_input("Age", min_value=1, max_value=110, value=37)
                income = st.number_input("Monthly income ($)", min_value=0, value=2800, step=100)
                dependents = st.number_input("Number of dependents", min_value=0, max_value=20, value=1)
                open_lines = st.number_input("Open credit lines", min_value=0, max_value=100, value=4)
                real_estate = st.number_input("Real-estate loans", min_value=0, max_value=30, value=0)
            with right:
                debt_ratio = st.number_input("Debt ratio", min_value=0.0, value=0.283, format="%.4f")
                credit_util = st.number_input("Credit utilization", min_value=0.0, value=0.231, format="%.4f")
                dpd_30_59 = st.number_input("30–59 days past due", min_value=0, max_value=100, value=0)
                dpd_60_89 = st.number_input("60–89 days past due", min_value=0, max_value=100, value=0)
                dpd_90plus = st.number_input("90+ days past due", min_value=0, max_value=100, value=0)

            submitted = st.form_submit_button(
                "Estimate Default Risk",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            input_values = {
                "Age": age,
                "MonthlyIncome": income,
                "Dependents": dependents,
                "OpenCreditLines": open_lines,
                "RealEstateLoans": real_estate,
                "DebtRatio": debt_ratio,
                "CreditUtil": credit_util,
                "DPD_30_59": dpd_30_59,
                "DPD_60_89": dpd_60_89,
                "DPD_90plus": dpd_90plus,
            }
            missing = [name for name in features if name not in input_values]
            if missing:
                st.error(f"The saved model expects unknown input features: {missing}")
            else:
                input_df = pd.DataFrame([input_values])[features]
                probability = float(model.predict_proba(input_df)[0, 1])
                prediction = int(probability >= threshold)

                st.divider()
                st.subheader("Prediction result")
                p1, p2, p3 = st.columns(3)
                p1.metric("Estimated default probability", f"{probability:.2%}")
                p2.metric("Decision threshold", f"{threshold:.4f}")
                p3.metric("Model classification", "Higher risk" if prediction else "Lower risk")

                if prediction:
                    st.error(
                        "The estimated probability is at or above the selected "
                        "threshold. The model classifies this borrower as higher risk."
                    )
                else:
                    st.success(
                        "The estimated probability is below the selected "
                        "threshold. The model classifies this borrower as lower risk."
                    )

                st.subheader("Individual SHAP explanation")
                if background is None:
                    st.warning(
                        "Add `shap_background.pkl` to enable individual SHAP explanations."
                    )
                else:
                    try:
                        with st.spinner("Calculating the individual SHAP explanation..."):
                            explainer = build_individual_explainer(
                                model, background, tuple(features)
                            )
                            individual_shap = explainer(
                                input_df,
                                max_evals=10 * (2 * len(features) + 1),
                            )
                        show_shap_plot("waterfall", individual_shap[0])
                    except Exception as exc:
                        st.error(f"Could not calculate the individual SHAP explanation: {exc}")

# --------------------------------- Footer ----------------------------------
st.sidebar.divider()
st.sidebar.caption("Credit Default Risk · XGBoost + Random Forest · SHAP")
st.divider()
st.caption(
    "Educational portfolio project. Predictions are model estimates and should "
    "not be used as the sole basis for real lending or other consequential decisions."
)
