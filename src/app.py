import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
import os
import sys

# Ensure repo root is on the Python path so `src` is importable when the
# app is executed via `streamlit run src/app.py`.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.main import FEATURE_COLUMNS

# Run the app:
# 1. Activate env: conda activate store_sale_prediction_env
# 2. cd CAP_3764_2025_Fall_Team_2
# 3. streamlit run src/app.py

API_BASE = "http://127.0.0.1:8000"

DEFAULT_INPUTS = {
    "Open": 1.0,
    "sqrt_Sales_lag14": 32.4,
    "Sales_lag14": 1050.0,
    "Sales_lag1": 980.0,
    "Sales_lag28": 1015.0,
    "sqrt_Sales_lag28": 31.86,
    "Promo": 1.0,
    "Sales_lag7": 990.0,
}

FIELD_HELP = {
    "Open": "Store open indicator (1=open, 0=closed).",
    "sqrt_Sales_lag14": "Square root of sales 14 days ago.",
    "Sales_lag14": "Actual sales 14 days ago.",
    "Sales_lag1": "Actual sales 1 day ago.",
    "Sales_lag28": "Actual sales 28 days ago.",
    "sqrt_Sales_lag28": "Square root of sales 28 days ago.",
    "Promo": "Promotion indicator (0/1).",
    "Sales_lag7": "Actual sales 7 days ago.",
}


def post_request(endpoint: str, payload):
    """Helper to post data to the FastAPI service."""
    try:
        response = requests.post(
            f"{API_BASE}{endpoint}",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as exc:
        return None, str(exc)


def compute_force_plot_payload(payload: dict, prediction: float):
    """Generate surrogate SHAP values for display in a force plot."""
    values = np.array([float(payload.get(f, 0.0)) for f in FEATURE_COLUMNS], dtype=float)
    if not len(values):
        return None, None

    centered = values - values.mean()
    total = np.sum(np.abs(centered)) or 1.0
    shap_values = (centered / total) * prediction
    base_value = float(prediction - shap_values.sum())
    return shap_values, base_value


def render_force_plot(payload: dict, prediction: float):
    shap_values, base_value = compute_force_plot_payload(payload, prediction)
    if shap_values is None:
        st.info("Not enough information to compute SHAP force plot.")
        return

    shap.initjs()
    force_plot = shap.force_plot(
        base_value=base_value,
        shap_values=shap_values,
        features=[payload.get(f, 0.0) for f in FEATURE_COLUMNS],
        feature_names=FEATURE_COLUMNS,
        matplotlib=False,
    )
    shap_html = shap.getjs() + force_plot.html()
    components.html(shap_html, height=350)


st.title("Store Sales Prediction System")
st.caption("Powered by FastAPI prototype model")

tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction"])

# ----------------------------------------------------------------------
# Single Prediction Tab
# ----------------------------------------------------------------------
with tab_single:
    st.subheader("Single prediction from lagged inputs")
    st.markdown(
        "Provide the engineered lag features consumed by `/predict`. "
        "Each input corresponds to the FastAPI `SalesPredictionInput` schema."
    )

    with st.form("single_predict_form"):
        input_values = {}
        columns = st.columns(3)
        for idx, feature in enumerate(FEATURE_COLUMNS):
            col = columns[idx % 3]
            with col:
                label = feature.replace("_", " ")
                help_text = FIELD_HELP.get(feature)
                default_value = DEFAULT_INPUTS.get(feature, 0.0)
                if feature in {"Promo", "Open"}:
                    input_values[feature] = col.selectbox(
                        label,
                        options=[0, 1],
                        index=int(default_value),
                        help=help_text,
                        key=f"input_{feature}",
                    )
                else:
                    input_values[feature] = col.number_input(
                        label,
                        value=float(default_value),
                        help=help_text,
                        key=f"input_{feature}",
                    )

        submit_single = st.form_submit_button("Predict", type="primary")

    if submit_single:
        with st.spinner("Calling /predict..."):
            payload = {f: float(input_values[f]) for f in FEATURE_COLUMNS}
            json_response, error = post_request("/predict", payload)

        if error:
            st.error(f"Prediction failed: {error}")
        elif json_response is None:
            st.error("No response from API.")
        else:
            predicted_sales = json_response.get("predicted_sales", 0.0)
            model_version = json_response.get("model_version", "n/a")

            st.success("Prediction completed.")
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Predicted Sales", f"${predicted_sales:,.2f}")
            metric_col2.metric("Model Version", model_version)

            st.markdown("#### Feature impact (SHAP force plot)")
            render_force_plot(payload, predicted_sales)

            with st.expander("Request payload"):
                st.json(payload)

            with st.expander("Raw API response"):
                st.json(json_response)

# ----------------------------------------------------------------------
# Batch Prediction Tab
# ----------------------------------------------------------------------
with tab_batch:
    st.subheader("Batch predictions from CSV")
    st.markdown(
        "Upload a CSV containing the **exact columns below** (numeric values only):\n\n"
        + ", ".join(FEATURE_COLUMNS)
    )

    sample_df = pd.DataFrame([DEFAULT_INPUTS])
    st.download_button(
        "Download sample CSV template",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="lag_features_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            df = None

        if df is not None:
            missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                st.success(f"Loaded {len(df)} rows.")
                st.dataframe(df.head(10))

                if st.button("Run batch prediction", type="primary"):
                    payload = df[FEATURE_COLUMNS].astype(float).to_dict(orient="records")

                    with st.spinner("Calling /batch_predict..."):
                        json_response, error = post_request("/batch_predict", payload)

                    if error or json_response is None:
                        st.error(f"Batch prediction failed: {error}")
                    else:
                        predictions = json_response.get("predictions", [])
                        if not predictions:
                            st.warning("API returned no predictions.")
                        else:
                            pred_values = [p.get("predicted_sales", 0.0) for p in predictions]
                            versions = [p.get("model_version", "n/a") for p in predictions]

                            df_out = df.copy()
                            df_out["predicted_sales"] = pred_values[: len(df_out)]
                            df_out["model_version"] = versions[: len(df_out)]

                            st.success(
                                f"Batch prediction finished in {json_response.get('processing_time', 0):.2f}s "
                                f"for {json_response.get('total_records', 0)} records."
                            )

                            stats_col1, stats_col2 = st.columns(2)
                            stats_col1.metric("Total predicted", f"${df_out['predicted_sales'].sum():,.2f}")
                            stats_col2.metric("Average predicted", f"${df_out['predicted_sales'].mean():,.2f}")

                            st.dataframe(df_out.head(20))

                            st.download_button(
                                "Download predictions CSV",
                                data=df_out.to_csv(index=False).encode("utf-8"),
                                file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                            )

# Footer
st.markdown("---")
st.caption("CAP 3764 - Advanced Data Science | FastAPI + Streamlit Prototype")
