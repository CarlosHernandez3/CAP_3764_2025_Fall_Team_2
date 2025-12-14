import streamlit as st
import requests
import pandas as pd
from datetime import datetime

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Store Sales Prediction", layout="wide")
st.title("Store Sales Prediction System")
st.markdown("*CAP 3764 – Advanced Data Science*")

FEATURE_COLUMNS = [
    "Sales_lag14",
    "sqrt_Sales_lag14",
    "Sales_lag1",
    "Sales_lag28",
    "sqrt_Sales_lag28",
    "Promo",
    "Sales_lag7",
    "sqrt_Sales_lag7",
    "DayOfWeek",
    "Sales_lag365",
    "Customers_lag1",
    "Customers_lag7",
    "sqrt_Customers_lag7",
    "Customers_lag365",
    "Customers_lag28",
]

tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction"])

# =========================================================
# Single Prediction
# =========================================================
with tab_single:
    st.subheader("Single Sales Prediction (Lag-Based Features)")

    st.markdown(
        """
        Enter **engineered lag features** exactly as used during model training.
        """
    )

    cols = st.columns(3)
    inputs = {}

    for i, feature in enumerate(FEATURE_COLUMNS):
        with cols[i % 3]:
            if feature == "DayOfWeek":
                inputs[feature] = st.selectbox(
                    "DayOfWeek (0=Mon)",
                    options=list(range(7)),
                    index=0,
                )
            elif feature == "Promo":
                inputs[feature] = st.selectbox("Promo", [0.0, 1.0])
            else:
                inputs[feature] = st.number_input(feature, value=0.0)

    if st.button("Predict Sales", type="primary"):
        try:
            with st.spinner("Running prediction..."):
                r = requests.post(
                    f"{API_BASE}/predict",
                    json=inputs,
                    timeout=10,
                )
                r.raise_for_status()
                result = r.json()

            st.success("Prediction completed")

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Sales", f"${result['predicted_sales']:,.2f}")
            col2.metric("Confidence", f"{result['confidence']:.1%}")
            col3.metric("Model Version", result["model_version"])

            with st.expander("Submitted Payload"):
                st.json(inputs)

        except requests.exceptions.ConnectionError:
            st.error("FastAPI server not running on localhost:8000")
        except requests.exceptions.RequestException as e:
            st.error(f"Prediction failed: {e}")

# =========================================================
# Batch Prediction
# =========================================================
with tab_batch:
    st.subheader("Batch Prediction via CSV")

    st.markdown(
        """
        Upload a CSV containing **only the following columns**:
        """
    )
    st.code(", ".join(FEATURE_COLUMNS))

    if st.button("Download CSV Template"):
        template = pd.DataFrame({c: [0.0] for c in FEATURE_COLUMNS})
        st.download_button(
            "Download Template",
            data=template.to_csv(index=False),
            file_name="sales_lag_feature_template.csv",
            mime="text/csv",
        )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None

        if df is not None:
            missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                st.success(f"{len(df)} rows loaded")
                st.dataframe(df.head())

                if st.button("Run Batch Prediction", type="primary"):
                    records = df[FEATURE_COLUMNS].to_dict(orient="records")

                    try:
                        with st.spinner("Processing batch predictions..."):
                            r = requests.post(
                                f"{API_BASE}/batch_predict",
                                json=records,
                                timeout=120,
                            )
                            r.raise_for_status()
                            result = r.json()

                        preds = result["predictions"]

                        df_out = df.copy()
                        df_out["predicted_sales"] = [p["predicted_sales"] for p in preds]
                        df_out["confidence"] = [p["confidence"] for p in preds]

                        st.success(
                            f"Completed {result['total_records']} predictions "
                            f"in {result['processing_time']:.2f}s"
                        )

                        st.dataframe(df_out.head())

                        st.download_button(
                            "Download Results",
                            data=df_out.to_csv(index=False),
                            file_name=f"batch_predictions_{datetime.now():%Y%m%d_%H%M%S}.csv",
                            mime="text/csv",
                        )

                    except requests.exceptions.ConnectionError:
                        st.error("FastAPI server not running")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Batch prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center">
        <b>Store Sales Prediction System</b><br>
        CAP 3764 – Florida International University<br>
        Team: Luis D. Jimenez & Carlos Hernandez
    </div>
    """,
    unsafe_allow_html=True,
)
