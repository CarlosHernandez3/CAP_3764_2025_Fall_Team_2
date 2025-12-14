import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Run the app: 
# 1. Open another terminal
# 2. Activate env (run in terminal: conda activate store_sale_prediction_env)
# 3. Set your working directory to the ROOT folder (run in terminal: cd your_path/CAP_3764_2025_Fall_Team_2)
# 4. Execute app.py (run in terminal: streamlit run src/app.py)

API_BASE = "http://localhost:8000"

st.title("Store Sales Prediction System")
st.markdown("*Advanced Data Science - CAP 3764*")

tab_predict, tab_batch = st.tabs(
    ["Single Prediction", "Batch Predict"]
)

# -------------------------------------------------
# Tab 1: Single Store Sales Prediction
# -------------------------------------------------
with tab_predict:
    st.subheader("Single Store Sales Prediction")
    
    st.markdown("Enter the store and temporal features to predict sales:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Store Information**")
        store_nbr = st.number_input("Store Number", min_value=1, max_value=100, value=1)
        store_type = st.selectbox("Store Type", ["A", "B", "C", "D", "E"])
        cluster = st.number_input("Store Cluster", min_value=1, max_value=20, value=1)
    
    with col2:
        st.markdown("**Temporal Features**")
        date = st.date_input("Date", value=datetime.now())
        day_of_week = st.selectbox("Day of Week", 
                                   ["Monday", "Tuesday", "Wednesday", "Thursday", 
                                    "Friday", "Saturday", "Sunday"])
        is_weekend = st.checkbox("Is Weekend?", value=(day_of_week in ["Saturday", "Sunday"]))
        is_holiday = st.checkbox("Is Holiday?")
    
    with col3:
        st.markdown("**Promotion & Product**")
        onpromotion = st.number_input("Items on Promotion", min_value=0, max_value=1000, value=0)
        family = st.selectbox("Product Family", 
                             ["AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", 
                              "BOOKS", "BREAD/BAKERY", "CLEANING", "DAIRY", 
                              "DELI", "EGGS", "FROZEN FOODS", "GROCERY I", 
                              "GROCERY II", "HARDWARE", "HOME AND KITCHEN I"])
        oil_price = st.number_input("Oil Price (USD)", min_value=0.0, max_value=200.0, value=50.0)

    if st.button("Predict Sales", type="primary"):
        # Map day of week to number
        day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
                      "Friday": 4, "Saturday": 5, "Sunday": 6}
        
        payload = {
            "store_nbr": int(store_nbr),
            "family": family,
            "onpromotion": int(onpromotion),
            "store_type": store_type,
            "cluster": int(cluster),
            "date": str(date),
            "day_of_week": day_mapping[day_of_week],
            "is_weekend": int(is_weekend),
            "is_holiday": int(is_holiday),
            "oil_price": float(oil_price)
        }
        
        try:
            with st.spinner("Predicting sales..."):
                r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
                r.raise_for_status()
                result = r.json()
                
                predicted_sales = result.get("predicted_sales", 0.0)
                confidence = result.get("confidence", 0.0)
                
                st.success("Prediction completed!")
                
                # Display results in metrics
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Predicted Sales", f"${predicted_sales:,.2f}")
                with metric_col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                
                # Additional insights
                with st.expander("Prediction Details"):
                    st.json(payload)
                    
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure the FastAPI server is running on http://localhost:8000")
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")


# -------------------------------------------------
# Tab 2: Batch Prediction
# -------------------------------------------------
with tab_batch:
    st.subheader("Batch Sales Prediction from CSV")

    st.markdown("""
    Upload a CSV file with the following columns:
    - **store_nbr**: Store number (integer)
    - **family**: Product family (string)
    - **onpromotion**: Number of items on promotion (integer)
    - **store_type**: Type of store (A, B, C, D, or E)
    - **cluster**: Store cluster (integer)
    - **date**: Date (YYYY-MM-DD format)
    - **day_of_week**: Day of week (0=Monday, 6=Sunday)
    - **is_weekend**: Weekend indicator (0 or 1)
    - **is_holiday**: Holiday indicator (0 or 1)
    - **oil_price**: Oil price in USD (float)
    """)

    # Sample data download
    if st.button("Download Sample CSV Template"):
        sample_data = pd.DataFrame({
            'store_nbr': [1, 2, 3],
            'family': ['GROCERY I', 'BEVERAGES', 'DAIRY'],
            'onpromotion': [10, 5, 15],
            'store_type': ['A', 'B', 'C'],
            'cluster': [1, 2, 3],
            'date': ['2024-01-01', '2024-01-01', '2024-01-01'],
            'day_of_week': [0, 0, 0],
            'is_weekend': [0, 0, 0],
            'is_holiday': [1, 1, 1],
            'oil_price': [50.5, 50.5, 50.5]
        })
        csv_sample = sample_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Template",
            data=csv_sample,
            file_name="sales_prediction_template.csv",
            mime="text/csv"
        )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None

        required_cols = [
            "store_nbr", "family", "onpromotion", "store_type", "cluster",
            "date", "day_of_week", "is_weekend", "is_holiday", "oil_price"
        ]

        if df is not None:
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                st.success(f"File uploaded successfully! {len(df)} rows detected.")
                st.write("**Preview of uploaded data:**")
                st.dataframe(df.head(10))

                if st.button("Run Batch Prediction", type="primary"):
                    records = df[required_cols].to_dict(orient="records")
                    
                    # Convert to proper types
                    items = []
                    for r in records:
                        try:
                            items.append({
                                "store_nbr": int(r["store_nbr"]),
                                "family": str(r["family"]),
                                "onpromotion": int(r["onpromotion"]),
                                "store_type": str(r["store_type"]),
                                "cluster": int(r["cluster"]),
                                "date": str(r["date"]),
                                "day_of_week": int(r["day_of_week"]),
                                "is_weekend": int(r["is_weekend"]),
                                "is_holiday": int(r["is_holiday"]),
                                "oil_price": float(r["oil_price"])
                            })
                        except (ValueError, KeyError) as e:
                            st.warning(f"Skipping row due to data error: {e}")
                            continue

                    if not items:
                        st.error("No valid rows to process.")
                    else:
                        try:
                            with st.spinner(f"Processing {len(items)} predictions..."):
                                r = requests.post(
                                    f"{API_BASE}/batch_predict", 
                                    json=items, 
                                    timeout=120
                                )
                                r.raise_for_status()
                                result = r.json()
                                preds = result.get("predictions", [])
                                
                                if not preds:
                                    st.warning("No predictions returned.")
                                else:
                                    # Add predictions to dataframe
                                    predicted_sales = [p["predicted_sales"] for p in preds]
                                    confidence = [p.get("confidence", 0.0) for p in preds]
                                    
                                    df_out = df.copy()
                                    df_out["predicted_sales"] = predicted_sales[:len(df_out)]
                                    df_out["confidence"] = confidence[:len(df_out)]
                                    
                                    st.success(f"Batch predictions completed! Processed {len(preds)} rows.")
                                    
                                    # Summary statistics
                                    st.markdown("Summary Statistics")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Total Predicted Sales", f"${df_out['predicted_sales'].sum():,.2f}")
                                    with col2:
                                        st.metric("Average Sales", f"${df_out['predicted_sales'].mean():,.2f}")
                                    with col3:
                                        st.metric("Avg Confidence", f"{df_out['confidence'].mean():.1%}")
                                    
                                    st.markdown("Results Preview")
                                    st.dataframe(df_out.head(20))

                                    # Download results
                                    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                                    st.download_button(
                                        "Download Full Results as CSV",
                                        data=csv_bytes,
                                        file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                    )
                        except requests.exceptions.ConnectionError:
                            st.error("Cannot connect to API. Make sure the FastAPI server is running on http://localhost:8000")
                        except requests.exceptions.RequestException as e:
                            st.error(f"Batch prediction request failed: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><b>Store Sales Prediction System</b> | CAP 3764 - Advanced Data Science</p>
    <p>Team: Luis D. Jimenez & Carlos Hernandez | Florida International University</p>
</div>
""", unsafe_allow_html=True)