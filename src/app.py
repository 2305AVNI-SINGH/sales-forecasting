"""Streamlit dashboard: actual vs predicted sales for any Rossmann store.

Run `python src/train.py --data_dir data` first (creates outputs/lgbm_final.pkl),
then start the dashboard with:
    streamlit run app.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from features import FEATURES, add_features, clean, load_data, make_model_df  # noqa: E402

DATA_DIR = "data"
MODEL_PATH = "outputs/lgbm_final.pkl"
RESULTS_PATH = "outputs/results.csv"
SPLIT_DATE = "2015-06-19"

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
st.title("Sales Forecasting Dashboard")
st.caption("7-day-ahead daily sales forecast for Rossmann stores (LightGBM)")


@st.cache_data(show_spinner="Preparing data and predictions...")
def get_predictions():
    """Build features, predict on the validation window and return the results."""
    model = joblib.load(MODEL_PATH)
    df = add_features(clean(load_data(DATA_DIR)))
    valid = make_model_df(df)
    valid = valid[valid["Date"] >= SPLIT_DATE].copy()
    valid["Predicted"] = np.expm1(model.predict(valid[FEATURES]))
    return valid


try:
    valid = get_predictions()
except FileNotFoundError:
    st.error("Model or data not found. Run `python src/train.py --data_dir data` first.")
    st.stop()

# Sidebar: choose a store
stores = sorted(valid["Store"].unique())
store_id = st.sidebar.selectbox("Select a store", stores)
store_df = valid[valid["Store"] == store_id].set_index("Date").sort_index()

# Metrics for the selected store
mae = (store_df["Sales"] - store_df["Predicted"]).abs().mean()
nonzero = store_df[store_df["Sales"] > 0]  # avoid dividing by zero in MAPE
mape = ((nonzero["Sales"] - nonzero["Predicted"]).abs() / nonzero["Sales"]).mean() * 100
c1, c2, c3 = st.columns(3)
c1.metric("Store", int(store_id))
c2.metric("MAE (this store)", f"{mae:,.0f}")
c3.metric("MAPE (this store)", f"{mape:.1f}%")

# Actual vs predicted for the selected store
st.subheader(f"Store {store_id}: actual vs predicted sales")
st.line_chart(store_df[["Sales", "Predicted"]])

# Total sales across all stores
st.subheader("All stores: actual vs predicted total sales")
total = valid.groupby("Date")[["Sales", "Predicted"]].sum()
st.line_chart(total)

# Overall model comparison table
if os.path.exists(RESULTS_PATH):
    st.subheader("Model comparison (validation window)")
    st.dataframe(pd.read_csv(RESULTS_PATH, index_col=0))
