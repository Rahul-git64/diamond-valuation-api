"""Streamlit interactive gemological diamond valuation & batch portfolio console."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Diamond Valuation API",
    page_icon="💎",
    layout="wide",
)

st.title("💎 GIA Gemological Diamond Valuation & Batch Portfolio Console")
st.markdown(
    "Enforces strict **GIA ordinal hierarchies** (`Fair -> Ideal`, `J -> D`, `I1 -> IF`), "
    "physical geometry validation (`volume_mm3`, `table_to_depth_ratio`), and **KS/PSI Drift Supervision**."
)

col1, col2, col3 = st.columns(3)

with col1:
    carat = st.slider("Carat Weight", 0.20, 5.00, 1.15, 0.05)
    cut = st.selectbox("GIA Cut Grade", ["Ideal", "Premium", "Very Good", "Good", "Fair"])
    color = st.selectbox("GIA Color Grade (D = Colorless, J = Tinted)", ["D", "E", "F", "G", "H", "I", "J"])

with col2:
    clarity = st.selectbox("GIA Clarity Grade", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])
    depth = st.number_input("Total Depth (%)", min_value=50.0, max_value=75.0, value=61.8, step=0.2)
    table = st.number_input("Table Width (%)", min_value=50.0, max_value=75.0, value=57.0, step=0.5)

with col3:
    x = st.number_input("Length x (mm)", min_value=3.0, max_value=15.0, value=6.72, step=0.05)
    y = st.number_input("Width y (mm)", min_value=3.0, max_value=15.0, value=6.75, step=0.05)
    z = st.number_input("Depth z (mm)", min_value=2.0, max_value=10.0, value=4.16, step=0.05)

if st.button("Appraise Diamond", type="primary"):
    payload = {
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z,
    }
    try:
        resp = httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)
        resp.raise_for_status()
        res = resp.json()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Estimated Value (USD)", f"${res['predicted_price_usd']:,.2f}")
        m2.metric("Price / Carat", f"${res['price_per_carat_usd']:,.2f}/ct")
        m3.metric("Physical Volume", f"{res['volume_mm3']:.1f} mm³")
        m4.metric("GIA Summary", res["gia_grade_summary"])
    except Exception as exc:
        st.error(f"Valuation request failed: {exc}")

st.divider()
if st.button("Inspect Carat & GIA Grade Drift (/monitor/drift)"):
    try:
        drift_resp = httpx.get(f"{API_URL}/monitor/drift", timeout=10.0)
        st.json(drift_resp.json())
    except Exception as exc:
        st.error(f"Drift telemetry request failed: {exc}")
