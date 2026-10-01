"""
ForecastGPT -- Streamlit frontend.

This UI is a thin client: it never computes anything itself. Every number,
chart, and forecast shown here comes from calling the FastAPI backend
(which in turn only ever calls real pandas/statsmodels/xgboost/sklearn
code, or the LangGraph agent for the chat tab). This separation means you
could swap this Streamlit app for a React app later without touching any
analysis logic.
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="ForecastGPT", page_icon="📈", layout="wide")

if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None
if "dataset_meta" not in st.session_state:
    st.session_state.dataset_meta = None
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def api_get(path: str, **kwargs):
    return requests.get(f"{BACKEND_URL}{path}", timeout=60, **kwargs)


def api_post(path: str, **kwargs):
    return requests.post(f"{BACKEND_URL}{path}", timeout=120, **kwargs)


# ---------------------------------------------------------------------------
# Sidebar: upload + health
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📈 ForecastGPT")
    st.caption("Conversational time-series analysis & forecasting")

    try:
        health = api_get("/health").json()
        st.success("Backend: connected") if health["status"] == "ok" else st.error("Backend: unhealthy")
        st.write(f"LLM provider: `{health['llm_provider']}`")
        st.write("Qdrant: " + ("🟢 connected" if health["qdrant_connected"] else "🔴 not connected"))
    except Exception:
        st.error("Cannot reach backend. Is it running? (`docker compose up`)")

    st.divider()
    st.subheader("1. Upload your CSV")
    uploaded_file = st.file_uploader("Time-series CSV (needs a date column and a numeric value column)", type="csv")
    if uploaded_file and st.button("Upload & analyze", use_container_width=True):
        with st.spinner("Uploading and profiling dataset..."):
            resp = api_post("/upload", files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")})
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.dataset_id = data["dataset_id"]
            st.session_state.dataset_meta = data
            st.session_state.conversation_id = None
            st.session_state.chat_history = []
            st.success(f"Loaded dataset `{data['dataset_id']}` ({data['n_rows']} rows)")
        else:
            st.error(f"Upload failed: {resp.json().get('detail', resp.text)}")

    if st.session_state.dataset_id:
        st.info(f"Active dataset: `{st.session_state.dataset_id}`")


if not st.session_state.dataset_id:
    st.title("Welcome to ForecastGPT")
    st.write(
        "Upload a time-series CSV in the sidebar to get started. "
        "Then explore the tabs below or just chat with the agent."
    )
    st.stop()

dataset_id = st.session_state.dataset_id
meta = st.session_state.dataset_meta

tabs = st.tabs(
    ["Overview", "Time Series", "Forecast", "Model Comparison", "Anomalies", "Chat"]
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", meta["n_rows"])
    c2.metric("Start", meta["start_date"])
    c3.metric("End", meta["end_date"])
    c4.metric("Frequency", meta.get("inferred_frequency") or "irregular")
    st.write("Preview:")
    st.dataframe(pd.DataFrame(meta["preview"]), use_container_width=True)

    if st.button("Run full analysis (profile, trend, seasonality, anomalies)"):
        with st.spinner("Analyzing..."):
            resp = api_post("/analyze", json={"dataset_id": dataset_id})
        if resp.status_code == 200:
            result = resp.json()
            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Trend**")
                st.write(result["trend"]["summary"])
                st.markdown("**Seasonality**")
                st.write(result["seasonality"]["summary"])
            with colB:
                st.markdown("**Profile**")
                st.json(result["profile"])
                st.markdown("**Anomalies**")
                st.write(result["anomalies"]["summary"])
        else:
            st.error(resp.text)

# ---------------------------------------------------------------------------
# Time Series
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Time series visualization")
    df = pd.DataFrame(meta["preview"])
    resp = api_post("/analyze", json={"dataset_id": dataset_id, "checks": ["profile"]})
    full_resp = api_get(f"/dataset/{dataset_id}")
    # Re-fetch full series via forecast endpoint's history (cheap trick: horizon=1)
    fc_resp = api_post("/forecast", json={"dataset_id": dataset_id, "model": "naive", "horizon": 1})
    if fc_resp.status_code == 200:
        history = fc_resp.json()["history"]
        hist_df = pd.DataFrame(history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df["date"], y=hist_df["value"], mode="lines", name="value"))
        fig.update_layout(title="Full time series", xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(fc_resp.text)

# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Generate a forecast")
    colA, colB = st.columns(2)
    model = colA.selectbox("Model", ["sarima", "xgboost", "moving_average", "naive"])
    horizon = colB.number_input("Horizon (periods)", min_value=1, max_value=180, value=14)

    if st.button("Forecast", type="primary"):
        with st.spinner(f"Fitting {model} and forecasting {horizon} periods..."):
            resp = api_post(
                "/forecast", json={"dataset_id": dataset_id, "model": model, "horizon": int(horizon)}
            )
        if resp.status_code == 200:
            result = resp.json()
            hist_df = pd.DataFrame(result["history"])
            fc_df = pd.DataFrame(result["forecast"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_df["date"], y=hist_df["value"], mode="lines", name="History"))
            fig.add_trace(go.Scatter(x=fc_df["date"], y=fc_df["value"], mode="lines", name="Forecast", line=dict(dash="dash")))
            if fc_df["lower"].notna().all():
                fig.add_trace(
                    go.Scatter(
                        x=list(fc_df["date"]) + list(fc_df["date"])[::-1],
                        y=list(fc_df["upper"]) + list(fc_df["lower"])[::-1],
                        fill="toself",
                        fillcolor="rgba(99,110,250,0.15)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="95% interval",
                        showlegend=True,
                    )
                )
            fig.update_layout(title=f"{model} forecast ({horizon} periods)", xaxis_title="Date", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)

            if result.get("validation_metrics"):
                m = result["validation_metrics"]
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE", f"{m['mae']:.3f}")
                c2.metric("RMSE", f"{m['rmse']:.3f}")
                c3.metric("MAPE", f"{m['mape']:.2f}%" if m.get("mape") is not None else "n/a")

            with st.expander("Forecast table"):
                st.dataframe(fc_df, use_container_width=True)
        else:
            st.error(resp.text)

# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Compare all forecasting models")
    cmp_horizon = st.number_input("Validation horizon", min_value=1, max_value=90, value=14, key="cmp_horizon")
    if st.button("Compare models"):
        with st.spinner("Fitting naive, moving_average, sarima, and xgboost on a held-out window..."):
            resp = api_post(
                "/forecast",
                json={"dataset_id": dataset_id, "model": "sarima", "horizon": int(cmp_horizon), "compare_all": True},
            )
        if resp.status_code == 200:
            result = resp.json()
            comparison = result["all_model_comparison"]
            cmp_df = pd.DataFrame(comparison)
            fig = go.Figure(go.Bar(x=cmp_df["model"], y=cmp_df["rmse"]))
            fig.update_layout(title="Validation RMSE by model (lower is better)", xaxis_title="Model", yaxis_title="RMSE")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(cmp_df, use_container_width=True)
            best = cmp_df.iloc[0]["model"]
            st.success(f"Best model on this validation window: **{best}**")
        else:
            st.error(resp.text)

# ---------------------------------------------------------------------------
# Anomalies
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Anomaly detection")
    method = st.selectbox("Method", ["zscore", "isolation_forest"])
    if st.button("Detect anomalies"):
        with st.spinner("Detecting anomalies..."):
            resp = api_post("/analyze", json={"dataset_id": dataset_id, "checks": ["anomalies"]})
        if resp.status_code == 200:
            anomalies = resp.json()["anomalies"]
            fc_resp = api_post("/forecast", json={"dataset_id": dataset_id, "model": "naive", "horizon": 1})
            hist_df = pd.DataFrame(fc_resp.json()["history"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_df["date"], y=hist_df["value"], mode="lines", name="value"))
            if anomalies["anomalies"]:
                a_df = pd.DataFrame(anomalies["anomalies"])
                fig.add_trace(
                    go.Scatter(
                        x=a_df["date"], y=a_df["value"], mode="markers",
                        marker=dict(color="red", size=10, symbol="x"), name="anomaly",
                    )
                )
            fig.update_layout(title=f"Anomalies ({anomalies['method']})", xaxis_title="Date", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)
            st.write(anomalies["summary"])
            if anomalies["anomalies"]:
                st.dataframe(pd.DataFrame(anomalies["anomalies"]), use_container_width=True)
        else:
            st.error(resp.text)

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Chat with the ForecastGPT agent")
    st.caption(
        "Ask about trends, seasonality, anomalies, forecasts, model comparisons, or forecasting "
        "concepts in general. The agent calls real analysis tools -- it never makes up numbers."
    )

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
            if turn.get("tool_calls"):
                with st.expander("🔧 Tools the agent called"):
                    for tc in turn["tool_calls"]:
                        st.code(f"{tc['tool']}({tc['args']})", language="python")

    user_msg = st.chat_input("e.g. 'Is there seasonality in this data?' or 'Forecast the next 30 days'")
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = api_post(
                    "/chat",
                    json={
                        "dataset_id": dataset_id,
                        "message": user_msg,
                        "conversation_id": st.session_state.conversation_id,
                    },
                )
            if resp.status_code == 200:
                result = resp.json()
                st.session_state.conversation_id = result["conversation_id"]
                st.write(result["reply"])
                if result["tool_calls_made"]:
                    with st.expander("🔧 Tools the agent called"):
                        for tc in result["tool_calls_made"]:
                            st.code(f"{tc['tool']}({tc['args']})", language="python")
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": result["reply"], "tool_calls": result["tool_calls_made"]}
                )
            else:
                st.error(resp.text)
