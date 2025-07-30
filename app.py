import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# === Page Setup ===
st.set_page_config(page_title="EV Forecast", layout="wide")

# === Theme Toggle Button ===
dark_mode = st.toggle("☀️/🌙", value=True)

# Smart contrast handling
bg_color = "#1c1c1c" if dark_mode else "#ffffff"
text_color = "#f0f0f0" if dark_mode else "#111111"

# === Load model ===
model = joblib.load("forecasting_ev_model.pkl")

# === Styling ===
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .css-18e3th9, .css-1d391kg {{
            color: {text_color};
        }}
        .css-10trblm, .css-15zrgzn {{
            color: {text_color};
        }}
    </style>
""", unsafe_allow_html=True)

# === Title and Intro ===
st.markdown(f"""
    <div style='text-align: center; font-size: 36px; font-weight: bold; color: {text_color}; margin-top: 20px;'>
        🔮 EV Adoption Forecaster for a County in Washington State
    </div>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div style='text-align: center; font-size: 22px; font-weight: bold; padding-top: 10px; margin-bottom: 25px; color: {text_color};'>
        Welcome to the Electric Vehicle (EV) Adoption Forecast tool.
    </div>
""", unsafe_allow_html=True)

st.image("ev-car-factory.jpg", use_container_width=True)

st.markdown(f"""
    <div style='text-align: left; font-size: 22px; padding-top: 10px; color: {text_color};'>
        Select a county and see the forecasted EV adoption trend for the next 3 years.
    </div>
""", unsafe_allow_html=True)

# === Load Data ===
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

county_list = sorted(df["County"].dropna().unique().tolist())
county = st.selectbox("Select a County", county_list)

if county not in df["County"].unique():
    st.warning(f"County '{county}' not found in dataset.")
    st.stop()

county_df = df[df["County"] == county].sort_values("Date")
county_code = county_df["county_encoded"].iloc[0]

# === Forecasting ===
historical_ev = list(county_df["Electric Vehicle (EV) Total"].values[-6:])
cumulative_ev = list(np.cumsum(historical_ev))
months_since_start = county_df["months_since_start"].max()
latest_date = county_df["Date"].max()

future_rows = []
forecast_horizon = 36

for i in range(1, forecast_horizon + 1):
    forecast_date = latest_date + pd.DateOffset(months=i)
    months_since_start += 1
    lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
    roll_mean = np.mean([lag1, lag2, lag3])
    pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
    pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
    ev_growth_slope = np.polyfit(range(6), cumulative_ev[-6:], 1)[0]

    new_row = {
        "months_since_start": months_since_start,
        "county_encoded": county_code,
        "ev_total_lag1": lag1,
        "ev_total_lag2": lag2,
        "ev_total_lag3": lag3,
        "ev_total_roll_mean_3": roll_mean,
        "ev_total_pct_change_1": pct_change_1,
        "ev_total_pct_change_3": pct_change_3,
        "ev_growth_slope": ev_growth_slope,
    }

    pred = model.predict(pd.DataFrame([new_row]))[0]
    future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})
    historical_ev.append(pred)
    historical_ev = historical_ev[-6:]
    cumulative_ev.append(cumulative_ev[-1] + pred)
    cumulative_ev = cumulative_ev[-6:]

# === Combine Historical & Forecast Data ===
historical_cum = county_df[["Date", "Electric Vehicle (EV) Total"]].copy()
historical_cum["Source"] = "Historical"
historical_cum["Cumulative EV"] = historical_cum["Electric Vehicle (EV) Total"].cumsum()

forecast_df = pd.DataFrame(future_rows)
forecast_df["Source"] = "Forecast"
forecast_df["Cumulative EV"] = forecast_df["Predicted EV Total"].cumsum() + historical_cum["Cumulative EV"].iloc[-1]

combined = pd.concat(
    [
        historical_cum[["Date", "Cumulative EV", "Source"]],
        forecast_df[["Date", "Cumulative EV", "Source"]],
    ],
    ignore_index=True,
)

# === Plotting ===
st.subheader(f"📊 Cumulative EV Forecast for {county} County")

fig = px.line(
    combined,
    x="Date",
    y="Cumulative EV",
    color="Source",
    title=f"Cumulative EV Trend - {county} (3 Years Forecast)",
    markers=True,
    template="plotly_dark" if dark_mode else "plotly_white",
)
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# === CSV Download ===
st.download_button(
    "📥 Download Forecast CSV",
    forecast_df.to_csv(index=False),
    file_name=f"{county}_ev_forecast.csv",
    mime="text/csv",
)

# === Growth Summary ===
historical_total = historical_cum["Cumulative EV"].iloc[-1]
forecasted_total = forecast_df["Cumulative EV"].iloc[-1]

if historical_total > 0:
    growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
    trend = "increase 📈" if growth_pct > 0 else "decrease 📉"
    st.success(
        f"EV adoption in **{county}** is expected to show a **{trend} of {growth_pct:.2f}%** over the next 3 years."
    )
else:
    st.warning("Historical EV total is zero; percentage change can't be calculated.")

# === Multi-County Comparison ===
st.markdown("---")
st.header("Compare EV Adoption Trends for up to 4 Counties")

multi_counties = st.multiselect(
    "Select up to 4 counties to compare", county_list, max_selections=4
)

if multi_counties:
    comp_data = []

    for cty in multi_counties:
        cty_df = df[df["County"] == cty].sort_values("Date")
        cty_code = cty_df["county_encoded"].iloc[0]
        hist_ev = list(cty_df["Electric Vehicle (EV) Total"].values[-6:])
        cum_ev = list(np.cumsum(hist_ev))
        months_since = cty_df["months_since_start"].max()
        last_date = cty_df["Date"].max()
        rows = []

        for i in range(1, forecast_horizon + 1):
            forecast_date = last_date + pd.DateOffset(months=i)
            months_since += 1
            lag1, lag2, lag3 = hist_ev[-1], hist_ev[-2], hist_ev[-3]
            roll_mean = np.mean([lag1, lag2, lag3])
            pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
            pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
            ev_slope = np.polyfit(range(6), cum_ev[-6:], 1)[0]

            new_row = {
                "months_since_start": months_since,
                "county_encoded": cty_code,
                "ev_total_lag1": lag1,
                "ev_total_lag2": lag2,
                "ev_total_lag3": lag3,
                "ev_total_roll_mean_3": roll_mean,
                "ev_total_pct_change_1": pct_change_1,
                "ev_total_pct_change_3": pct_change_3,
                "ev_growth_slope": ev_slope,
            }

            pred = model.predict(pd.DataFrame([new_row]))[0]
            rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})
            hist_ev.append(pred)
            hist_ev = hist_ev[-6:]
            cum_ev.append(cum_ev[-1] + pred)
            cum_ev = cum_ev[-6:]

        hist_cum = cty_df[["Date", "Electric Vehicle (EV) Total"]].copy()
        hist_cum["Cumulative EV"] = hist_cum["Electric Vehicle (EV) Total"].cumsum()
        fc_df = pd.DataFrame(rows)
        fc_df["Cumulative EV"] = fc_df["Predicted EV Total"].cumsum() + hist_cum["Cumulative EV"].iloc[-1]

        combined_cty = pd.concat(
            [hist_cum[["Date", "Cumulative EV"]], fc_df[["Date", "Cumulative EV"]]],
            ignore_index=True,
        )
        combined_cty["County"] = cty
        comp_data.append(combined_cty)

    final_df = pd.concat(comp_data, ignore_index=True)

    st.subheader("📈 Comparison of Cumulative EV Adoption Trends")

    fig = px.line(
        final_df,
        x="Date",
        y="Cumulative EV",
        color="County",
        title="EV Adoption Trends: Historical + 3-Year Forecast",
        template="plotly_dark" if dark_mode else "plotly_white",
        markers=True,
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    growth_texts = []
    for cty in multi_counties:
        cty_df = final_df[final_df["County"] == cty].reset_index(drop=True)
        historical = cty_df["Cumulative EV"].iloc[-forecast_horizon - 1]
        forecasted = cty_df["Cumulative EV"].iloc[-1]
        if historical > 0:
            growth_pct = ((forecasted - historical) / historical) * 100
            growth_texts.append(f"{cty}: {growth_pct:.2f}%")
        else:
            growth_texts.append(f"{cty}: N/A")

    st.success("Forecasted growth over 3 years — " + " | ".join(growth_texts))

# === Footer ===
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; font-size: 15px; color: gray;'>
    Developed by <strong>Micheal</strong> • AICTE Internship Cycle 2 • 2025
</div>
""", unsafe_allow_html=True)
