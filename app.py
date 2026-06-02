"""
app.py — OSCM Cape Verde Ocean Data Dashboard
==============================================
A Streamlit application that fetches and visualises live oceanographic data
for the Ocean Science Centre Mindelo (OSCM) region via REST APIs.

Run locally:
    pip install streamlit plotly requests pandas numpy
    streamlit run app.py

Deploy free on Streamlit Community Cloud:
    1. Push this repo to GitHub
    2. Go to https://share.streamlit.io → connect repo → set app.py as entrypoint
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from data import fetch_marine, fetch_climate, fetch_argo, OSCM_LAT, OSCM_LON, BBOX

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OSCM Ocean Data Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Overall background */
  .stApp { background-color: #0b1d2e; color: #dce9f5; }
  section[data-testid="stSidebar"] { background-color: #0d2237; }

  /* Metric cards */
  div[data-testid="metric-container"] {
      background: linear-gradient(135deg, #112b44 0%, #0f2236 100%);
      border: 1px solid #1e4a6e;
      border-radius: 10px;
      padding: 14px 18px;
  }
  div[data-testid="metric-container"] label { color: #7db5d8 !important; font-size: 0.78rem; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
      color: #e8f4fd !important; font-size: 1.6rem; font-weight: 700;
  }

  /* Section headers */
  .section-header {
      font-size: 1.05rem;
      font-weight: 600;
      color: #7db5d8;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin: 1.4rem 0 0.5rem;
      padding-bottom: 4px;
      border-bottom: 1px solid #1e4a6e;
  }

  /* Status badge */
  .badge-live   { background:#0e4f2f; color:#4ade80; padding:3px 10px; border-radius:20px; font-size:0.75rem; }
  .badge-cached { background:#3a2e0e; color:#facc15; padding:3px 10px; border-radius:20px; font-size:0.75rem; }

  /* QC legend */
  .qc-good { color: #4ade80; font-weight:600; }
  .qc-bad  { color: #f87171; font-weight:600; }

  /* Caption text */
  .caption { font-size: 0.78rem; color: #5a8aa8; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme shared config ─────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0b1d2e",
    font=dict(color="#dce9f5", family="monospace", size=11),
    xaxis=dict(gridcolor="#1a3550", linecolor="#1a3550", zeroline=False),
    yaxis=dict(gridcolor="#1a3550", linecolor="#1a3550", zeroline=False),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e4a6e", borderwidth=1),
)
OCEAN_BLUE   = "#3b9edd"
WAVE_TEAL    = "#2ec4b6"
WARM_AMBER   = "#f4a261"
GOOD_GREEN   = "#4ade80"
BAD_RED      = "#f87171"


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🌊 OSCM Data Explorer")
    st.markdown(
        "**Ocean Science Centre Mindelo** · São Vicente, Cape Verde  \n"
        f"`{OSCM_LAT}°N, {abs(OSCM_LON)}°W`",
    )
    st.divider()

    days_back = st.slider("Time window (days)", min_value=7, max_value=90, value=30, step=7)

    st.divider()
    st.markdown("**Data sources**")
    st.markdown(
        "- [Open-Meteo Marine API](https://open-meteo.com/en/docs/marine-weather-api)  \n"
        "- [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api)  \n"
        "- [IFREMER ERDDAP (Argo)](https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.html)",
    )
    st.divider()
    st.markdown(
        "<div class='caption'>Built as a REST API portfolio project "
        "for a GEOMAR Data Scientist application. "
        "All data via open, unauthenticated APIs. "
        "QC follows Argo quality control conventions.</div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_all(days: int):
    marine,  marine_live  = fetch_marine(days)
    climate, climate_live = fetch_climate(days)
    argo,    argo_live    = fetch_argo(max(days, 60))  # Argo needs a wider window
    return marine, climate, argo, marine_live, climate_live, argo_live

with st.spinner("Fetching data from APIs…"):
    marine_df, climate_df, argo_df, marine_live, climate_live, argo_live = load_all(days_back)


# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════

col_title, col_badge, col_ts = st.columns([3, 1.5, 2])
with col_title:
    st.markdown("# OSCM Cape Verde · Ocean Data Explorer")
with col_badge:
    st.markdown("")
    def _badge(live: bool, name: str) -> str:
        cls = "badge-live" if live else "badge-cached"
        lbl = "LIVE" if live else "DEMO"
        return f"<span class='{cls}'>{name}: {lbl}</span>"
    st.markdown(
        _badge(marine_live,  "🌊 Marine") + "<br>" +
        _badge(climate_live, "🌡️ Climate") + "<br>" +
        _badge(argo_live,    "🔵 Argo"),
        unsafe_allow_html=True,
    )
with col_ts:
    st.markdown("")
    st.markdown(
        f"<div class='caption' style='text-align:right'>"
        f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}<br>"
        f"Window: last {days_back} days</div>",
        unsafe_allow_html=True,
    )

st.divider()


# ═════════════════════════════════════════════════════════════════════════════
# KPI METRICS ROW
# ═════════════════════════════════════════════════════════════════════════════

k1, k2, k3, k4, k5, k6 = st.columns(6)

mean_wh   = marine_df["wave_height"].mean()
max_wh    = marine_df["wave_height"].max()
mean_temp = climate_df["temperature_2m_max"].mean()
total_prec = climate_df["precipitation_sum"].sum()
n_floats  = argo_df["platform_number"].nunique() if len(argo_df) > 0 else 0
qc_pass   = (marine_df.get("wh_qc", pd.Series([1]*len(marine_df))) == 1).mean() * 100

k1.metric("Mean Wave Height", f"{mean_wh:.2f} m")
k2.metric("Max Wave Height",  f"{max_wh:.2f} m")
k3.metric("Mean Air Temp",    f"{mean_temp:.1f} °C")
k4.metric("Total Precip",     f"{total_prec:.1f} mm")
k5.metric("Argo Floats",      f"{n_floats}")
k6.metric("QC Pass Rate",     f"{qc_pass:.1f} %")

st.markdown("")


# ═════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

tab_marine, tab_climate, tab_argo, tab_qc, tab_about = st.tabs([
    "🌊 Marine Conditions",
    "🌡️ Climate",
    "🔵 Argo Floats",
    "✅ Quality Control",
    "ℹ️ About",
])


# ─── TAB 1: Marine ───────────────────────────────────────────────────────────

with tab_marine:
    st.markdown("<div class='section-header'>Wave Height — Hourly</div>", unsafe_allow_html=True)

    # Colour by QC flag
    wh_qc    = marine_df.get("wh_qc", pd.Series([1]*len(marine_df), index=marine_df.index))
    good_idx = wh_qc == 1
    bad_idx  = wh_qc != 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=marine_df.index[good_idx], y=marine_df["wave_height"][good_idx],
        mode="lines", name="QC Good", line=dict(color=OCEAN_BLUE, width=1.2),
    ))
    if bad_idx.any():
        fig.add_trace(go.Scatter(
            x=marine_df.index[bad_idx], y=marine_df["wave_height"][bad_idx],
            mode="markers", name="QC Suspect", marker=dict(color=BAD_RED, size=6, symbol="x"),
        ))
    # Rolling 24h mean
    rolling = marine_df["wave_height"].rolling(24).mean()
    fig.add_trace(go.Scatter(
        x=rolling.index, y=rolling,
        mode="lines", name="24h mean",
        line=dict(color=WARM_AMBER, width=2, dash="dash"),
    ))
    fig.update_layout(**PLOT_LAYOUT, height=280,
                      yaxis_title="m", xaxis_title=None,
                      title=dict(text="Significant Wave Height · QC-flagged spikes marked in red",
                                 font=dict(size=12)))
    st.plotly_chart(fig, width='stretch')

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("<div class='section-header'>Wave Period</div>", unsafe_allow_html=True)
        fig2 = go.Figure(go.Scatter(
            x=marine_df.index, y=marine_df["wave_period"],
            mode="lines", line=dict(color=WAVE_TEAL, width=1.2), fill="tozeroy",
            fillcolor="rgba(46,196,182,0.12)",
        ))
        fig2.update_layout(**PLOT_LAYOUT, height=220, yaxis_title="s",
                           title=dict(text="Peak Wave Period", font=dict(size=12)))
        st.plotly_chart(fig2, width='stretch')

    with col_r:
        st.markdown("<div class='section-header'>Wave Direction (polar)</div>", unsafe_allow_html=True)
        if "wave_direction" in marine_df.columns:
            daily_dir = marine_df["wave_direction"].resample("D").mean()
            fig3 = go.Figure(go.Barpolar(
                r=[1]*len(daily_dir),
                theta=daily_dir.values,
                width=8,
                marker_color=OCEAN_BLUE,
                marker_line_color="#0b1d2e",
                marker_line_width=0.5,
                opacity=0.8,
            ))
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0b1d2e",
                font=dict(color="#dce9f5", size=10),
                polar=dict(
                    bgcolor="#0b1d2e",
                    radialaxis=dict(visible=False),
                    angularaxis=dict(direction="clockwise", color="#5a8aa8"),
                ),
                margin=dict(l=10, r=10, t=40, b=10),
                height=220,
                title=dict(text="Daily Mean Wave Direction", font=dict(size=12)),
            )
            st.plotly_chart(fig3, width='stretch')

    st.markdown("<div class='section-header'>Wind Wave vs Swell</div>", unsafe_allow_html=True)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=marine_df.index, y=marine_df["wave_height"],
        mode="lines", name="Total wave height", line=dict(color=OCEAN_BLUE, width=1),
    ))
    if "wind_wave_height" in marine_df.columns:
        fig4.add_trace(go.Scatter(
            x=marine_df.index, y=marine_df["wind_wave_height"],
            mode="lines", name="Wind-wave component", line=dict(color=WARM_AMBER, width=1),
            fill="tozeroy", fillcolor="rgba(244,162,97,0.08)",
        ))
        swell = marine_df["wave_height"] - marine_df["wind_wave_height"].clip(lower=0)
        fig4.add_trace(go.Scatter(
            x=marine_df.index, y=swell.clip(lower=0),
            mode="lines", name="Swell estimate", line=dict(color=WAVE_TEAL, width=1),
        ))
    fig4.update_layout(**PLOT_LAYOUT, height=220, yaxis_title="m",
                       title=dict(text="Wave decomposition (Total / Wind-wave / Swell estimate)",
                                  font=dict(size=12)))
    st.plotly_chart(fig4, width='stretch')


# ─── TAB 2: Climate ──────────────────────────────────────────────────────────

with tab_climate:
    st.markdown("<div class='section-header'>Temperature Range</div>", unsafe_allow_html=True)

    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(
        x=climate_df.index, y=climate_df["temperature_2m_max"],
        mode="lines", name="Daily max", line=dict(color=WARM_AMBER, width=2),
    ))
    fig_t.add_trace(go.Scatter(
        x=climate_df.index, y=climate_df["temperature_2m_min"],
        mode="lines", name="Daily min", line=dict(color=OCEAN_BLUE, width=2),
        fill="tonexty", fillcolor="rgba(59,158,221,0.1)",
    ))
    fig_t.update_layout(**PLOT_LAYOUT, height=260, yaxis_title="°C",
                        title=dict(text="Air temperature 2m (min / max) · OSCM", font=dict(size=12)))
    st.plotly_chart(fig_t, width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='section-header'>Precipitation</div>", unsafe_allow_html=True)
        fig_p = go.Figure(go.Bar(
            x=climate_df.index, y=climate_df["precipitation_sum"],
            marker_color=OCEAN_BLUE, marker_line_width=0, opacity=0.85,
        ))
        fig_p.update_layout(**PLOT_LAYOUT, height=220, yaxis_title="mm",
                            title=dict(text="Daily precipitation", font=dict(size=12)))
        st.plotly_chart(fig_p, width='stretch')

    with col_b:
        st.markdown("<div class='section-header'>Max Wind Speed</div>", unsafe_allow_html=True)
        fig_w = go.Figure(go.Scatter(
            x=climate_df.index, y=climate_df["windspeed_10m_max"],
            mode="lines+markers", line=dict(color=WAVE_TEAL, width=1.5),
            marker=dict(size=4, color=WAVE_TEAL),
            fill="tozeroy", fillcolor="rgba(46,196,182,0.08)",
        ))
        fig_w.update_layout(**PLOT_LAYOUT, height=220, yaxis_title="km/h",
                            title=dict(text="Max wind speed 10m", font=dict(size=12)))
        st.plotly_chart(fig_w, width='stretch')

    # Correlation heatmap: wave height (daily mean) vs climate
    st.markdown("<div class='section-header'>Variable Correlation</div>", unsafe_allow_html=True)
    marine_daily = marine_df[["wave_height","wave_period"]].resample("D").mean()
    merged = marine_daily.join(climate_df[["temperature_2m_max","precipitation_sum","windspeed_10m_max"]], how="inner")
    corr   = merged.corr().round(2)
    labels = ["Wave ht", "Wave period", "Temp max", "Precip", "Wind max"]
    fig_c  = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels, text=corr.values.astype(str),
        texttemplate="%{text}", colorscale="RdBu", zmid=0,
        zmin=-1, zmax=1, showscale=True,
        colorbar=dict(tickfont=dict(color="#dce9f5")),
    ))
    fig_c.update_layout(**PLOT_LAYOUT, height=280,
                        title=dict(text="Pearson correlation — marine & climate variables (daily)", font=dict(size=12)))
    st.plotly_chart(fig_c, width='stretch')


# ─── TAB 3: Argo ─────────────────────────────────────────────────────────────

with tab_argo:
    if len(argo_df) == 0:
        st.info("No Argo profiles found for this region/period. Widen the time window in the sidebar.")
    else:
        n_prof = argo_df.groupby("platform_number")["time"].nunique().sum()
        st.markdown(
            f"**{argo_df['platform_number'].nunique()} floats** · "
            f"**{len(argo_df):,} depth levels** from last {days_back} days  "
            f"(bounding box: {BBOX['lat_min']}–{BBOX['lat_max']}°N, "
            f"{abs(BBOX['lon_max'])}–{abs(BBOX['lon_min'])}°W)"
        )

        col_map, col_prof = st.columns([1.2, 1])

        with col_map:
            st.markdown("<div class='section-header'>Float positions</div>", unsafe_allow_html=True)
            surface = argo_df[argo_df["pres"] <= 5].copy()
            fig_map = go.Figure()
            for fid in surface["platform_number"].unique():
                sub = surface[surface["platform_number"] == fid].sort_values("time")
                fig_map.add_trace(go.Scattergeo(
                    lat=sub["latitude"], lon=sub["longitude"],
                    mode="lines+markers",
                    name=f"Float {fid}",
                    marker=dict(size=7),
                    line=dict(width=1),
                ))
            # OSCM location marker
            fig_map.add_trace(go.Scattergeo(
                lat=[16.88], lon=[-24.99],
                mode="markers+text",
                name="OSCM",
                marker=dict(size=12, color="#f4a261", symbol="star"),
                text=["OSCM"], textposition="top right",
                textfont=dict(color="#f4a261"),
                showlegend=True,
            ))
            fig_map.update_geos(
                bgcolor="#0b1d2e",
                showland=True, landcolor="#112b44",
                showocean=True, oceancolor="#0b1d2e",
                showcoastlines=True, coastlinecolor="#1e4a6e",
                showframe=False,
                lonaxis_range=[BBOX["lon_min"]-3, BBOX["lon_max"]+3],
                lataxis_range=[BBOX["lat_min"]-3, BBOX["lat_max"]+3],
            )
            fig_map.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#dce9f5", size=10),
                margin=dict(l=0, r=0, t=0, b=0),
                height=340,
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
            )
            st.plotly_chart(fig_map, width='stretch')

        with col_prof:
            st.markdown("<div class='section-header'>T/S profiles</div>", unsafe_allow_html=True)
            # Latest profile per float
            # Get the timestamp of the latest profile for each float,
            # then filter the original DataFrame — avoids groupby/apply
            # collapsing platform_number into the index.
            latest_times = (
                argo_df.groupby("platform_number")["time"].max().reset_index()
                .rename(columns={"time": "latest_time"})
            )
            latest_profiles = (
                argo_df.merge(latest_times, on="platform_number")
                .query("time == latest_time")
                .drop(columns="latest_time")
                .sort_values("pres")
                .reset_index(drop=True)
            )

            fig_ts = make_subplots(rows=1, cols=2, shared_yaxes=True,
                                   subplot_titles=["Temperature (°C)", "Salinity (PSU)"])
            colors = px.colors.qualitative.Pastel
            for i, fid in enumerate(latest_profiles["platform_number"].unique()):
                sub = latest_profiles[latest_profiles["platform_number"] == fid]
                clr = colors[i % len(colors)]
                # Temperature
                fig_ts.add_trace(go.Scatter(
                    x=sub["temp"], y=sub["pres"],
                    mode="lines", name=f"Float {fid}",
                    line=dict(color=clr, width=1.5),
                    showlegend=True,
                ), row=1, col=1)
                # Salinity
                fig_ts.add_trace(go.Scatter(
                    x=sub["psal"], y=sub["pres"],
                    mode="lines", name=f"Float {fid}",
                    line=dict(color=clr, width=1.5),
                    showlegend=False,
                ), row=1, col=2)

            fig_ts.update_yaxes(autorange="reversed", title_text="Pressure (dbar)",
                                gridcolor="#1a3550", linecolor="#1a3550")
            fig_ts.update_xaxes(gridcolor="#1a3550", linecolor="#1a3550")
            fig_ts.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0b1d2e",
                font=dict(color="#dce9f5", size=10),
                margin=dict(l=10, r=10, t=30, b=10),
                height=340,
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
            )
            # Style subplot title colors
            for ann in fig_ts.layout.annotations:
                ann.font.color = "#7db5d8"
            st.plotly_chart(fig_ts, width='stretch')

        # T-S diagram
        st.markdown("<div class='section-header'>T–S Diagram</div>", unsafe_allow_html=True)
        argo_clean = argo_df.dropna(subset=["temp", "psal"])
        fig_ts_diag = go.Figure(go.Scatter(
            x=argo_clean["psal"], y=argo_clean["temp"],
            mode="markers",
            marker=dict(
                color=argo_clean["pres"], colorscale="Viridis_r",
                size=3, opacity=0.5, showscale=True,
                colorbar=dict(title="Pressure (dbar)", tickfont=dict(color="#dce9f5")),
            ),
            text=argo_clean["platform_number"].astype(str),
            hovertemplate="Float %{text}<br>T=%{y:.2f}°C  S=%{x:.3f}",
        ))
        fig_ts_diag.update_layout(
            **PLOT_LAYOUT, height=300,
            xaxis_title="Salinity (PSU)", yaxis_title="Temperature (°C)",
            title=dict(text="T–S diagram · all floats coloured by pressure depth",
                       font=dict(size=12)),
        )
        st.plotly_chart(fig_ts_diag, width='stretch')


# ─── TAB 4: Quality Control ──────────────────────────────────────────────────

with tab_qc:
    st.markdown(
        "Quality control follows conventions from the "
        "[Argo QC Manual](https://doi.org/10.13155/33951). "
        "Flag values: **1** = Good data · **4** = Bad data (suspect)."
    )

    col_q1, col_q2 = st.columns(2)

    with col_q1:
        st.markdown("<div class='section-header'>Wave height QC flags over time</div>",
                    unsafe_allow_html=True)
        if "wh_qc" in marine_df.columns:
            daily_qc = marine_df["wh_qc"].resample("D").apply(
                lambda x: (x == 4).sum() / len(x) * 100 if len(x) > 0 else 0
            )
            colors_qc = [BAD_RED if v > 5 else GOOD_GREEN for v in daily_qc]
            fig_qc1 = go.Figure(go.Bar(
                x=daily_qc.index, y=daily_qc.values,
                marker_color=colors_qc, marker_line_width=0,
            ))
            fig_qc1.update_layout(**PLOT_LAYOUT, height=220,
                                  yaxis_title="% flagged bad",
                                  title=dict(text="Daily % of bad QC flags (wave height)",
                                             font=dict(size=12)))
            st.plotly_chart(fig_qc1, width='stretch')

    with col_q2:
        st.markdown("<div class='section-header'>Argo T/S QC summary</div>",
                    unsafe_allow_html=True)
        if len(argo_df) > 0 and "temp_qc" in argo_df.columns:
            qc_summary = pd.DataFrame({
                "Variable":  ["Temperature", "Salinity"],
                "Good (1)":  [
                    (argo_df["temp_qc"] == 1).sum(),
                    (argo_df.get("psal_qc", pd.Series([1]*len(argo_df))) == 1).sum(),
                ],
                "Bad (4)": [
                    (argo_df["temp_qc"] == 4).sum(),
                    (argo_df.get("psal_qc", pd.Series([1]*len(argo_df))) == 4).sum(),
                ],
            })
            fig_qc2 = go.Figure()
            fig_qc2.add_trace(go.Bar(name="Good (1)", x=qc_summary["Variable"],
                                     y=qc_summary["Good (1)"], marker_color=GOOD_GREEN))
            fig_qc2.add_trace(go.Bar(name="Bad (4)",  x=qc_summary["Variable"],
                                     y=qc_summary["Bad (4)"],  marker_color=BAD_RED))
            fig_qc2.update_layout(**PLOT_LAYOUT, height=220, barmode="stack",
                                  yaxis_title="# observations",
                                  title=dict(text="Argo profile QC flag distribution",
                                             font=dict(size=12)))
            st.plotly_chart(fig_qc2, width='stretch')

    # QC table: worst days
    st.markdown("<div class='section-header'>Flagged observations detail</div>",
                unsafe_allow_html=True)
    if "wh_qc" in marine_df.columns:
        bad = marine_df[marine_df["wh_qc"] == 4][["wave_height","wave_period","wh_qc"]].copy()
        bad.index = bad.index.strftime("%Y-%m-%d %H:%M")
        bad.columns = ["Wave Height (m)", "Wave Period (s)", "QC Flag"]
        if len(bad) > 0:
            st.dataframe(
                bad.style
                   .highlight_max(subset=["Wave Height (m)"], color="#3a1a1a")
                   .format({"Wave Height (m)": "{:.2f}", "Wave Period (s)": "{:.1f}"}),
                height=180,
                width='stretch',
            )
        else:
            st.success("No bad QC flags in this time window.")


# ─── TAB 5: About ────────────────────────────────────────────────────────────

with tab_about:
    st.markdown("""
## About this dashboard

This application was built as a REST API skills demonstration for a
**Data Scientist position at [GEOMAR Helmholtz Centre for Ocean Research Kiel](https://www.geomar.de)**.

### What it demonstrates

| Skill | Implementation |
|-------|---------------|
| **REST API consumption** | Three separate APIs queried with `requests`; JSON parsed into DataFrames |
| **Error handling & retry** | Exponential back-off, graceful fallback to synthetic data |
| **Data harmonisation** | Hourly marine data resampled to daily; merged with climate and Argo data |
| **QC / flagging** | Range checks and spike detection following Argo QC conventions (flags 1 & 4) |
| **Interactive visualisation** | Plotly charts: time series, polar plots, profiles, T-S diagram, map |
| **Caching** | `st.cache_data` with 1-hour TTL for efficient repeated loading |
| **FAIR principles** | All data sourced from open, documented, citable APIs |

### Data sources

| Source | Variables | Update frequency |
|--------|-----------|-----------------|
| [Open-Meteo Marine API](https://open-meteo.com/en/docs/marine-weather-api) | Wave height, direction, period | Hourly, NRT |
| [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api) | Air temp, precipitation, wind | Daily |
| [IFREMER ERDDAP — Argo](https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.html) | T/S profiles at depth | As floats surface (~10 days) |

### Study region
The **Ocean Science Centre Mindelo (OSCM)** at São Vicente, Cape Verde (16.88°N, 24.99°W)
is GEOMAR's primary West African research site. This dashboard covers the surrounding
Atlantic region (12–20°N, 20–27°W), the area relevant to the Ocean Centre of Excellence
for West Africa (Ocean-CofE) project.

### Code
Full source code and documentation: see the GitHub repository linked from this deployment.
    """)
