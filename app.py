"""
app.py — OSCM Cape Verde Ocean Data Dashboard
==============================================
A Streamlit application that fetches and visualises live oceanographic data
for the Ocean Science Centre Mindelo (OSCM) region via REST APIs.

To run locally:
    pip install streamlit plotly requests pandas numpy
    streamlit run app.py
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
        "**Ocean Science Centre Mindelo** \n" 
        "São Vicente, Cape Verde  \n"
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
        "for DTO-Ocean_CofE Data Scientist application at GEOMAR. "
        "All data via open, unauthenticated APIs. "
        "QC follows Argo quality control conventions.</div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_all(days: int, version: int = 2):
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
    st.markdown("# OSCM Data Explorer")
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
mean_sst    = marine_df["sea_surface_temperature"].mean() if "sea_surface_temperature" in marine_df.columns else None
mean_temp = climate_df["temperature_2m_max"].mean()
total_prec = climate_df["precipitation_sum"].sum()
n_floats  = argo_df["platform_number"].nunique()
qc_pass   = (marine_df.get("wh_qc", pd.Series([1]*len(marine_df))) == 1).mean() * 100

k1.metric("Mean Wave Height", f"{mean_wh:.2f} m")
k2.metric("Mean SST",  f"{mean_sst:.1f} °C" if mean_sst is not None else "N/A")
k3.metric("Mean Air Temp",    f"{mean_temp:.1f} °C")
k4.metric("Total Precip",     f"{total_prec:.1f} mm")
k5.metric("Argo Floats",      f"{n_floats}")
k6.metric("QC Pass Rate",     f"{qc_pass:.1f} %")

st.markdown("")


# ═════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

tab_marine, tab_climate, tab_argo, tab_qc, tab_anomaly, tab_about = st.tabs([
    "🌊 Marine Conditions",
    "🌡️ Climate",
    "🔵 Argo Floats",
    "✅ Quality Control",
    "🤖 Anomaly Detection",
    "ℹ️ About",
])


# ─── TAB 1: Marine ───────────────────────────────────────────────────────────

with tab_marine:
    # SST - Hourly plot
    st.markdown("<div class='section-header'>Sea Surface Temperature — Hourly</div>", unsafe_allow_html=True)

    if "sea_surface_temperature" in marine_df.columns:
        # Colour by QC flag
        sst_qc = marine_df["sst_qc"].fillna(1)
        good_idx = sst_qc == 1
        bad_idx = sst_qc != 1

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=marine_df.index[good_idx], y=marine_df["sea_surface_temperature"][good_idx],
            mode="lines", name="QC Good", line=dict(color=OCEAN_BLUE, width=1.2),
        ))
        if bad_idx.any():
            fig1.add_trace(go.Scatter(
                x=marine_df.index[bad_idx], y=marine_df["sea_surface_temperature"][bad_idx],
                mode="markers", name="QC Suspect", marker=dict(color=BAD_RED, size=6, symbol="x"),
            ))
        # Rolling 24h mean
        rolling = marine_df["sea_surface_temperature"].rolling(24).mean()
        fig1.add_trace(go.Scatter(
            x=rolling.index, y=rolling,
            mode="lines", name="24h mean",
            line=dict(color=WARM_AMBER, width=2, dash="dash"),
        ))
        fig1.update_layout(**PLOT_LAYOUT, height=280,
                           yaxis_title="°C", xaxis_title=None,
                           title=dict(text="Water Temperature Near the Sea Surface · QC-flagged spikes marked in red",
                                      font=dict(size=12)))
        st.plotly_chart(fig1, width='stretch')

    else:
        st.info("Sea surface temperature not available for this time window.")

    # Wave Height - Hourly plot
    st.markdown("<div class='section-header'>Wave Height — Hourly</div>", unsafe_allow_html=True)

    # Colour by QC flag
    wh_qc    = marine_df.get("wh_qc", pd.Series([1]*len(marine_df), index=marine_df.index))
    good_idx = wh_qc == 1
    bad_idx  = wh_qc != 1

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=marine_df.index[good_idx], y=marine_df["wave_height"][good_idx],
        mode="lines", name="QC Good", line=dict(color=OCEAN_BLUE, width=1.2),
    ))
    if bad_idx.any():
        fig2.add_trace(go.Scatter(
            x=marine_df.index[bad_idx], y=marine_df["wave_height"][bad_idx],
            mode="markers", name="QC Suspect", marker=dict(color=BAD_RED, size=6, symbol="x"),
        ))
    # Rolling 24h mean
    rolling = marine_df["wave_height"].rolling(24).mean()
    fig2.add_trace(go.Scatter(
        x=rolling.index, y=rolling,
        mode="lines", name="24h mean",
        line=dict(color=WARM_AMBER, width=2, dash="dash"),
    ))
    fig2.update_layout(**PLOT_LAYOUT, height=280,
                       yaxis_title="m", xaxis_title=None,
                       title=dict(text="Wave Height · QC-flagged spikes marked in red",
                                 font=dict(size=12)))
    st.plotly_chart(fig2, width='stretch')

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("<div class='section-header'>Wave Period</div>", unsafe_allow_html=True)
        fig3 = go.Figure(go.Scatter(
            x=marine_df.index, y=marine_df["wave_period"],
            mode="lines", line=dict(color=WAVE_TEAL, width=1.2), fill="tozeroy",
            fillcolor="rgba(46,196,182,0.12)",
        ))
        fig3.update_layout(**PLOT_LAYOUT, height=220, yaxis_title="s",
                           title=dict(text="Peak Wave Period", font=dict(size=12)))
        st.plotly_chart(fig3, width='stretch')

    with col_r:
        st.markdown("<div class='section-header'>Wave Direction (polar)</div>", unsafe_allow_html=True)
        if "wave_direction" in marine_df.columns:
            daily_dir = marine_df["wave_direction"].resample("D").mean()
            fig4 = go.Figure(go.Barpolar(
                r=[1]*len(daily_dir),
                theta=daily_dir.values,
                width=8,
                marker_color=OCEAN_BLUE,
                marker_line_color="#0b1d2e",
                marker_line_width=0.5,
                opacity=0.8,
            ))
            fig4.update_layout(
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
            st.plotly_chart(fig4, width='stretch')

    st.markdown("<div class='section-header'>Wind Wave vs Swell</div>", unsafe_allow_html=True)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=marine_df.index, y=marine_df["wave_height"],
        mode="lines", name="Total wave height", line=dict(color=OCEAN_BLUE, width=1),
    ))
    if "wind_wave_height" in marine_df.columns:
        fig5.add_trace(go.Scatter(
            x=marine_df.index, y=marine_df["wind_wave_height"],
            mode="lines", name="Wind-wave component", line=dict(color=WARM_AMBER, width=1),
            fill="tozeroy", fillcolor="rgba(244,162,97,0.08)",
        ))
        swell = marine_df["wave_height"] - marine_df["wind_wave_height"].clip(lower=0)
        fig5.add_trace(go.Scatter(
            x=marine_df.index, y=swell.clip(lower=0),
            mode="lines", name="Swell estimate", line=dict(color=WAVE_TEAL, width=1),
        ))
    fig5.update_layout(**PLOT_LAYOUT, height=220, yaxis_title="m",
                       title=dict(text="Wave decomposition (Total / Wind-wave / Swell estimate)",
                                  font=dict(size=12)))
    st.plotly_chart(fig5, width='stretch')


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

    # ── Section 1: Marine QC summary ─────────────────────────────────────────
    st.markdown("<div class='section-header'>Marine QC — daily % flagged bad</div>",
                unsafe_allow_html=True)

    # Build one time-series bar chart per marine QC variable that exists
    marine_qc_vars = [
        ("wh_qc", "Wave Height"),
        ("sst_qc", "Sea Surface Temperature"),
        ("wp_qc", "Wave Period"),
    ]
    available_marine_qc = [(col, label) for col, label in marine_qc_vars
                           if col in marine_df.columns]

    if available_marine_qc:
        n_cols = len(available_marine_qc)
        qc_cols = st.columns(n_cols)
        for i, (col, label) in enumerate(available_marine_qc):
            daily_pct = marine_df[col].resample("D").apply(
                lambda x: (x == 4).sum() / len(x) * 100 if len(x) > 0 else 0
            )
            bar_colors = [BAD_RED if v > 5 else GOOD_GREEN for v in daily_pct]
            fig = go.Figure(go.Bar(
                x=daily_pct.index, y=daily_pct.values,
                marker_color=bar_colors, marker_line_width=0,
            ))
            fig.update_layout(**PLOT_LAYOUT, height=200,
                              yaxis_title="% flagged",
                              title=dict(text=label, font=dict(size=12)))
            with qc_cols[i]:
                st.plotly_chart(fig, width='stretch')
    else:
        st.info("No marine QC columns available.")

    # ── Section 2: Argo QC summary ───────────────────────────────────────────
    st.markdown("<div class='section-header'>Argo QC — flag distribution</div>",
                unsafe_allow_html=True)

    if len(argo_df) > 0 and "temp_qc" in argo_df.columns:
        argo_qc_vars = [
            ("temp_qc", "Temperature"),
            ("psal_qc", "Salinity"),
        ]
        qc_summary = pd.DataFrame([
            {
                "Variable": label,
                "Good (1)": (argo_df[col] == 1).sum(),
                "Bad (4)": (argo_df[col] == 4).sum(),
            }
            for col, label in argo_qc_vars if col in argo_df.columns
        ])
        fig_argo_qc = go.Figure()
        fig_argo_qc.add_trace(go.Bar(name="Good (1)", x=qc_summary["Variable"],
                                     y=qc_summary["Good (1)"], marker_color=GOOD_GREEN))
        fig_argo_qc.add_trace(go.Bar(name="Bad (4)", x=qc_summary["Variable"],
                                     y=qc_summary["Bad (4)"], marker_color=BAD_RED))
        fig_argo_qc.update_layout(**PLOT_LAYOUT, height=220, barmode="stack",
                                  yaxis_title="# observations",
                                  title=dict(text="Argo profile QC flag distribution",
                                             font=dict(size=12)))
        st.plotly_chart(fig_argo_qc, width='stretch')
    else:
        st.info("No Argo QC data available.")

    # ── Section 3: Flagged marine observations ───────────────────────────────
    st.markdown("<div class='section-header'>Flagged marine observations</div>",
                unsafe_allow_html=True)

    # Collect all hourly rows where ANY marine QC flag == 4
    marine_flag_cols = {
        "wh_qc": "wave_height",
        "sst_qc": "sea_surface_temperature",
        "wp_qc": "wave_period",
    }
    # Only keep columns that are actually present
    present_flag_cols = {k: v for k, v in marine_flag_cols.items() if k in marine_df.columns}
    present_value_cols = [v for v in present_flag_cols.values() if v in marine_df.columns]

    if present_flag_cols:
        any_bad = (marine_df[[*present_flag_cols]] == 4).any(axis=1)
        bad_marine = marine_df[any_bad][present_value_cols + list(present_flag_cols.keys())].copy()
        bad_marine.index = bad_marine.index.strftime("%Y-%m-%d %H:%M")
        # Rename for display
        rename_map = {
            "wave_height": "Wave Height (m)",
            "sea_surface_temperature": "SST (°C)",
            "wave_period": "Wave Period (s)",
            "wh_qc": "Wave Ht QC",
            "sst_qc": "SST QC",
            "wp_qc": "Wave Period QC",
        }
        bad_marine = bad_marine.rename(columns=rename_map)
        if len(bad_marine) > 0:
            value_display_cols = [rename_map[v] for v in present_value_cols]
            fmt = {c: "{:.2f}" for c in value_display_cols}
            st.dataframe(
                bad_marine.style.format(fmt, na_rep="—"),
                height=200,
                width='stretch',
            )
        else:
            st.success("No bad marine QC flags in this time window.")

    # ── Section 4: Flagged Argo observations ─────────────────────────────────
    st.markdown("<div class='section-header'>Flagged Argo observations</div>",
                unsafe_allow_html=True)

    argo_flag_cols = {"temp_qc": "temp", "psal_qc": "psal"}
    present_argo_flags = {k: v for k, v in argo_flag_cols.items() if k in argo_df.columns}

    if len(argo_df) > 0 and present_argo_flags:
        any_bad_argo = (argo_df[[*present_argo_flags]] == 4).any(axis=1)
        bad_argo = argo_df[any_bad_argo][
            ["platform_number", "time", "pres"] +
            [v for v in present_argo_flags.values() if v in argo_df.columns] +
            list(present_argo_flags.keys())
            ].copy()
        bad_argo["time"] = pd.to_datetime(bad_argo["time"]).dt.strftime("%Y-%m-%d %H:%M")
        bad_argo = bad_argo.rename(columns={
            "platform_number": "Float ID",
            "time": "Time (UTC)",
            "pres": "Pressure (dbar)",
            "temp": "Temp (°C)",
            "psal": "Salinity (PSU)",
            "temp_qc": "Temp QC",
            "psal_qc": "Sal QC",
        })
        if len(bad_argo) > 0:
            fmt_argo = {"Temp (°C)": "{:.3f}", "Salinity (PSU)": "{:.3f}",
                        "Pressure (dbar)": "{:.0f}"}
            st.dataframe(
                bad_argo.style.format(fmt_argo, na_rep="—"),
                height=200,
                width='stretch',
            )
        else:
            st.success("No bad Argo QC flags in this time window.")
    else:
        st.info("No Argo data available.")


# ─── TAB 5: Anomaly Detection ────────────────────────────────────────────────

with tab_anomaly:
    st.markdown(
        "Two complementary unsupervised ML methods detect anomalies in the marine time series. "
        "**Isolation Forest** flags unusual multivariate combinations — e.g. low wave height "
        "paired with an unusually long period. "
        "**LSTM Autoencoder** learns the normal 24-hour temporal pattern and flags timesteps "
        "where reconstruction error is high — catching contextual anomalies that look "
        "plausible in isolation but break the expected sequence. "
        "Points flagged by **both** methods are the highest-confidence anomalies."
    )

    # ── Run ML (cached separately so it only re-runs when data changes) ───────
    @st.cache_data(ttl=3600, show_spinner=False)
    def run_anomaly_detection(marine_hash: int, version: int = 1):
        """
        Fits Isolation Forest and trains LSTM Autoencoder on the marine time series.
        Cached by a hash of the DataFrame so it re-runs only when data changes.
        Returns a dict of results rather than complex objects, which Streamlit
        can serialise cleanly for caching.
        """
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        df = marine_df.copy()

        # ── Feature engineering ───────────────────────────────────────────────
        core_vars = [v for v in ["wave_height", "wave_period", "sea_surface_temperature"]
                     if v in df.columns]

        for var in core_vars:
            df[f"{var}_roll_mean"] = df[var].rolling(6, min_periods=1).mean()
            df[f"{var}_roll_std"]  = df[var].rolling(6, min_periods=1).std().fillna(0)
            df[f"{var}_diff"]      = df[var].diff().fillna(0)

        df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

        features = (
            core_vars +
            [f"{v}_roll_mean" for v in core_vars] +
            [f"{v}_roll_std"  for v in core_vars] +
            [f"{v}_diff"      for v in core_vars] +
            ["hour_sin", "hour_cos"]
        )
        features = [f for f in features if f in df.columns]

        df_feat  = df[features].dropna()
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(df_feat.values)

        # ── Isolation Forest ──────────────────────────────────────────────────
        iso = IsolationForest(n_estimators=200, contamination=0.02,
                              random_state=42, n_jobs=-1)
        iso.fit(X_scaled)
        iso_scores  = -iso.score_samples(X_scaled)
        iso_anomaly = pd.Series(iso.predict(X_scaled) == -1, index=df_feat.index)

        # ── LSTM Autoencoder ──────────────────────────────────────────────────
        lstm_available = False
        lstm_anomaly   = pd.Series(False, index=df_feat.index)
        lstm_scores    = pd.Series(np.nan, index=df_feat.index)
        lstm_threshold = None

        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            from tensorflow import keras

            WINDOW = 24
            X_seq  = np.array([X_scaled[i:i+WINDOW]
                                for i in range(len(X_scaled) - WINDOW + 1)])
            n_feat = X_scaled.shape[1]

            inp = keras.Input(shape=(WINDOW, n_feat))
            x   = keras.layers.LSTM(32, activation="tanh",
                                     return_sequences=False)(inp)
            x   = keras.layers.RepeatVector(WINDOW)(x)
            x   = keras.layers.LSTM(32, activation="tanh",
                                     return_sequences=True)(x)
            out = keras.layers.TimeDistributed(
                      keras.layers.Dense(n_feat))(x)

            ae = keras.Model(inp, out)
            ae.compile(optimizer="adam", loss="mse")
            ae.fit(X_seq, X_seq, epochs=20, batch_size=64,
                   validation_split=0.1, verbose=0,
                   callbacks=[keras.callbacks.EarlyStopping(
                       patience=3, restore_best_weights=True,
                       monitor="val_loss")])

            X_pred       = ae.predict(X_seq, verbose=0)
            mse          = np.mean((X_seq - X_pred) ** 2, axis=(1, 2))
            scores_arr   = np.full(len(df_feat), np.nan)
            scores_arr[WINDOW-1:] = mse

            lstm_threshold = float(np.nanpercentile(scores_arr, 95))
            lstm_scores    = pd.Series(scores_arr, index=df_feat.index)
            lstm_anomaly   = pd.Series(scores_arr > lstm_threshold,
                                       index=df_feat.index)
            lstm_available = True

        except Exception as e:
            pass   # graceful fallback: show IF results only

        combined = iso_anomaly & (lstm_anomaly if lstm_available else iso_anomaly)

        return {
            "index":          df_feat.index,
            "core_vars":      core_vars,
            "df_feat":        df_feat[core_vars],           # values only
            "iso_scores":     pd.Series(iso_scores, index=df_feat.index),
            "iso_anomaly":    iso_anomaly,
            "lstm_scores":    lstm_scores,
            "lstm_anomaly":   lstm_anomaly,
            "lstm_threshold": lstm_threshold,
            "lstm_available": lstm_available,
            "combined":       combined,
        }

    # ── Run button + session state ───────────────────────────────────────────
    # ML training is expensive (~30s for LSTM), so we only run it on demand.
    # st.session_state persists values across Streamlit reruns within the same
    # browser session, so results stay visible after the button is clicked.

    marine_hash = hash((
        len(marine_df),
        str(marine_df.index.min()),
        str(marine_df.index.max()),
        round(float(marine_df["wave_height"].iloc[0]), 3),
    ))

    # Initialise session state keys on first load
    if "anomaly_res" not in st.session_state:
        st.session_state.anomaly_res  = None
        st.session_state.anomaly_hash = None

    col_btn, col_note = st.columns([1, 4])
    with col_btn:
        run_clicked = st.button(
            "▶ Run anomaly detection",
            type="primary",
            use_container_width=True,
        )
    with col_note:
        if st.session_state.anomaly_res is None:
            st.info("Click the button to fit the models and detect anomalies.")
        elif st.session_state.anomaly_hash != marine_hash:
            st.warning("The underlying data has changed. Re-run to update results.")
        else:
            st.success("Results are up to date.")

    if run_clicked:
        with st.spinner("Running anomaly detection… (first run may take ~30s for LSTM training)"):
            st.session_state.anomaly_res  = run_anomaly_detection(marine_hash)
            st.session_state.anomaly_hash = marine_hash

    # Only show results if the model has been run at least once
    if st.session_state.anomaly_res is None:
        st.stop()

    res = st.session_state.anomaly_res

    iso_anomaly    = res["iso_anomaly"]
    lstm_anomaly   = res["lstm_anomaly"]
    lstm_available = res["lstm_available"]
    combined       = res["combined"]
    iso_scores     = res["iso_scores"]
    lstm_scores    = res["lstm_scores"]
    lstm_threshold = res["lstm_threshold"]
    df_feat        = res["df_feat"]
    core_vars      = res["core_vars"]

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Isolation Forest flags",   f"{iso_anomaly.sum()}")
    a2.metric("LSTM Autoencoder flags",
              f"{lstm_anomaly.sum()}" if lstm_available else "N/A (no TF)")
    a3.metric("Combined (both methods)",  f"{combined.sum()}")
    a4.metric("High-confidence rate",
              f"{combined.mean()*100:.1f} %")

    # ── Per-variable time series ───────────────────────────────────────────────
    label_map = {
        "wave_height":             "Wave Height (m)",
        "wave_period":             "Wave Period (s)",
        "sea_surface_temperature": "SST (°C)",
    }
    COLORS_AN = {
        "normal":   "#3b9edd",
        "iso":      "#f4a261",
        "lstm":     "#2ec4b6",
        "combined": "#f87171",
    }

    for var in core_vars:
        st.markdown(
            f"<div class='section-header'>{label_map.get(var, var)} — anomalies</div>",
            unsafe_allow_html=True,
        )
        vals     = df_feat[var]
        fig_an   = go.Figure()

        # Normal points
        normal = ~iso_anomaly
        fig_an.add_trace(go.Scatter(
            x=vals.index[normal], y=vals[normal],
            mode="lines", name="Normal",
            line=dict(color=COLORS_AN["normal"], width=1),
        ))
        # Isolation Forest only
        iso_only = iso_anomaly & ~(lstm_anomaly if lstm_available else iso_anomaly)
        if iso_only.any():
            fig_an.add_trace(go.Scatter(
                x=vals.index[iso_only], y=vals[iso_only],
                mode="markers", name=f"Isolation Forest ({iso_only.sum()})",
                marker=dict(color=COLORS_AN["iso"], size=7, symbol="triangle-up"),
            ))
        # LSTM only
        if lstm_available:
            lstm_only = lstm_anomaly & ~iso_anomaly
            if lstm_only.any():
                fig_an.add_trace(go.Scatter(
                    x=vals.index[lstm_only], y=vals[lstm_only],
                    mode="markers", name=f"LSTM only ({lstm_only.sum()})",
                    marker=dict(color=COLORS_AN["lstm"], size=7,
                                symbol="triangle-down"),
                ))
        # Combined — both methods
        if combined.any():
            fig_an.add_trace(go.Scatter(
                x=vals.index[combined], y=vals[combined],
                mode="markers", name=f"Both methods ({combined.sum()})",
                marker=dict(color=COLORS_AN["combined"], size=10,
                            symbol="x", line=dict(width=2)),
            ))

        fig_an.update_layout(
            **PLOT_LAYOUT, height=240,
            yaxis_title=label_map.get(var, var),
            legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)",
                        font=dict(size=9)),
        )
        st.plotly_chart(fig_an, width="stretch")

    # ── Score panels ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Anomaly scores over time</div>",
                unsafe_allow_html=True)

    n_score_cols = 2 if lstm_available else 1
    score_cols   = st.columns(n_score_cols)

    with score_cols[0]:
        iso_thresh_val = float(np.percentile(iso_scores, 98))
        fig_iso = go.Figure()
        fig_iso.add_trace(go.Scatter(
            x=iso_scores.index, y=iso_scores,
            mode="lines", line=dict(color=COLORS_AN["iso"], width=0.9),
            fill="tozeroy", fillcolor="rgba(244,162,97,0.08)",
            name="Score",
        ))
        fig_iso.add_hline(
            y=iso_thresh_val, line_dash="dash",
            line_color=COLORS_AN["combined"], line_width=1.2,
            annotation_text="Threshold", annotation_font_color="#dce9f5",
        )
        fig_iso.update_layout(
            **PLOT_LAYOUT, height=220,
            yaxis_title="Score",
            title=dict(text="Isolation Forest — anomaly score (higher = more anomalous)",
                       font=dict(size=11)),
        )
        st.plotly_chart(fig_iso, width="stretch")

    if lstm_available and lstm_threshold is not None:
        with score_cols[1]:
            fig_lstm = go.Figure()
            fig_lstm.add_trace(go.Scatter(
                x=lstm_scores.index, y=lstm_scores,
                mode="lines", line=dict(color=COLORS_AN["lstm"], width=0.9),
                fill="tozeroy", fillcolor="rgba(46,196,182,0.08)",
                name="Reconstruction error",
            ))
            fig_lstm.add_hline(
                y=lstm_threshold, line_dash="dash",
                line_color=COLORS_AN["combined"], line_width=1.2,
                annotation_text="p95 threshold",
                annotation_font_color="#dce9f5",
            )
            fig_lstm.update_layout(
                **PLOT_LAYOUT, height=220,
                yaxis_title="MSE",
                title=dict(
                    text="LSTM Autoencoder — reconstruction error (higher = more anomalous)",
                    font=dict(size=11)),
            )
            st.plotly_chart(fig_lstm, width="stretch")

    # ── Flagged observations table ────────────────────────────────────────────
    st.markdown("<div class='section-header'>Highest-confidence anomalies (both methods)</div>",
                unsafe_allow_html=True)

    if combined.sum() > 0:
        top = df_feat[combined].copy()
        top["iso_score"] = iso_scores[combined]
        if lstm_available:
            top["lstm_mse"] = lstm_scores[combined]
        top = top.sort_values("iso_score", ascending=False)
        top.index = top.index.strftime("%Y-%m-%d %H:%M UTC")
        rename = {v: label_map.get(v, v) for v in core_vars}
        rename.update({"iso_score": "IF Score", "lstm_mse": "LSTM MSE"})
        top = top.rename(columns=rename)
        fmt = {label_map.get(v, v): "{:.3f}" for v in core_vars}
        fmt.update({"IF Score": "{:.3f}", "LSTM MSE": "{:.4f}"})
        st.dataframe(
            top.style.format(fmt, na_rep="—")
               .background_gradient(subset=["IF Score"],
                                    cmap="Reds", vmin=0),
            width="stretch",
        )
        st.markdown(
            "<div class='caption'>These are the observations flagged as anomalous by both "
            "Isolation Forest and the LSTM Autoencoder. Sorted by Isolation Forest score "
            "(higher = more anomalous in the multivariate feature space). "
            "In a real operational pipeline, these would be prioritised for manual inspection "
            "or fed into a downstream QC workflow.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.success("No high-confidence anomalies detected in this time window.")

    if not lstm_available:
        st.info(
            "LSTM Autoencoder results are not shown because TensorFlow is not installed "
            "in this environment. Add `tensorflow` to requirements.txt and redeploy to enable it."
        )


# ─── TAB 6: About ────────────────────────────────────────────────────────────

with tab_about:
    st.markdown("""
## About this dashboard

This application was built as a REST API skills demonstration for **DTO-Ocean_CofE 
Data Scientist** position at [GEOMAR Helmholtz Centre for Ocean Research Kiel](https://www.geomar.de)**.

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
| [Open-Meteo Marine API](https://open-meteo.com/en/docs/marine-weather-api) | SST, Wave height, direction, period | Hourly, NRT |
| [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api) | Air temp, precipitation, wind | Daily |
| [IFREMER ERDDAP — Argo](https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.html) | T/S profiles at depth | As floats surface (~10 days) |

### Study region
The **Ocean Science Centre Mindelo (OSCM)** at São Vicente, Cape Verde (16.88°N, 24.99°W)
is GEOMAR's primary West African research site. This dashboard covers the surrounding
Atlantic region (12–20°N, 20–27°W), the area relevant to the Ocean Centre of Excellence
for West Africa (Ocean-CofE) project.

### Code
Full source code and documentation: https://github.com/blockecology/oscm-dashboard/
    """)
