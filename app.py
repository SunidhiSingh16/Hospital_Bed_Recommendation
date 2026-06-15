import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

from src.data_loader import load_hospital_data, prepare_ml_features, add_department_breakdown
from src.poisson_model import PoissonPredictor
from src.ml_model import HospitalAdmissionPredictor
from src.ensemble_model import EnsembleForecaster, make_forward_forecast

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HBD Intelligence · Clinical Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────────────
# CSS  — full professional overhaul
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #070B14; }
::-webkit-scrollbar-thumb { background: #1E2D45; border-radius: 3px; }

/* ── Main area ── */
[data-testid="stAppViewContainer"] { background: #070B14; }
[data-testid="block-container"] { padding-top: 1.2rem; padding-bottom: 2rem; }

/* ── Sidebar — permanently fixed, never collapsible ── */
[data-testid="stSidebar"] {
    background: #0A0F1E;
    border-right: 1px solid #151F35;
    transform: none !important;
    min-width: 21rem !important;
    width: 21rem !important;
    visibility: visible !important;
    display: flex !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
    transform: translateX(0) !important;
    min-width: 21rem !important;
    width: 21rem !important;
}
/* Hide Streamlit's built-in collapse arrow button */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
section[data-testid="stSidebar"] > div > button,
section[data-testid="stSidebar"] > div > div > button {
    display: none !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0; }
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stNumberInput input {
    background: #0F1829 !important;
    border: 1px solid #1E2D45 !important;
    color: #E2E8F0 !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
    color: #64748B !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.09em;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #151F35;
    gap: 0;
    padding: 0 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #475569;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 12px 24px;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    background: transparent !important;
    transition: color 0.15s;
}
.stTabs [data-baseweb="tab"]:hover { color: #94A3B8 !important; }
.stTabs [aria-selected="true"] {
    color: #3B82F6 !important;
    border-bottom: 2px solid #3B82F6 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 24px 0 0 0; }

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 100%);
    color: #fff;
    border: none;
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.05em;
    padding: 10px 18px;
    width: 100%;
    transition: all 0.18s;
    text-transform: uppercase;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #2563EB, #3B82F6);
    box-shadow: 0 4px 18px rgba(59,130,246,0.35);
    transform: translateY(-1px);
}

/* ── Download buttons ── */
[data-testid="stDownloadButton"] > button {
    background: transparent;
    color: #3B82F6;
    border: 1px solid #1E3A5F;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    width: 100%;
    padding: 9px 16px;
    transition: all 0.18s;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(59,130,246,0.08);
    border-color: #3B82F6;
}

/* ── Divider ── */
hr { border-color: #151F35 !important; margin: 1.2rem 0 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0D1321;
    border: 1px solid #151F35;
    border-radius: 8px;
}
[data-testid="stExpander"] summary { color: #64748B !important; font-size: 0.8rem !important; }

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div { background: #1D4ED8 !important; }

/* ── Info/warning/error boxes ── */
[data-testid="stInfo"], [data-testid="stWarning"], [data-testid="stSuccess"], [data-testid="stError"] {
    border-radius: 7px;
    font-size: 0.83rem;
}

/* ── DataFrame ── */
[data-testid="stDataFrame"] {
    border: 1px solid #151F35;
    border-radius: 8px;
    overflow: hidden;
}

/* ── Metric (native) ── */
[data-testid="stMetricValue"] { font-weight: 800; }


</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# UI HELPER COMPONENTS
# ──────────────────────────────────────────────────────────────────────────────
def kpi_card(label, value, subtitle="", accent="#3B82F6"):
    return f"""
<div style="
    background: linear-gradient(145deg,#0D1829 0%,#0A1220 100%);
    border:1px solid #151F35;
    border-radius:10px;
    padding:18px 16px 14px;
    position:relative;
    overflow:hidden;
    min-height:105px;
">
    <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{accent};"></div>
    <div style="font-size:0.62rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.13em;color:#3D5170;margin-bottom:9px;">{label}</div>
    <div style="font-size:1.75rem;font-weight:800;color:#F1F5F9;
                line-height:1.1;font-variant-numeric:tabular-nums;">{value}</div>
    <div style="font-size:0.71rem;color:#4B6280;margin-top:7px;">{subtitle}</div>
</div>"""


def alert_html(level, title, body):
    cfg = {
        "green":  ("#10B981", "rgba(16,185,129,0.07)",  "rgba(16,185,129,0.18)", "✓"),
        "yellow": ("#F59E0B", "rgba(245,158,11,0.07)",  "rgba(245,158,11,0.18)", "⚠"),
        "red":    ("#EF4444", "rgba(239,68,68,0.07)",   "rgba(239,68,68,0.20)",  "✕"),
    }
    color, bg, border_c, icon = cfg[level]
    return f"""
<div style="
    background:{bg};
    border:1px solid {border_c};
    border-left:3px solid {color};
    border-radius:7px;
    padding:14px 18px;
    margin:14px 0 6px;
    display:flex;gap:14px;align-items:flex-start;
">
    <div style="width:22px;height:22px;border-radius:50%;background:{color};
                display:flex;align-items:center;justify-content:center;
                font-size:0.75rem;font-weight:800;color:#fff;flex-shrink:0;margin-top:1px;">{icon}</div>
    <div>
        <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.09em;color:{color};margin-bottom:3px;">{title}</div>
        <div style="font-size:0.86rem;color:#CBD5E1;line-height:1.5;">{body}</div>
    </div>
</div>"""


def section_label(text):
    st.markdown(
        f'<div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.13em;color:#2563EB;padding-bottom:8px;'
        f'border-bottom:1px solid #151F35;margin-bottom:14px;">{text}</div>',
        unsafe_allow_html=True
    )


def apply_chart_style(fig):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#64748B", size=11),
        title_font=dict(color="#94A3B8", size=12, family="Inter, sans-serif"),
        margin=dict(t=44, b=28, l=8, r=8),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#151F35",
            borderwidth=1,
            font=dict(color="#94A3B8", size=10),
        ),
    )
    fig.update_xaxes(
        gridcolor="#0F1A2D",
        linecolor="#151F35",
        tickfont=dict(color="#475569", size=10),
        zerolinecolor="#151F35",
    )
    fig.update_yaxes(
        gridcolor="#0F1A2D",
        linecolor="#151F35",
        tickfont=dict(color="#475569", size=10),
        zerolinecolor="#151F35",
    )
    return fig


def occupancy_gauge(rate, capacity, label):
    color = "#10B981" if rate < 70 else "#F59E0B" if rate < 85 else "#EF4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rate,
        number={
            "suffix": "%",
            "font": {"size": 42, "color": "#F1F5F9", "family": "Inter, sans-serif"},
            "valueformat": ".1f",
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#1E2D45",
                "tickfont": {"color": "#334155", "size": 9},
                "nticks": 6,
            },
            "bar": {"color": color, "thickness": 0.20},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 70],   "color": "rgba(16,185,129,0.08)"},
                {"range": [70, 85],  "color": "rgba(245,158,11,0.08)"},
                {"range": [85, 100], "color": "rgba(239,68,68,0.08)"},
            ],
            "threshold": {
                "line": {"color": "#475569", "width": 1.5},
                "thickness": 0.65,
                "value": 85,
            },
        },
        title={
            "text": (
                f"<span style='font-size:12px;color:#64748B;font-family:Inter'>{label}</span>"
                f"<br><span style='font-size:9px;color:#334155;font-family:Inter'>{capacity} beds · critical threshold 85%</span>"
            ),
            "font": {"color": "#64748B", "size": 12},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(t=80, b=5, l=24, r=24),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — all configuration
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand header
    st.markdown("""
<div style="
    background:linear-gradient(135deg,#0D1829 0%,#0A1220 100%);
    border-bottom:1px solid #151F35;
    padding:22px 20px 18px;
    margin-bottom:0;
">
    <div style="font-size:0.6rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.15em;color:#2563EB;margin-bottom:6px;">Clinical Analytics Platform</div>
    <div style="font-size:1.1rem;font-weight:800;color:#F1F5F9;line-height:1.2;">
        HBD Intelligence
    </div>
    <div style="font-size:0.72rem;color:#334155;margin-top:4px;">Bed Demand Forecasting System · v2.0</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='padding:0 16px;'>", unsafe_allow_html=True)
    st.write("")

    # Hospital profile
    section_label("Hospital Profile")
    hospital_name = st.text_input("Facility Name", value="City General Hospital",
                                   label_visibility="visible")
    department = st.selectbox("Department Context",
                               ["All Wards", "Internal Medicine", "Surgery",
                                "ICU / Critical Care", "Emergency Medicine"])

    st.write("")
    section_label("Capacity Configuration")
    bed_capacity = st.number_input("Total Bed Capacity", min_value=50,
                                    max_value=5000, value=200, step=10)
    icu_capacity = st.number_input("ICU Bed Capacity", min_value=5,
                                    max_value=500, value=30, step=5)

    st.write("")
    section_label("Simulation Scenario")
    scenario = st.selectbox("Load Profile", [
        "normal", "high_variation", "pandemic", "summer_surge", "custom"
    ], format_func=lambda x: {
        "normal": "Normal Operations",
        "high_variation": "High Variation",
        "pandemic": "Pandemic Surge",
        "summer_surge": "Summer Surge",
        "custom": "Custom Parameters",
    }[x])

    st.write("")
    section_label("Forecasting Engine")
    model_type = st.selectbox("Model Architecture", [
        "Ensemble (Poisson + GBM + XGBoost)",
        "Poisson + XGBoost",
        "Poisson + GBM",
        "Poisson Only",
    ])

    st.write("")
    with st.expander("Advanced Parameters"):
        base_admissions = st.slider("Base Daily Admissions", 10, 200, 50)
        weekday_impact = st.slider("Weekday Effect", 0, 50, 10)
        seasonal_impact = st.slider("Seasonal Amplitude", 0, 50, 15)
        special_events_count = st.slider("Surge Event Count", 0, 20, 5)
        special_events_magnitude = st.slider("Surge Magnitude", 10, 100, 30)
        start_date = st.date_input("Start Date",
                                    value=datetime.now().replace(month=1, day=1))

    st.write("")
    generate_btn = st.button("⟳  Generate Dataset", key="generate")

    # System status
    now_str = datetime.now().strftime("%H:%M · %d %b %Y")
    st.markdown(f"""
<div style="
    border-top:1px solid #151F35;
    padding:14px 0 8px;
    margin-top:24px;
">
    <div style="font-size:0.62rem;color:#2D4060;text-transform:uppercase;
                letter-spacing:0.1em;margin-bottom:6px;">System Status</div>
    <div style="display:flex;align-items:center;gap:8px;">
        <div style="width:7px;height:7px;border-radius:50%;background:#10B981;
                    box-shadow:0 0 6px #10B981;"></div>
        <span style="font-size:0.8rem;font-weight:600;color:#10B981;">Operational</span>
    </div>
    <div style="font-size:0.7rem;color:#2D4060;margin-top:5px;">{now_str}</div>
</div>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE + DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_default_data():
    df = load_hospital_data(scenario="normal")
    return add_department_breakdown(df)


if "df" not in st.session_state:
    st.session_state.df = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if st.session_state.df is None:
    try:
        st.session_state.df = load_default_data()
        st.session_state.data_loaded = True
    except Exception:
        st.session_state.data_loaded = False

if generate_btn:
    with st.spinner("Synthesising patient admission cohort…"):
        try:
            if scenario == "custom":
                params = {
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "base_admissions": base_admissions,
                    "weekday_impact": weekday_impact,
                    "seasonal_impact": seasonal_impact,
                    "special_events_count": special_events_count,
                    "special_events_magnitude": special_events_magnitude,
                }
                df_new = load_hospital_data(scenario="custom", custom_params=params)
            else:
                df_new = load_hospital_data(scenario=scenario)
            df_new = add_department_breakdown(df_new)
            st.session_state.df = df_new
            st.session_state.data_loaded = True
        except Exception as exc:
            st.error(f"Data generation error: {exc}")
            st.stop()

if not st.session_state.data_loaded or st.session_state.df is None:
    st.markdown(alert_html("yellow", "No data loaded",
                           "Use the sidebar to configure parameters and click Generate Dataset."),
                unsafe_allow_html=True)
    st.stop()

df = st.session_state.df


# ──────────────────────────────────────────────────────────────────────────────
# HEADER BANNER
# ──────────────────────────────────────────────────────────────────────────────
last_7_avg = df.tail(7)["daily_admissions"].mean()
prev_7_avg = df.tail(14).head(7)["daily_admissions"].mean()
trend_pct = ((last_7_avg - prev_7_avg) / prev_7_avg * 100) if prev_7_avg > 0 else 0.0
trend_arrow = "↑" if trend_pct > 0 else "↓"
trend_color = "#EF4444" if trend_pct > 5 else "#10B981" if trend_pct < -5 else "#F59E0B"
scenario_label = {
    "normal": "NORMAL OPERATIONS", "high_variation": "HIGH VARIATION",
    "pandemic": "PANDEMIC SURGE", "summer_surge": "SUMMER SURGE", "custom": "CUSTOM SCENARIO"
}.get(scenario, scenario.upper())
model_badge = "ENSEMBLE" if "Ensemble" in model_type else "ML + POISSON" if "Only" not in model_type else "POISSON ONLY"

st.markdown(f"""
<div style="
    background:linear-gradient(135deg,#0A1220 0%,#0D1829 50%,#0A1220 100%);
    border:1px solid #151F35;
    border-radius:12px;
    padding:22px 28px;
    margin-bottom:20px;
    position:relative;
    overflow:hidden;
">
    <div style="position:absolute;top:0;left:0;right:0;height:2px;
                background:linear-gradient(90deg,#1D4ED8,#3B82F6,#06B6D4,#8B5CF6);"></div>
    <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
        <div>
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                <span style="font-size:0.62rem;font-weight:700;text-transform:uppercase;
                             letter-spacing:0.15em;color:#2563EB;">Hospital Bed Demand Intelligence</span>
                <span style="font-size:0.6rem;font-weight:600;background:rgba(59,130,246,0.12);
                             color:#3B82F6;border:1px solid rgba(59,130,246,0.25);
                             border-radius:4px;padding:2px 8px;text-transform:uppercase;
                             letter-spacing:0.08em;">{scenario_label}</span>
                <span style="font-size:0.6rem;font-weight:600;background:rgba(139,92,246,0.10);
                             color:#8B5CF6;border:1px solid rgba(139,92,246,0.22);
                             border-radius:4px;padding:2px 8px;text-transform:uppercase;
                             letter-spacing:0.08em;">{model_badge}</span>
            </div>
            <div style="font-size:1.55rem;font-weight:800;color:#F1F5F9;line-height:1.2;
                        letter-spacing:-0.01em;">{hospital_name}</div>
            <div style="font-size:0.78rem;color:#334155;margin-top:5px;">{department}</div>
        </div>
        <div style="text-align:right;min-width:160px;">
            <div style="font-size:0.6rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.1em;color:#2D4060;margin-bottom:4px;">7-Day Admission Trend</div>
            <div style="font-size:1.4rem;font-weight:800;color:{trend_color};">
                {trend_arrow} {abs(trend_pct):.1f}%
            </div>
            <div style="font-size:0.7rem;color:#2D4060;margin-top:4px;">
                vs prior 7-day window
            </div>
            <div style="font-size:0.68rem;color:#1E3050;margin-top:8px;">
                Updated {datetime.now().strftime('%H:%M · %d %b %Y')}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# KPI STRIP
# ──────────────────────────────────────────────────────────────────────────────
avg_daily = df["daily_admissions"].mean()
std_daily = df["daily_admissions"].std()
peak_val = int(df["daily_admissions"].max())
peak_date = df.loc[df["daily_admissions"].idxmax(), "date"].strftime("%d %b")
weekday_avg = df[df["is_weekend"] == 0]["daily_admissions"].mean()
weekend_avg = df[df["is_weekend"] == 1]["daily_admissions"].mean()
seasonal_delta = (
    df[df["month"].isin([12, 1, 2])]["daily_admissions"].mean()
    - df[df["month"].isin([6, 7, 8])]["daily_admissions"].mean()
)
baseline_occ = (avg_daily / bed_capacity) * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(kpi_card(
    "Avg Daily Census",
    f"{avg_daily:.1f}",
    f"σ = {std_daily:.1f}  ·  {len(df)}-day cohort",
    "#3B82F6"
), unsafe_allow_html=True)
c2.markdown(kpi_card(
    "Peak Single-Day",
    f"{peak_val}",
    f"Recorded on {peak_date}",
    "#8B5CF6"
), unsafe_allow_html=True)
c3.markdown(kpi_card(
    "Weekday Load",
    f"{weekday_avg:.1f}",
    f"+{weekday_avg - weekend_avg:.1f} vs weekend avg",
    "#06B6D4"
), unsafe_allow_html=True)
c4.markdown(kpi_card(
    "Seasonal Δ  (Win–Sum)",
    f"{seasonal_delta:+.1f}",
    "admissions · winter vs summer",
    "#F59E0B"
), unsafe_allow_html=True)
c5.markdown(kpi_card(
    "Baseline Occupancy",
    f"{baseline_occ:.1f}%",
    f"{bed_capacity} total beds configured",
    "#10B981" if baseline_occ < 70 else "#F59E0B" if baseline_occ < 85 else "#EF4444"
), unsafe_allow_html=True)

st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab_census, tab_forecast, tab_report = st.tabs([
    "Patient Census",
    "Demand Forecast",
    "Clinical Report",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 · Patient Census
# ═══════════════════════════════════════════════════════════════════════════════
with tab_census:

    # ── Time series ──────────────────────────────────────────────────────────
    section_label("Admission Time Series")
    roll_avg = df["daily_admissions"].rolling(7, min_periods=1).mean()
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df["date"], y=df["daily_admissions"],
        name="Daily Census",
        line=dict(color="rgba(59,130,246,0.35)", width=1),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.04)",
        mode="lines",
    ))
    fig_ts.add_trace(go.Scatter(
        x=df["date"], y=roll_avg,
        name="7-Day Rolling Avg",
        line=dict(color="#3B82F6", width=2),
        mode="lines",
    ))
    fig_ts.add_hline(
        y=avg_daily, line_dash="dot", line_color="#475569",
        annotation_text=f"Mean {avg_daily:.0f}",
        annotation_font=dict(color="#475569", size=10),
    )
    fig_ts.update_layout(
        title="Daily Hospital Admissions with 7-Day Rolling Average",
        hovermode="x unified",
    )
    apply_chart_style(fig_ts)
    st.plotly_chart(fig_ts, use_container_width=True)

    # ── Weekly pattern + Heatmap ──────────────────────────────────────────────
    col_l, col_r = st.columns([1, 1.3], gap="medium")

    with col_l:
        section_label("Day-of-Week Admission Pattern")
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        weekly = (
            df.groupby("day_of_week")["daily_admissions"].mean()
            .reset_index()
            .assign(day=lambda d: d["day_of_week"].map(lambda x: day_names[x]))
        )
        colors_w = [
            "#3B82F6" if d not in ["Sat", "Sun"] else "#334155"
            for d in weekly["day"]
        ]
        fig_w = go.Figure(go.Bar(
            x=weekly["day"], y=weekly["daily_admissions"],
            marker_color=colors_w,
            text=weekly["daily_admissions"].round(1),
            textposition="outside",
            textfont=dict(color="#64748B", size=10),
        ))
        fig_w.update_layout(
            title="Avg Admissions by Day of Week",
            showlegend=False,
            bargap=0.25,
        )
        apply_chart_style(fig_w)
        st.plotly_chart(fig_w, use_container_width=True)

    with col_r:
        section_label("Admission Intensity Heatmap")
        pivot = df.pivot_table(
            values="daily_admissions",
            index="day_of_week", columns="month",
            aggfunc="mean"
        )
        pivot.index = [day_names[i] for i in pivot.index]
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot.columns = [month_names[m - 1] for m in pivot.columns]
        fig_heat = px.imshow(
            pivot,
            title="Admission Intensity  (Day × Month)",
            color_continuous_scale=[[0, "#0D1829"], [0.5, "#1D4ED8"], [1, "#3B82F6"]],
            labels=dict(color="Avg"),
            text_auto=".0f",
        )
        fig_heat.update_traces(textfont_size=9, textfont_color="#E2E8F0")
        fig_heat.update_layout(coloraxis_showscale=False)
        apply_chart_style(fig_heat)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Department breakdown ──────────────────────────────────────────────────
    section_label("Department-Level Admission Decomposition")
    st.caption(
        "Admission streams decomposed into ICU (≈15%), Emergency (≈22%), and General Ward (≈63%) "
        "using stochastic proportional allocation. Bounded by clinically derived fraction ranges."
    )

    fig_dept = go.Figure()
    fig_dept.add_trace(go.Scatter(
        x=df["date"], y=df["icu_admissions"],
        name="ICU / Critical Care", stackgroup="one",
        line=dict(width=0),
        fillcolor="rgba(239,68,68,0.70)",
    ))
    fig_dept.add_trace(go.Scatter(
        x=df["date"], y=df["emergency_admissions"],
        name="Emergency Medicine", stackgroup="one",
        line=dict(width=0),
        fillcolor="rgba(245,158,11,0.60)",
    ))
    fig_dept.add_trace(go.Scatter(
        x=df["date"], y=df["general_admissions"],
        name="General Ward", stackgroup="one",
        line=dict(width=0),
        fillcolor="rgba(59,130,246,0.50)",
    ))
    fig_dept.update_layout(
        title="Stacked Department Admissions",
        hovermode="x unified",
    )
    apply_chart_style(fig_dept)
    fig_dept.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_dept, use_container_width=True)

    # Department KPI cards
    d1, d2, d3 = st.columns(3)
    icu_avg = df["icu_admissions"].mean()
    emr_avg = df["emergency_admissions"].mean()
    gen_avg = df["general_admissions"].mean()
    _safe_total = avg_daily if avg_daily > 0 else 1.0
    d1.markdown(kpi_card(
        "ICU  ·  Avg Daily",
        f"{icu_avg:.1f}",
        f"{icu_avg / _safe_total * 100:.0f}% of total  ·  {icu_capacity} ICU beds",
        "#EF4444"
    ), unsafe_allow_html=True)
    d2.markdown(kpi_card(
        "Emergency  ·  Avg Daily",
        f"{emr_avg:.1f}",
        f"{emr_avg / _safe_total * 100:.0f}% of total",
        "#F59E0B"
    ), unsafe_allow_html=True)
    d3.markdown(kpi_card(
        "General Ward  ·  Avg Daily",
        f"{gen_avg:.1f}",
        f"{gen_avg / _safe_total * 100:.0f}% of total",
        "#3B82F6"
    ), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 · Demand Forecast
# ═══════════════════════════════════════════════════════════════════════════════

# Initialise prediction state
ml_prediction = None
metrics = None
importance = None
forecast_df = None
model_label = "ML"
fitted_model = None

with tab_forecast:

    # ── Poisson baseline ─────────────────────────────────────────────────────
    poisson_model = PoissonPredictor()
    recent_30 = df.tail(30)["daily_admissions"].values
    poisson_model.fit(recent_30)
    lower, upper = poisson_model.get_most_likely_range(confidence=0.80)
    poisson_occ = (poisson_model.lambda_ / bed_capacity) * 100

    section_label("Occupancy Forecast Overview")

    if model_type == "Poisson Only":
        # Single centred gauge
        _, gc, _ = st.columns([1, 1, 1])
        with gc:
            st.plotly_chart(
                occupancy_gauge(poisson_occ, bed_capacity, "Poisson Baseline Occupancy"),
                use_container_width=True,
            )
        alert_lvl = "green" if poisson_occ < 70 else "yellow" if poisson_occ < 85 else "red"
        alert_titles = {
            "green": "Normal Load — Within Safe Threshold",
            "yellow": "Elevated Load — Monitor Closely",
            "red": "Surge Condition — Activate Contingency Protocol",
        }
        st.markdown(alert_html(
            alert_lvl,
            alert_titles[alert_lvl],
            f"Poisson model predicts <strong>{poisson_occ:.1f}%</strong> bed occupancy "
            f"({poisson_model.lambda_:.0f} expected admissions · {bed_capacity} total beds). "
            f"80% prediction interval: <strong>{lower:.0f} – {upper:.0f}</strong> admissions."
        ), unsafe_allow_html=True)

    else:
        # Train ML / ensemble
        X, y = prepare_ml_features(df)
        train_size = int(0.8 * len(df))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        if model_type == "Ensemble (Poisson + GBM + XGBoost)":
            model_label = "Ensemble"
            fitted_model = EnsembleForecaster()
            fitted_model.fit(X_train, y_train, recent_30)
        else:
            mtype = "xgboost" if "XGBoost" in model_type else "gbm"
            model_label = "XGBoost" if mtype == "xgboost" else "GBM"
            fitted_model = HospitalAdmissionPredictor(model_type=mtype)
            fitted_model.fit(X_train, y_train)

        metrics = (
            fitted_model.evaluate(X_test, y_test)
            if hasattr(fitted_model, "evaluate")
            else None
        )
        ml_prediction = float(fitted_model.predict(X.iloc[-1:])[0])
        ml_occ = (ml_prediction / bed_capacity) * 100

        if isinstance(fitted_model, EnsembleForecaster):
            importance = fitted_model.get_feature_importance(X.columns)
            forecast_df = fitted_model.forecast_next_n_days(df, n=7)
        else:
            importance = fitted_model.get_feature_importance(X.columns)
            forecast_df = make_forward_forecast(fitted_model, df, n=7)

        # Dual gauges
        ga, gb = st.columns(2, gap="medium")
        with ga:
            st.plotly_chart(
                occupancy_gauge(poisson_occ, bed_capacity, "Poisson Statistical Baseline"),
                use_container_width=True,
            )
        with gb:
            st.plotly_chart(
                occupancy_gauge(ml_occ, bed_capacity, f"{model_label} Forecast Occupancy"),
                use_container_width=True,
            )

        # Alert (ML-driven)
        alert_lvl = "green" if ml_occ < 70 else "yellow" if ml_occ < 85 else "red"
        delta_pp = ml_occ - poisson_occ
        st.markdown(alert_html(
            alert_lvl,
            {
                "green": "Normal Load — Within Safe Threshold",
                "yellow": "Elevated Load — Pre-position Resources",
                "red": "Surge Condition — Activate Emergency Capacity Protocol",
            }[alert_lvl],
            f"{model_label} model predicts <strong>{ml_prediction:.0f} admissions</strong> "
            f"→ <strong>{ml_occ:.1f}% occupancy</strong> ({bed_capacity} beds). "
            f"Delta vs Poisson baseline: <strong>{delta_pp:+.1f} pp</strong>. "
            f"Poisson 80% CI: {lower:.0f}–{upper:.0f}."
        ), unsafe_allow_html=True)

        st.divider()

        # ── Model intelligence ─────────────────────────────────────────────
        section_label("Model Performance & Feature Intelligence")

        met_a, met_b, met_c, met_d = st.columns(4)
        if metrics:
            met_a.markdown(kpi_card(
                "Mean Absolute Error",
                f"{metrics['mae']:.2f}",
                "admissions · test set",
                "#3B82F6"
            ), unsafe_allow_html=True)
            met_b.markdown(kpi_card(
                "RMSE",
                f"{metrics['rmse']:.2f}",
                "root mean squared error",
                "#8B5CF6"
            ), unsafe_allow_html=True)
        met_c.markdown(kpi_card(
            f"{model_label} Prediction",
            f"{ml_prediction:.0f}",
            "next-day point estimate",
            "#06B6D4"
        ), unsafe_allow_html=True)
        met_d.markdown(kpi_card(
            "Poisson λ",
            f"{poisson_model.lambda_:.1f}",
            f"80% CI: {lower:.0f}–{upper:.0f}",
            "#10B981"
        ), unsafe_allow_html=True)

        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

        col_imp, col_dist = st.columns([1, 1], gap="medium")

        with col_imp:
            section_label("Feature Importance")
            imp_df = (
                pd.DataFrame({"Feature": list(importance.keys()),
                               "Importance": list(importance.values())})
                .sort_values("Importance")
            )
            fig_imp = go.Figure(go.Bar(
                x=imp_df["Importance"],
                y=imp_df["Feature"],
                orientation="h",
                marker=dict(
                    color=imp_df["Importance"],
                    colorscale=[[0, "#0D1829"], [1, "#3B82F6"]],
                    line=dict(width=0),
                ),
                text=imp_df["Importance"].round(3),
                textposition="outside",
                textfont=dict(color="#475569", size=10),
            ))
            fig_imp.update_layout(title="Predictive Feature Weights", showlegend=False)
            apply_chart_style(fig_imp)
            st.plotly_chart(fig_imp, use_container_width=True)

        with col_dist:
            section_label("Poisson Arrival Distribution")
            fig_poi = poisson_model.plot_distribution()
            st.pyplot(fig_poi, use_container_width=True)

        st.divider()

    # ── 7-day forward forecast ────────────────────────────────────────────────
    if model_type != "Poisson Only" and forecast_df is not None:
        section_label("7-Day Forward Admission Forecast")
        st.caption(
            "Forward projections extrapolated over trained model weights using future calendar features. "
            "Confidence bands = ±1.96σ of trailing 7-day admission volatility."
        )

        hist_tail = df.tail(21)
        fig_fc = go.Figure()

        # Historical
        fig_fc.add_trace(go.Scatter(
            x=hist_tail["date"], y=hist_tail["daily_admissions"],
            name="Historical",
            line=dict(color="#334155", width=1.5),
            mode="lines",
        ))

        # CI band (drawn before the line so it's behind)
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
            y=pd.concat([forecast_df["ci_upper"], forecast_df["ci_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(59,130,246,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Confidence Band",
            showlegend=True,
        ))

        # Forecast line
        fig_fc.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df["prediction"],
            name=f"{model_label} Forecast",
            line=dict(color="#3B82F6", width=2.5),
            mode="lines+markers",
            marker=dict(size=7, color="#3B82F6",
                        line=dict(color="#070B14", width=1.5)),
        ))

        # Capacity threshold
        fig_fc.add_hline(
            y=bed_capacity,
            line_dash="dot", line_color="#EF4444", line_width=1,
            annotation_text=f"Max Capacity  {bed_capacity}",
            annotation_font=dict(color="#EF4444", size=10),
            annotation_position="top right",
        )

        # Surge zone shading
        fig_fc.add_hrect(
            y0=bed_capacity * 0.85, y1=bed_capacity * 1.05,
            fillcolor="rgba(239,68,68,0.04)",
            line_width=0,
            annotation_text="Surge Zone",
            annotation_font=dict(color="#EF4444", size=9),
            annotation_position="top left",
        )

        fig_fc.update_layout(
            title="21-Day Context Window + 7-Day Forward Forecast",
            hovermode="x unified",
        )
        apply_chart_style(fig_fc)
        fig_fc.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Forecast table — styled
        section_label("Forecast Table")
        fc_display = forecast_df.copy()
        fc_display["date"] = fc_display["date"].dt.strftime("%a  %d %b %Y")
        fc_display["prediction"] = fc_display["prediction"].round(1)
        fc_display["ci_lower"] = fc_display["ci_lower"].round(1)
        fc_display["ci_upper"] = fc_display["ci_upper"].round(1)
        fc_display["Occupancy %"] = (
            (fc_display["prediction"] / bed_capacity * 100).round(1).astype(str) + "%"
        )
        fc_display.columns = [
            "Date", "Predicted Admissions", "Lower CI  (95%)",
            "Upper CI  (95%)", "Est. Occupancy"
        ]
        st.dataframe(fc_display, use_container_width=True, hide_index=True)

    elif model_type == "Poisson Only":
        section_label("Poisson Distribution Analysis")
        _, pc, _ = st.columns([0.5, 2, 0.5])
        with pc:
            fig_poi2 = poisson_model.plot_distribution()
            st.pyplot(fig_poi2, use_container_width=True)
        st.markdown(f"""
<div style="background:#0D1829;border:1px solid #151F35;border-radius:8px;
            padding:16px 20px;margin-top:12px;display:grid;
            grid-template-columns:1fr 1fr 1fr;gap:16px;text-align:center;">
    <div>
        <div style="font-size:0.6rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Fitted λ</div>
        <div style="font-size:1.5rem;font-weight:800;color:#3B82F6;">{poisson_model.lambda_:.1f}</div>
    </div>
    <div>
        <div style="font-size:0.6rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">80% Prediction Interval</div>
        <div style="font-size:1.5rem;font-weight:800;color:#F1F5F9;">{lower:.0f} – {upper:.0f}</div>
    </div>
    <div>
        <div style="font-size:0.6rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Expected Occupancy</div>
        <div style="font-size:1.5rem;font-weight:800;color:{'#10B981' if poisson_occ < 70 else '#F59E0B' if poisson_occ < 85 else '#EF4444'};">{poisson_occ:.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 · Clinical Report
# ═══════════════════════════════════════════════════════════════════════════════
with tab_report:
    section_label("Executive Summary")

    report_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    report_md = f"""### Hospital Bed Demand Intelligence — Clinical Forecast Report

**Facility:** {hospital_name}  ·  **Department:** {department}
**Generated:** {report_time}  ·  **Scenario:** {scenario.replace("_", " ").title()}  ·  **Model:** {model_type}  ·  **Bed Capacity:** {bed_capacity}

---

#### Cohort Overview

| Metric | Value |
|--------|-------|
| Observation Period | {df["date"].min().strftime("%d %b %Y")} – {df["date"].max().strftime("%d %b %Y")} |
| Total Observation Days | {len(df)} |
| Mean Daily Admissions | {avg_daily:.1f} ± {std_daily:.1f} σ |
| Single-Day Peak | {peak_val} admissions ({peak_date}) |
| 7-Day Admission Trend | {trend_arrow} {abs(trend_pct):.1f}% vs prior 7-day window |
| Weekday vs Weekend Δ | +{weekday_avg - weekend_avg:.1f} admissions on weekdays |
| Winter – Summer Seasonal Δ | {seasonal_delta:.1f} admissions |
| Baseline Occupancy Rate | {baseline_occ:.1f}% ({avg_daily:.0f} / {bed_capacity} beds) |

#### Department-Level Breakdown (Average Daily)

| Department | Avg Daily Admissions | Share of Total | Capacity Ref |
|-----------|---------------------|---------------|-------------|
| ICU / Critical Care | {df["icu_admissions"].mean():.1f} | {df["icu_admissions"].mean() / avg_daily * 100:.0f}% | {icu_capacity} ICU beds |
| Emergency Medicine | {df["emergency_admissions"].mean():.1f} | {df["emergency_admissions"].mean() / avg_daily * 100:.0f}% | — |
| General Ward | {df["general_admissions"].mean():.1f} | {df["general_admissions"].mean() / avg_daily * 100:.0f}% | — |

#### Poisson Statistical Model

| Parameter | Value |
|-----------|-------|
| Fitted λ (MLE arrival rate, 30-day window) | {poisson_model.lambda_:.2f} |
| 80% Prediction Interval | {lower:.0f} – {upper:.0f} admissions |
| Predicted Bed Occupancy | {poisson_occ:.1f}% |
"""

    if ml_prediction is not None:
        report_md += f"""
#### Machine Learning Forecast — {model_type}

| Metric | Value |
|--------|-------|
| Next-Day Point Prediction | {ml_prediction:.1f} admissions |
| Predicted Bed Occupancy | {ml_prediction / bed_capacity * 100:.1f}% |
| Δ vs Poisson Baseline | {ml_prediction / bed_capacity * 100 - poisson_occ:+.1f} percentage points |
"""
        if metrics:
            report_md += f"""| Mean Absolute Error (MAE) | {metrics["mae"]:.2f} admissions |
| Root Mean Squared Error (RMSE) | {metrics["rmse"]:.2f} admissions |
"""
        if importance:
            report_md += "\n**Feature Importance Ranking:**\n\n"
            for feat, val in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                report_md += f"- `{feat}`: {val:.3f}\n"

    if forecast_df is not None:
        report_md += "\n#### 7-Day Forward Admission Forecast\n\n"
        report_md += "| Date | Predicted | Lower CI | Upper CI | Est. Occupancy |\n"
        report_md += "|------|-----------|----------|----------|---------------|\n"
        for _, row in forecast_df.iterrows():
            occ = row["prediction"] / bed_capacity * 100
            report_md += (
                f"| {row['date'].strftime('%a %d %b')} "
                f"| {row['prediction']:.0f} "
                f"| {row['ci_lower']:.0f} "
                f"| {row['ci_upper']:.0f} "
                f"| {occ:.1f}% |\n"
            )

    st.markdown(report_md)
    st.divider()

    section_label("Export")
    dl1, dl2 = st.columns(2, gap="medium")

    with dl1:
        st.download_button(
            "↓  Download Report  (.md)",
            report_md,
            file_name=f"hbd_report_{scenario}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
        )

    with dl2:
        rows_export = [
            {"date": "next_day", "model": "Poisson",
             "prediction": poisson_model.lambda_,
             "ci_lower": lower, "ci_upper": upper},
        ]
        if ml_prediction is not None:
            rows_export.append({
                "date": "next_day", "model": model_type,
                "prediction": ml_prediction,
                "ci_lower": "", "ci_upper": "",
            })
        if forecast_df is not None:
            for _, r in forecast_df.iterrows():
                rows_export.append({
                    "date": r["date"].strftime("%Y-%m-%d"),
                    "model": f"{model_label}_forecast",
                    "prediction": round(r["prediction"], 1),
                    "ci_lower": round(r["ci_lower"], 1),
                    "ci_upper": round(r["ci_upper"], 1),
                })
        st.download_button(
            "↓  Download Predictions  (.csv)",
            pd.DataFrame(rows_export).to_csv(index=False),
            file_name=f"hbd_predictions_{scenario}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
