"""Streamlit app: python -m streamlit run app.py"""
import os

import altair as alt
import pandas as pd
import streamlit as st

from data import FEATURES, default_path, load
from llm import PROVIDERS, explain
from model import MODEL_PATH, load_artifacts, score, train

st.set_page_config(page_title="AI Financial Risk Assessment", page_icon="🛡️", layout="wide")

BAND_ORDER = ["LOW", "MODERATE", "HIGH", "VERY HIGH"]
BAND_COLORS = {"LOW": "#22C55E", "MODERATE": "#F59E0B", "HIGH": "#F97316", "VERY HIGH": "#EF4444"}
NICE = {
    "revolving_utilization": "Credit utilization", "age": "Age", "late_30_59": "Late 30-59 days",
    "debt_ratio": "Debt ratio", "monthly_income": "Monthly income",
    "open_credit_lines": "Open credit lines", "late_90_plus": "Late 90+ days",
    "real_estate_loans": "Real estate loans", "late_60_89": "Late 60-89 days",
    "dependents": "Dependents",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, label { font-family: 'Inter', sans-serif; }

/* Force a readable dark look even if .streamlit/config.toml is missing */
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
[data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"], [data-testid="stTickBarMin"], [data-testid="stTickBarMax"],
[data-testid="stThumbValue"], [data-testid="stSidebar"] * { color: #E6EDF7; }
.stApp input, .stApp textarea { color: #E6EDF7 !important; background: #131C2E !important; }
div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="base-input"] {
    background: #131C2E !important; border-color: #1F2C47 !important; color: #E6EDF7; }
[data-testid="stExpander"] { background: #131C2E; border: 1px solid #1F2C47; border-radius: 12px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color: #E6EDF7; }
[data-testid="stFileUploaderDropzone"] { background: #131C2E; }
[data-testid="stFileUploaderDropzone"] * { color: #B9C8E0; }
.stApp { background: radial-gradient(1100px 520px at 8% -8%, #16264f 0%, #0B1220 58%); }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; max-width: 1250px; }
section[data-testid="stSidebar"] { background: #0E1627; border-right: 1px solid #1E2A44; }

.hero { padding: 26px 30px; border-radius: 20px; margin-bottom: 14px;
        background: linear-gradient(120deg, rgba(34,211,165,.16), rgba(59,130,246,.16));
        border: 1px solid #22345a; }
.hero .tag { display:inline-block; font-size:.72rem; font-weight:700; letter-spacing:.14em; color:#22D3A5;
             background:rgba(34,211,165,.12); padding:4px 12px; border-radius:99px; margin-bottom:10px; }
.hero h1 { font-size:2.15rem; font-weight:800; margin:0 0 6px 0; padding:0;
           background: linear-gradient(90deg,#fff,#9be7d3 60%,#7fb2ff); -webkit-background-clip:text;
           -webkit-text-fill-color: transparent; }
.hero p { margin:0; color:#9FB2D1; font-size:1rem; }

.chips { display:flex; gap:10px; flex-wrap:wrap; margin: 6px 0 14px 0; }
.chip { background:#131C2E; border:1px solid #1F2C47; color:#B9C8E0; border-radius:99px;
        padding:6px 14px; font-size:.82rem; }
.chip b { color:#fff; }

.card { background: rgba(19,28,46,.88); border:1px solid #1F2C47; border-radius:18px;
        padding:20px 22px; margin-bottom:14px; }
.card h4 { margin:0 0 12px 0; font-size:1rem; font-weight:700; color:#E6EDF7; }
.label { color:#8FA3C4; font-size:.74rem; text-transform:uppercase; letter-spacing:.09em; font-weight:600; }
.value { font-size:1.9rem; font-weight:800; color:#fff; line-height:1.15; margin-top:4px; }
.sub { color:#8FA3C4; font-size:.82rem; margin-top:4px; }
.bigprob { font-size:3.6rem; font-weight:800; line-height:1.05; margin:6px 0 10px 0; }
.pill { display:inline-block; font-size:.78rem; font-weight:700; letter-spacing:.08em; padding:5px 14px;
        border-radius:99px; }

.meter { position:relative; margin:22px 0 30px 0; }
.meter .bar { display:flex; height:12px; border-radius:99px; overflow:hidden; }
.meter .marker { position:absolute; top:-7px; width:4px; height:26px; background:#fff; border-radius:3px;
                 box-shadow:0 0 0 3px rgba(255,255,255,.22); transform:translateX(-50%); }
.meter .ticks { position:relative; height:16px; margin-top:8px; font-size:.72rem; color:#8FA3C4; }
.meter .ticks span { position:absolute; transform:translateX(-50%); }

.drv { margin-bottom:13px; }
.drv-top { display:flex; justify-content:space-between; font-size:.9rem; color:#D5E0F2; margin-bottom:5px; }
.drv-top em { color:#8FA3C4; font-style:normal; font-size:.82rem; margin-left:6px; }
.drv-bar { height:8px; background:#1A2740; border-radius:99px; overflow:hidden; }
.drv-bar div { height:100%; border-radius:99px; }

.src { display:inline-block; font-size:.72rem; color:#8FA3C4; background:#131C2E; border:1px solid #1F2C47;
       border-radius:99px; padding:3px 11px; margin-top:6px; }
.empty { text-align:center; color:#8FA3C4; padding:46px 10px; }
.empty .big { font-size:2.4rem; }
.foot { text-align:center; color:#6F84A8; font-size:.78rem; margin-top:26px; }

.stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:none; }
.stTabs [data-baseweb="tab"] { background:#131C2E; border-radius:12px; padding:9px 20px; height:auto;
                               border:1px solid #1F2C47; }
.stTabs [aria-selected="true"] { background: linear-gradient(90deg,#22D3A5,#3B82F6) !important; border:none; }
.stTabs [aria-selected="true"] p { color:#06121f !important; font-weight:700; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

button[kind="primary"], button[data-testid="stBaseButton-primary"],
button[kind="primaryFormSubmit"], button[data-testid="stBaseButton-primaryFormSubmit"] {
    background: linear-gradient(90deg,#22D3A5,#3B82F6); color:#06121f; border:none; font-weight:700;
    border-radius:12px; padding:.6rem 1.2rem; }
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover { filter:brightness(1.08); color:#06121f; }
div[data-testid="stForm"] { border:1px solid #1F2C47; border-radius:18px; background:rgba(19,28,46,.88);
                            padding:18px 22px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------- html helpers ----------------
def H(s: str) -> str:
    return " ".join(line.strip() for line in s.strip().splitlines())


def md(s: str):
    st.markdown(H(s), unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "") -> str:
    return H(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div>'
             f'<div class="sub">{sub}</div></div>')


def pill(band: str) -> str:
    c = BAND_COLORS.get(band, "#8FA3C4")
    return f'<span class="pill" style="background:{c}22;color:{c};border:1px solid {c}66">{band} RISK</span>'


def risk_meter(p: float) -> str:
    cap = 0.5
    pos = min(p, cap) / cap * 100
    segs = [("LOW", 0.05), ("MODERATE", 0.10), ("HIGH", 0.20), ("VERY HIGH", 0.15)]
    bar = "".join(f'<div style="flex:{w};background:{BAND_COLORS[n]}"></div>' for n, w in segs)
    ticks = "".join(f'<span style="left:{x}%">{t}</span>'
                    for x, t in [(0, "0%"), (10, "5%"), (30, "15%"), (70, "35%"), (100, "50%+")])
    return H(f'<div class="meter"><div class="bar">{bar}</div>'
             f'<div class="marker" style="left:{pos}%"></div><div class="ticks">{ticks}</div></div>')


def fmt_val(f: str, v) -> str:
    if f == "monthly_income":
        return f"{float(v):,.0f}"
    if f in ("revolving_utilization", "debt_ratio"):
        return f"{float(v):.2f}"
    return f"{float(v):.0f}"


def drivers_html(drivers: list) -> str:
    mx = max((abs(d["impact"]) for d in drivers), default=1) or 1
    rows = ""
    for d in drivers:
        col = "#EF4444" if d["impact"] > 0 else "#22C55E"
        w = max(abs(d["impact"]) / mx * 100, 4)
        rows += (f'<div class="drv"><div class="drv-top"><span>{NICE[d["feature"]]}'
                 f'<em>{fmt_val(d["feature"], d["value"])}</em></span>'
                 f'<b style="color:{col}">{d["impact"] * 100:+.1f} pts</b></div>'
                 f'<div class="drv-bar"><div style="width:{w}%;background:{col}"></div></div></div>')
    legend = ('<div class="sub"><span style="color:#EF4444">&#9632;</span> raises risk &nbsp; '
              '<span style="color:#22C55E">&#9632;</span> lowers risk (vs. a typical borrower)</div>')
    return H(f'<div class="card"><h4>What drives this score</h4>{rows}{legend}</div>')


def style_chart(ch: alt.Chart) -> alt.Chart:
    return (ch.configure(background="transparent")
              .configure_view(strokeWidth=0)
              .configure_axis(labelColor="#8FA3C4", titleColor="#8FA3C4", gridColor="#1F2C47",
                              domainColor="#1F2C47", tickColor="#1F2C47", labelFontSize=12))


# ---------------- model state ----------------
@st.cache_resource
def _cached(mtime: float):
    return load_artifacts()


def get_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None
    return _cached(os.path.getmtime(MODEL_PATH))


def run_training(path=None):
    with st.spinner("First start: training the model on your dataset. Please wait, this page will appear when done..."):
        train(load(path))
    st.cache_resource.clear()


# ---------------- sidebar ----------------
with st.sidebar:
    md('<div class="label">Settings</div>')
    provider = st.selectbox("LLM provider (free tiers)", list(PROVIDERS), index=0)
    os.environ["LLM_PROVIDER"] = provider
    key_env = PROVIDERS[provider][1]
    if key_env:
        k = st.text_input(key_env, type="password")
        if k:
            os.environ[key_env] = k.strip()
        ok = bool(os.environ.get(key_env))
        md(f'<span class="src">{"🟢 key set" if ok else "⚪ no key: offline explanations"}</span>')
    else:
        md('<span class="src">🟢 local model, no key needed</span>')

    st.divider()
    md('<div class="label">Training data</div>')
    dp = default_path()
    st.caption("Default dataset: cs-training.csv found" if dp else "cs-training.csv not found (synthetic data)")
    if st.button("Retrain on default dataset"):
        run_training()
        st.rerun()
    csv = st.file_uploader("...or train on another CSV", type="csv")
    if csv and st.button("Train on uploaded CSV"):
        up_path = os.path.join(os.path.dirname(MODEL_PATH), "uploaded.csv")
        with open(up_path, "wb") as f:
            f.write(csv.getbuffer())
        run_training(up_path)
        st.rerun()

# ---------------- header (drawn first so the page is never blank) ----------------
md("""
<div class="hero">
  <div class="tag">AI &middot; CREDIT RISK</div>
  <h1>Financial Risk Assessment</h1>
  <p>Machine-learning default prediction with plain-English explanations from a free LLM.</p>
</div>
""")

# ---------------- ensure a trained model ----------------
art = get_artifacts()
if art is None or (default_path() and art["metrics"].get("source") == "synthetic"):
    run_training()
    art = get_artifacts()

model, baseline = art["model"], art["baseline"]
metrics, test_scored = art["metrics"], art["test_scored"]
n_rows = metrics["n_train"] + metrics["n_test"]

# ---------------- header ----------------
md(f"""
<div class="chips">
  <div class="chip">Dataset <b>{metrics['source']}</b></div>
  <div class="chip">Rows <b>{n_rows:,}</b></div>
  <div class="chip">Hold-out ROC-AUC <b>{metrics['auc']:.3f}</b></div>
  <div class="chip">Default rate <b>{metrics['default_rate']:.1%}</b></div>
</div>
""")

tab1, tab2, tab3 = st.tabs(["🎯  Assess applicant", "📊  Batch scoring", "🧠  Model insights"])

# ---------------- tab 1 ----------------
with tab1:
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        with st.form("applicant"):
            st.markdown("**Credit behaviour**")
            c1, c2 = st.columns(2)
            a = {}
            a["revolving_utilization"] = c1.slider("Credit card utilization", 0.0, 2.0, 0.6, 0.01)
            a["debt_ratio"] = c2.slider("Debt ratio", 0.0, 3.0, 0.5, 0.01)
            a["late_30_59"] = c1.number_input("Times 30-59 days late", 0, 20, 0)
            a["late_60_89"] = c2.number_input("Times 60-89 days late", 0, 20, 0)
            a["late_90_plus"] = c1.number_input("Times 90+ days late", 0, 20, 0)
            a["open_credit_lines"] = c2.number_input("Open credit lines", 0, 40, 6)
            st.markdown("**Profile**")
            d1, d2 = st.columns(2)
            a["monthly_income"] = d1.number_input("Monthly income", 0, 500000, 5000, 500)
            a["age"] = d2.slider("Age", 18, 90, 35)
            a["real_estate_loans"] = d1.number_input("Real estate loans", 0, 10, 1)
            a["dependents"] = d2.number_input("Dependents", 0, 10, 1)
            submitted = st.form_submit_button("Assess risk", type="primary")

    if submitted:
        res = score(model, baseline, a)
        with st.spinner("Generating explanation..."):
            text, src = explain(res)
        st.session_state["assessment"] = (res, text, src)

    with right:
        if "assessment" not in st.session_state:
            md("""
            <div class="card empty"><div class="big">🛡️</div>
            <h4>No assessment yet</h4>
            Fill in the applicant details and press <b>Assess risk</b>.</div>
            """)
        else:
            res, text, src = st.session_state["assessment"]
            p, band = res["default_probability"], res["risk_band"]
            c = BAND_COLORS[band]
            md(f"""
            <div class="card">
              <div class="label">Estimated default probability</div>
              <div class="bigprob" style="color:{c}">{p:.1%}</div>
              {pill(band)}
              {risk_meter(p)}
            </div>
            """)
            md(drivers_html(res["top_drivers"]))
            md('<div class="card"><h4>🤖 AI analyst explanation</h4></div>')
            st.markdown(text)
            md(f'<span class="src">Source: {src}</span>')

# ---------------- tab 2 ----------------
with tab2:
    st.markdown("##### Results on the trained dataset &nbsp;·&nbsp; held-out 25%, never seen in training")
    flagged = test_scored["risk_band"].isin(["HIGH", "VERY HIGH"]).mean()
    k1, k2, k3 = st.columns(3)
    k1.markdown(H(kpi("Applicants scored", f"{len(test_scored):,}", "held-out rows")), unsafe_allow_html=True)
    k2.markdown(H(kpi("Actual default rate", f"{test_scored['actual_default'].mean():.1%}",
                      "what really happened")), unsafe_allow_html=True)
    k3.markdown(H(kpi("Flagged high / very high", f"{flagged:.1%}", "by the model")), unsafe_allow_html=True)

    dist = (test_scored["risk_band"].value_counts().reindex(BAND_ORDER, fill_value=0)
            .rename_axis("band").reset_index(name="applicants"))
    actual = (test_scored.groupby("risk_band")["actual_default"].mean().reindex(BAND_ORDER)
              .fillna(0).rename_axis("band").reset_index(name="default_rate"))
    scale = alt.Scale(domain=BAND_ORDER, range=[BAND_COLORS[b] for b in BAND_ORDER])

    def band_chart(df, y, title, fmt):
        return style_chart(alt.Chart(df, title=alt.TitleParams(title, color="#E6EDF7", anchor="start",
                                                               fontSize=14, fontWeight=600))
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, size=58)
            .encode(x=alt.X("band:N", sort=BAND_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y(f"{y}:Q", title=None, axis=alt.Axis(format=fmt)),
                    color=alt.Color("band:N", scale=scale, legend=None),
                    tooltip=["band", alt.Tooltip(f"{y}:Q", format=fmt)])
            .properties(height=250))

    g1, g2 = st.columns(2)
    g1.altair_chart(band_chart(dist, "applicants", "Applicants per risk band", ",d"), theme=None)
    g2.altair_chart(band_chart(actual, "default_rate", "Actual default rate per band", ".0%"), theme=None)
    st.caption("A good model shows the actual default rate climbing from LOW to VERY HIGH.")

    st.markdown("##### Scored applicants")
    st.dataframe(
        test_scored, use_container_width=True, height=380,
        column_config={"default_probability": st.column_config.ProgressColumn(
            "default_probability", min_value=0.0, max_value=1.0, format="%.2f")})
    st.download_button("⬇ Download scored results", test_scored.to_csv(index=False), "scored_holdout.csv")

    with st.expander("Score a different CSV (optional)"):
        up = st.file_uploader("Applicants CSV (same feature columns)", type="csv", key="batch")
        if up:
            df = pd.read_csv(up)
            missing = [f for f in FEATURES if f not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                df["default_probability"] = model.predict_proba(df[FEATURES])[:, 1]
                st.dataframe(df.sort_values("default_probability", ascending=False))
                st.download_button("Download scored CSV", df.to_csv(index=False), "scored.csv")

# ---------------- tab 3 ----------------
with tab3:
    auc = metrics["auc"]
    verdict = ("Excellent (check for data leakage)" if auc >= 0.9 else "Strong" if auc >= 0.8 else
               "Good" if auc >= 0.7 else "Fair" if auc >= 0.6 else "Weak")
    i1, i2, i3 = st.columns(3)
    i1.markdown(H(kpi("ROC-AUC (hold-out)", f"{auc:.3f}", verdict)), unsafe_allow_html=True)
    i2.markdown(H(kpi("Training rows", f"{metrics['n_train']:,}", f"source: {metrics['source']}")),
                unsafe_allow_html=True)
    i3.markdown(H(kpi("Overall default rate", f"{metrics['default_rate']:.1%}",
                      "class imbalance: defaults are rare")), unsafe_allow_html=True)
    md(f"""
    <div class="card"><div class="label">Where this model sits (0.5 = coin flip, 1.0 = perfect)</div>
    <div class="meter" style="margin-bottom:6px"><div class="bar" style="background:#1A2740">
      <div style="width:{max(min((auc - 0.5) / 0.5, 1), 0) * 100}%;background:linear-gradient(90deg,#22D3A5,#3B82F6)"></div></div>
    <div class="ticks"><span style="left:0%">0.5</span><span style="left:50%">0.75</span>
    <span style="left:100%">1.0</span></div></div></div>
    """)

    imp = (pd.Series(metrics["importance"]).rename("importance").rename_axis("feature").reset_index())
    imp["feature"] = imp["feature"].map(NICE)
    st.altair_chart(style_chart(
        alt.Chart(imp, title=alt.TitleParams("Which features the model relies on (permutation importance)",
                                             color="#E6EDF7", anchor="start", fontSize=14, fontWeight=600))
        .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8, color="#22D3A5")
        .encode(y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("importance:Q", title="Drop in AUC when shuffled"),
                tooltip=["feature", alt.Tooltip("importance:Q", format=".4f")])
        .properties(height=330)), theme=None)
    with st.expander("Classification report (precision / recall at a 50% cutoff)"):
        st.code(metrics["report"])

md('<div class="foot">Decision support only, not a final credit decision. '
   'The ML model calculates risk; the LLM only explains it.</div>')