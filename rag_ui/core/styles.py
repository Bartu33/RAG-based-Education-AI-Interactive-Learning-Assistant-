import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

:root{
  --teal:#2dd4bf; --teal2:#14b8a6; --bg:#0b0f14; --panel:#121923; --panel2:#171f2b;
  --line:#222d3b; --muted:#8b98a9; --text:#e6edf3;
}
html, body, .stApp, [class*="css"]{ font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif; }

/* ---------- arka plan ---------- */
.stApp{
  background:
    radial-gradient(900px 500px at 88% -8%, rgba(45,212,191,.11), transparent 60%),
    radial-gradient(700px 420px at -8% 108%, rgba(99,102,241,.09), transparent 60%),
    var(--bg);
}
header[data-testid="stHeader"]{ background:transparent; }
[data-testid="stDecoration"], #MainMenu, footer{ display:none !important; }

/* kicker'ın kesilmesini önleyen üst boşluk */
.block-container{ padding-top:3.5rem !important; padding-bottom:4rem; max-width:1100px; }

/* ---------- tipografi ---------- */
h1{ font-weight:800 !important; font-size:2.4rem !important; letter-spacing:-.03em; line-height:1.15 !important; padding:.2rem 0 .6rem !important; }
h2, h3{ font-weight:700 !important; letter-spacing:-.02em; }
p, li, label{ line-height:1.6; }
.kicker{
  display:inline-flex; align-items:center; gap:.5rem;
  font-family:'JetBrains Mono',monospace; font-size:.72rem; letter-spacing:.22em;
  color:var(--teal); text-transform:uppercase; margin:0 0 .5rem; padding:.28rem .7rem;
  border:1px solid rgba(45,212,191,.28); border-radius:999px; background:rgba(45,212,191,.07);
}
.kicker::before{ content:""; width:6px; height:6px; border-radius:50%; background:var(--teal); box-shadow:0 0 8px var(--teal); }
.muted{ color:var(--muted); }

/* ---------- yan menü ---------- */
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0f1620,#0a1017); border-right:1px solid var(--line); }
[data-testid="stSidebarNav"] a{ border-radius:10px; padding:.5rem .75rem; margin:2px 0; transition:background .15s; }
[data-testid="stSidebarNav"] a:hover{ background:rgba(45,212,191,.08); }
[data-testid="stSidebarNav"] a[aria-current="page"]{ background:rgba(45,212,191,.14); box-shadow:inset 3px 0 0 var(--teal); font-weight:600; }

/* ---------- butonlar ---------- */
button[kind="primary"], button[data-testid="stBaseButton-primary"]{
  background:linear-gradient(135deg,#2dd4bf,#14b8a6) !important; color:#04211d !important;
  border:0 !important; font-weight:700 !important; border-radius:12px !important;
  box-shadow:0 6px 20px rgba(45,212,191,.22); transition:transform .15s, box-shadow .15s;
}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover{ transform:translateY(-1px); box-shadow:0 10px 26px rgba(45,212,191,.32); }
button[kind="secondary"], button[data-testid="stBaseButton-secondary"]{
  border-radius:12px !important; border:1px solid var(--line) !important; background:var(--panel) !important; transition:border-color .15s, transform .15s;
}
button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover{ border-color:var(--teal) !important; color:var(--teal) !important; transform:translateY(-1px); }

/* ---------- girdiler ---------- */
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"], textarea{ border-radius:12px !important; }
[data-testid="stFileUploaderDropzone"]{ border:1.5px dashed rgba(45,212,191,.45); border-radius:16px; background:rgba(45,212,191,.04); padding:1.6rem; }
[data-testid="stFileUploaderDropzone"]:hover{ background:rgba(45,212,191,.08); }

/* ---------- kartlar / paneller ---------- */
[data-testid="stVerticalBlockBorderWrapper"]{ border-radius:16px !important; border-color:var(--line) !important; }
[data-testid="stMetric"]{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:1rem 1.2rem; }
[data-testid="stMetricLabel"] p{ color:var(--muted); font-size:.74rem; text-transform:uppercase; letter-spacing:.08em; }
[data-testid="stMetricValue"]{ font-weight:800; }
[data-testid="stExpander"] details{ border:1px solid var(--line) !important; border-radius:14px !important; background:var(--panel); }
[data-testid="stAlert"]{ border-radius:14px; }
[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:14px; overflow:hidden; }
.stProgress > div > div > div > div{ background:linear-gradient(90deg,#14b8a6,#2dd4bf); }
[data-testid="stPageLink"] a{ color:var(--teal); font-weight:600; border-radius:10px; padding:.25rem .6rem; }
[data-testid="stPageLink"] a:hover{ background:rgba(45,212,191,.1); }

/* ---------- sohbet ---------- */
[data-testid="stChatMessage"]{ background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:1rem 1.2rem; margin-bottom:.8rem; }
[data-testid="stChatInput"]{ border-radius:18px; }

/* ---------- özel bileşenler ---------- */
.pill{ display:inline-block; padding:.15rem .7rem; margin-right:.4rem; border-radius:999px; font-size:.74rem; font-weight:600;
  color:var(--teal); background:rgba(45,212,191,.1); border:1px solid rgba(45,212,191,.3); }
.hero{
  position:relative; overflow:hidden; padding:2.2rem 2.2rem 1.8rem; margin:.4rem 0 1.4rem;
  border:1px solid rgba(45,212,191,.25); border-radius:24px;
  background:linear-gradient(135deg, rgba(45,212,191,.12), rgba(99,102,241,.08) 60%, rgba(18,25,35,.6));
}
.hero h1{ font-size:2.7rem !important; margin:0 0 .4rem !important; padding:0 !important; }
.hero h1 span{ background:linear-gradient(90deg,#2dd4bf,#7dd3fc); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero p{ color:#b6c2d1; font-size:1.05rem; max-width:640px; margin:0; }
[class*="st-key-card_"]{
  background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:1.2rem 1.2rem .8rem;
  transition:transform .18s, border-color .18s, box-shadow .18s; height:100%;
}
[class*="st-key-card_"]:hover{ transform:translateY(-4px); border-color:rgba(45,212,191,.55); box-shadow:0 14px 34px rgba(0,0,0,.35); }
.ico{ width:46px; height:46px; display:grid; place-items:center; font-size:1.4rem; border-radius:14px;
  background:rgba(45,212,191,.12); border:1px solid rgba(45,212,191,.25); margin-bottom:.7rem; }
.card-title{ font-weight:700; font-size:1.05rem; margin-bottom:.2rem; }
.card-desc{ color:var(--muted); font-size:.88rem; line-height:1.5; margin-bottom:.4rem; min-height:2.6rem; }
.step{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.2rem; height:100%; }
.step b.n{ display:inline-grid; place-items:center; width:28px; height:28px; border-radius:50%; margin-bottom:.6rem;
  background:var(--teal); color:#04211d; font-size:.85rem; }
.step .t{ font-weight:700; margin-bottom:.15rem; }
.step .d{ color:var(--muted); font-size:.86rem; }
.privacy{ display:flex; gap:.8rem; align-items:center; margin-top:1.4rem; padding:.9rem 1.2rem; border-radius:14px;
  background:rgba(45,212,191,.06); border:1px solid rgba(45,212,191,.2); color:#b6c2d1; font-size:.9rem; }
</style>
"""

def inject():
    st.markdown(CSS, unsafe_allow_html=True)

def kicker(t):
    st.markdown(f'<div class="kicker">{t}</div>', unsafe_allow_html=True)