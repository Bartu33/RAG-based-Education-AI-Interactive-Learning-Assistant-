import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Nunito:wght@700;800;900&family=JetBrains+Mono:wght@500&display=swap');

/* Streamlit fades the whole page while it reruns (every click, the video player's play / pause,
   the transcription progress). The page is redrawn in place, so keep it at full opacity. */
[data-stale="true"]{ opacity:1 !important; transition:none !important; }
/* a long displayed formula scrolls sideways instead of being cut off in a narrow chat */
.katex-display{ overflow-x:auto; overflow-y:hidden; padding:.2rem 0; }

/* Theme: Khan Academy-like (white, one blue accent) everywhere; the practice questions use
   Duolingo-like 3D buttons, answer tiles and green / red feedback (see "quiz" below).
   --teal is kept as an alias of the accent for older rules. */
:root{
  --accent:#1865F2; --accent-dark:#0B4FC5; --accent-soft:#E8F0FE; --teal:var(--accent);
  --green:#58CC02; --green-dark:#46A302; --green-soft:#E9F8DC;
  --red:#E5484D; --red-dark:#C2373B; --red-soft:#FDECEC;
  --bg:#FFFFFF; --panel:#FFFFFF; --panel2:#F7F8FA; --line:#E3E6EB; --muted:#5F6B7A; --text:#21242C;
  --shadow:0 1px 2px rgba(33,36,44,.06), 0 4px 14px rgba(33,36,44,.06);
}
html, body, .stApp, [class*="css"]{ font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif; }
.stApp{ background:var(--bg); color:var(--text); }
header[data-testid="stHeader"]{ background:transparent; }
[data-testid="stDecoration"], #MainMenu, footer{ display:none !important; }
.block-container{ padding:2.6rem clamp(1.2rem, 3vw, 3.5rem) 4rem !important; max-width:1680px !important; }
[data-testid="stBottomBlockContainer"]{ max-width:1680px; }

/* ---------- typography ---------- */
h1{ font-weight:800 !important; font-size:2.2rem !important; letter-spacing:-.025em; line-height:1.15 !important;
  padding:.2rem 0 .6rem !important; color:var(--text); }
h2, h3{ font-weight:700 !important; letter-spacing:-.015em; color:var(--text); }
h1 [data-testid="stIconMaterial"], h2 [data-testid="stIconMaterial"], h3 [data-testid="stIconMaterial"]{ color:var(--accent); }
p, li, label{ line-height:1.6; }
.kicker{
  display:inline-flex; align-items:center; gap:.5rem; font-size:.72rem; font-weight:700; letter-spacing:.14em;
  color:var(--accent); text-transform:uppercase; margin:0 0 .4rem; padding:.25rem .7rem;
  border-radius:999px; background:var(--accent-soft);
}
.muted{ color:var(--muted); }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{ background:var(--panel2); border-right:1px solid var(--line); }
[data-testid="stSidebarNav"] a{ border-radius:10px; padding:.45rem .75rem; margin:1px 0; transition:background .15s; }
[data-testid="stSidebarNav"] a:hover{ background:#EDF0F5; }
[data-testid="stSidebarNav"] a[aria-current="page"]{ background:var(--accent-soft); box-shadow:inset 3px 0 0 var(--accent); }
[data-testid="stSidebarNav"] a[aria-current="page"] span{ color:var(--accent) !important; font-weight:700; }
[data-testid="stNavSectionHeader"]{ font-size:.7rem !important; font-weight:800 !important; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted) !important; margin-top:.6rem; }
.side-brand{ display:flex; align-items:center; gap:.6rem; margin:.2rem 0 .1rem; }
.side-brand .logo{ width:36px; height:36px; border-radius:10px; display:grid; place-items:center;
  background:var(--accent); color:#fff; font-weight:900; font-size:1.05rem; font-family:'Nunito',sans-serif; }
.side-brand .name{ font-weight:800; font-size:1.02rem; line-height:1.15; }
.side-brand .sub{ color:var(--muted); font-size:.74rem; }
.side-user{ display:flex; align-items:center; gap:.5rem; font-weight:600; }
/* the Log out button shows only its icon next to the student number */
.st-key-logout button p{ position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }
.st-key-logout button{ padding:.35rem .55rem !important; min-height:0; }
.side-user .av{ width:30px; height:30px; border-radius:50%; display:grid; place-items:center; background:var(--accent-soft);
  color:var(--accent); font-weight:800; font-size:.8rem; }

/* ---------- buttons (flat, Khan-like) ---------- */
button[kind="primary"], button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primaryFormSubmit"]{
  background:var(--accent) !important; color:#fff !important; border:0 !important; font-weight:700 !important;
  border-radius:10px !important; box-shadow:none; transition:background .15s, transform .15s;
}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-primaryFormSubmit"]:hover{ background:var(--accent-dark) !important; }
button[kind="primary"]:disabled, button[data-testid="stBaseButton-primary"]:disabled{ background:#AFC6F3 !important; }
button[kind="secondary"], button[data-testid="stBaseButton-secondary"]{
  border-radius:10px !important; border:1px solid #CBD2DC !important; background-color:#fff !important; color:var(--text) !important;
  font-weight:600 !important; transition:border-color .15s, color .15s, background .15s;
}
button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover{
  border-color:var(--accent) !important; color:var(--accent) !important; background-color:var(--accent-soft) !important; }

/* ---------- inputs ---------- */
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"], textarea{ border-radius:10px !important; }
[data-testid="stFileUploaderDropzone"]{ border:1.5px dashed #AFC6F3; border-radius:14px; background:#F8FAFF; padding:1.4rem; }
[data-testid="stFileUploaderDropzone"]:hover{ background:var(--accent-soft); }

/* ---------- cards / panels ---------- */
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stVerticalBlock"][class*="st-key-"][style*="border"]{ border-radius:14px !important; }
[data-testid="stMetric"]{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:1rem 1.2rem; box-shadow:var(--shadow); }
[data-testid="stMetricLabel"] p{ color:var(--muted); font-size:.74rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
[data-testid="stMetricValue"]{ font-weight:800; color:var(--text); }
[data-testid="stExpander"] details{ border:1px solid var(--line) !important; border-radius:12px !important; background:var(--panel); }
[data-testid="stAlert"]{ border-radius:12px; }
[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:12px; overflow:hidden; }
[data-testid="stPageLink"] a{ color:var(--accent); font-weight:700; border-radius:8px; padding:.2rem .5rem; }
[data-testid="stPageLink"] a:hover{ background:var(--accent-soft); }
[data-testid="stPageLink"] a span{ color:var(--accent) !important; }
[data-testid="stTabs"] [aria-selected="true"]{ color:var(--accent) !important; }

/* ---------- chat: student on the right in a blue bubble, assistant on the left in a card ---------- */
[data-testid="stChatMessage"]{ background:var(--panel); border:1px solid var(--line); border-radius:16px;
  padding:.9rem 1.1rem; margin-bottom:.8rem; box-shadow:var(--shadow); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){
  flex-direction:row-reverse; margin-left:auto; max-width:78%; width:fit-content;
  background:var(--accent); border-color:var(--accent); border-radius:16px 16px 4px 16px; box-shadow:none; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] *,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] *{ color:#fff !important; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"]{ text-align:left; }
[data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])):not(:has([data-testid="chatAvatarIcon-user"])){
  border-radius:16px 16px 16px 4px; }
[data-testid="stChatInput"]{ border-radius:14px; }
[data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"],
[data-testid="stChatMessage"] [data-testid^="chatAvatarIcon"]{ display:none !important; }
/* the role in small caps above each message instead of the avatar */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"]::before,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarImage"]) [data-testid="stChatMessageContent"]::before{
  content:"Assistant"; display:block; font-size:.7rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase;
  color:var(--accent); margin-bottom:.35rem; }

/* ---------- icon chips (instead of emojis) ---------- */
.mi{ font-family:'Material Symbols Rounded'; font-weight:normal; font-style:normal; font-size:1.25em; line-height:1;
  display:inline-block; vertical-align:-.22em; letter-spacing:normal; text-transform:none; white-space:nowrap;
  -webkit-font-feature-settings:'liga'; font-feature-settings:'liga'; }

/* ---------- home / login ---------- */
.pill{ display:inline-block; padding:.12rem .65rem; margin-right:.4rem; border-radius:999px; font-size:.74rem; font-weight:700;
  color:var(--accent); background:var(--accent-soft); }
.hero{ position:relative; overflow:hidden; padding:2rem 2.2rem 1.7rem; margin:.4rem 0 1.4rem; border-radius:20px;
  background:linear-gradient(120deg,#0B4FC5 0%,#1865F2 55%,#4F8DF7 100%); color:#fff; }
.hero h1{ font-size:2.4rem !important; margin:0 0 .4rem !important; padding:0 !important; color:#fff !important; }
.hero h1 .hl{ color:#FFE27A; }
.hero p{ color:#E6EEFF; font-size:1.04rem; max-width:660px; margin:0; }
[class*="st-key-card_"]{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.1rem .7rem;
  transition:transform .18s, border-color .18s, box-shadow .18s; height:100%; box-shadow:var(--shadow); }
[class*="st-key-card_"]:hover{ transform:translateY(-3px); border-color:#AFC6F3; box-shadow:0 10px 26px rgba(24,101,242,.12); }
.ico{ width:44px; height:44px; display:grid; place-items:center; font-size:1.35rem; border-radius:12px;
  background:var(--accent-soft); color:var(--accent); margin-bottom:.6rem; }
.ico .material-symbols-rounded{ font-size:1.5rem; }
.card-title{ font-weight:700; font-size:1.02rem; margin-bottom:.2rem; }
.card-desc{ color:var(--muted); font-size:.88rem; line-height:1.5; margin-bottom:.4rem; min-height:2.6rem; }
.step{ background:var(--panel2); border:1px solid var(--line); border-radius:14px; padding:1.1rem 1.2rem; height:100%; }
.step b.n{ display:inline-grid; place-items:center; width:28px; height:28px; border-radius:50%; margin-bottom:.6rem;
  background:var(--accent); color:#fff; font-size:.85rem; }
.step .t{ font-weight:700; margin-bottom:.15rem; }
.step .d{ color:var(--muted); font-size:.86rem; }
.privacy{ display:flex; gap:.8rem; align-items:center; margin-top:1.4rem; padding:.85rem 1.2rem; border-radius:12px;
  background:var(--panel2); border:1px solid var(--line); color:var(--muted); font-size:.9rem; }

/* dashboard (Overview) */
.dash-hello{ font-size:1.9rem; font-weight:800; letter-spacing:-.02em; margin:.2rem 0 .1rem; }
.dash-sub{ color:var(--muted); margin-bottom:1rem; }
.streak{ display:flex; align-items:center; gap:.9rem; padding:1rem 1.2rem; border-radius:16px; border:1px solid #FFE1B3;
  background:linear-gradient(120deg,#FFF6E5,#FFFBF2); height:100%; }
.streak .flame{ font-size:2.2rem; line-height:1; color:#E8890C; }
.streak .flame .mi{ font-size:2.6rem; }
.streak .n{ font-family:'Nunito',sans-serif; font-size:1.9rem; font-weight:900; color:#E8890C; line-height:1; }
.streak .l{ color:#8A5A10; font-size:.85rem; font-weight:600; }
.course-tile{ border-radius:14px; padding:.9rem 1rem .4rem; color:#fff; margin-bottom:.4rem; }
.course-tile .c{ font-family:'Nunito',sans-serif; font-weight:900; font-size:1.25rem; }
.course-tile .m{ opacity:.9; font-size:.82rem; }
[class*="st-key-course_"]{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:.6rem .6rem .7rem;
  box-shadow:var(--shadow); }
.continue{ display:flex; gap:.8rem; align-items:center; }
.continue .i{ width:40px; height:40px; border-radius:10px; display:grid; place-items:center; background:var(--accent-soft); color:var(--accent); font-size:1.2rem; }
.continue .t{ font-weight:700; }
.continue .d{ color:var(--muted); font-size:.85rem; }

/* login (views/login.py): brand bar, light introduction, sign-in card */
.lg-bar{ display:flex; align-items:center; gap:.65rem; }
.lg-bar .logo{ width:38px; height:38px; border-radius:10px; display:grid; place-items:center; background:var(--accent);
  color:#fff; font-weight:900; font-size:1.1rem; font-family:'Nunito',sans-serif; }
.lg-bar .name{ font-weight:800; font-size:1.15rem; }
.lg-bar .tag{ margin-left:.4rem; padding:.18rem .6rem; border-radius:999px; background:var(--accent-soft); color:var(--accent);
  font-size:.7rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
.lg-intro{ padding:.5rem 1rem 1rem 0; }
.lg-eyebrow{ color:var(--accent); font-weight:800; font-size:.78rem; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.6rem; }
.lg-intro h1{ font-size:clamp(2.2rem, 3.4vw, 3.4rem) !important; line-height:1.12 !important; margin:0 0 .9rem !important;
  padding:0 !important; letter-spacing:-.03em; }
.lg-intro h1 .hl{ color:var(--accent); }
.lg-intro p.lead{ color:var(--muted); font-size:1.08rem; max-width:620px; margin-bottom:1.6rem; }
.lg-feat{ display:flex; gap:.9rem; align-items:flex-start; margin:1rem 0; max-width:600px; }
.lg-feat .i{ flex:0 0 44px; height:44px; border-radius:12px; display:grid; place-items:center; background:#fff;
  border:1px solid var(--line); color:var(--accent); box-shadow:var(--shadow); font-size:1.3rem; }
.lg-feat .t{ font-weight:700; }
.lg-feat .d{ color:var(--muted); font-size:.92rem; }
.lg-note{ margin-top:1.6rem; color:var(--muted); font-size:.86rem; display:flex; gap:.4rem; align-items:center; }
.st-key-login_box{ background:#fff; border:1px solid var(--line); border-radius:20px; padding:2rem 2rem 1.4rem;
  box-shadow:0 2px 4px rgba(33,36,44,.04), 0 18px 48px rgba(24,101,242,.10); }
.lg-title{ font-size:1.7rem; font-weight:800; letter-spacing:-.02em; }
.lg-sub{ color:var(--muted); margin:.2rem 0 1rem; }
/* the Log in / Register tabs as one segmented control */
.st-key-login_box [role="tablist"]{ background:var(--panel2); border-radius:12px; padding:4px; gap:4px; border:0 !important; }
.st-key-login_box [role="tab"]{ flex:1; justify-content:center; border-radius:9px; padding:.5rem 0 !important; margin:0 !important; }
.st-key-login_box [role="tab"][aria-selected="true"]{ background:#fff; box-shadow:0 1px 3px rgba(33,36,44,.12); }
.st-key-login_box [role="tab"] p{ font-weight:700; }
.st-key-login_box [data-baseweb="tab-highlight"], .st-key-login_box [data-baseweb="tab-border"],
.st-key-login_box [data-testid="stTabs"] [role="tablist"] + div:empty{ display:none !important; }
.st-key-login_box button[data-testid="stBaseButton-primaryFormSubmit"]{ padding:.65rem 0 !important; font-size:1rem; }
.lg-intro p.lead{ font-size:1.15rem; }
.lg-how{ padding:1.6rem 1.8rem; border-radius:20px; background:rgba(255,255,255,.75);
  border:1px solid var(--line); box-shadow:var(--shadow); }
.lg-how-title{ font-weight:800; font-size:.78rem; letter-spacing:.12em; text-transform:uppercase; color:var(--accent); margin-bottom:1rem; }
.lg-steps{ display:grid; grid-template-columns:repeat(3, 1fr); gap:1.4rem; }
.lg-step{ position:relative; padding-left:3.6rem; }
.lg-step .n{ position:absolute; left:0; top:0; width:2.6rem; height:2.6rem; border-radius:50%; display:grid; place-items:center;
  background:var(--accent); color:#fff; font-weight:900; font-family:'Nunito',sans-serif; font-size:1.1rem; }
.lg-step .i{ display:none; }
.lg-step .t{ font-weight:700; margin-bottom:.15rem; }
.lg-step .d{ color:var(--muted); font-size:.9rem; line-height:1.5; }
.lg-foot{ display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap; padding-top:1rem;
  border-top:1px solid var(--line); color:var(--muted); font-size:.8rem; }
.lg-foot .lg-bar .logo{ width:30px; height:30px; font-size:.9rem; border-radius:8px; }
.lg-foot .lg-bar .name{ font-size:.95rem; color:var(--text); }
@media (max-width: 900px){ .lg-steps{ grid-template-columns:1fr; } }

/* ---------- quiz (Question Generation): Duolingo-like ---------- */
.st-key-qg_setup button[kind="primary"], [class*="st-key-qcard_"] button[kind="primary"],
.st-key-qg_setup button[data-testid="stBaseButton-primary"], [class*="st-key-qcard_"] button[data-testid="stBaseButton-primary"]{
  background:var(--green) !important; border-radius:14px !important; box-shadow:0 4px 0 var(--green-dark) !important;
  text-transform:uppercase; letter-spacing:.06em; font-family:'Nunito','Inter',sans-serif !important; font-weight:900 !important;
  padding:.55rem 1.4rem !important; transform:translateY(0); transition:transform .06s, box-shadow .06s !important; }
.st-key-qg_setup button[kind="primary"]:hover, [class*="st-key-qcard_"] button[kind="primary"]:hover,
.st-key-qg_setup button[data-testid="stBaseButton-primary"]:hover, [class*="st-key-qcard_"] button[data-testid="stBaseButton-primary"]:hover{
  background:#61D90A !important; }
.st-key-qg_setup button[kind="primary"]:active, [class*="st-key-qcard_"] button[kind="primary"]:active,
.st-key-qg_setup button[data-testid="stBaseButton-primary"]:active, [class*="st-key-qcard_"] button[data-testid="stBaseButton-primary"]:active{
  transform:translateY(4px); box-shadow:0 0 0 var(--green-dark) !important; }
.st-key-qg_setup button[kind="primary"]:disabled, [class*="st-key-qcard_"] button[kind="primary"]:disabled,
.st-key-qg_setup button[data-testid="stBaseButton-primary"]:disabled, [class*="st-key-qcard_"] button[data-testid="stBaseButton-primary"]:disabled{
  background:#E5E5E5 !important; color:#AFAFAF !important; box-shadow:0 4px 0 #CFCFCF !important; }
[class*="st-key-qcard_"] button[kind="secondary"], [class*="st-key-qcard_"] button[data-testid="stBaseButton-secondary"]{
  border-radius:14px !important; border:2px solid #E5E5E5 !important; box-shadow:0 3px 0 #E5E5E5; font-weight:800 !important; }
[class*="st-key-qcard_"]{ border:2px solid var(--line); border-radius:18px; padding:1.1rem 1.3rem 1rem; background:var(--panel); margin-bottom:.4rem; }
.q-head{ display:flex; align-items:center; gap:.5rem; margin-bottom:.3rem; }
.q-num{ display:inline-grid; place-items:center; min-width:30px; height:30px; padding:0 .4rem; border-radius:10px;
  background:var(--accent); color:#fff; font-family:'Nunito',sans-serif; font-weight:900; }
.q-text{ font-size:1.12rem; font-weight:700; line-height:1.5; margin:.3rem 0 .6rem; }
/* answer options as big tiles (still the same radio widget underneath) */
[class*="st-key-qcard_"] [data-testid="stElementContainer"]:has([data-testid="stRadio"]),
[class*="st-key-qcard_"] [data-testid="stRadio"]{ width:100% !important; }
[class*="st-key-qcard_"] [role="radiogroup"]{ display:grid !important; grid-template-columns:1fr 1fr; gap:.6rem; margin-bottom:.6rem; width:100%; }
[class*="st-key-qcard_"] [role="radiogroup"] > div{ margin:0 !important; width:100%; }
[class*="st-key-qcard_"] [role="radiogroup"] > div > label{ width:100%; box-sizing:border-box; margin:0 !important; padding:.8rem 1rem;
  border:2px solid #E5E5E5; border-radius:14px; box-shadow:0 3px 0 #E5E5E5; background:#fff; cursor:pointer;
  transition:border-color .12s, background .12s; }
[class*="st-key-qcard_"] [role="radiogroup"] > div > label:hover{ border-color:#84D8FF; background:#F2FBFF; }
[class*="st-key-qcard_"] [role="radiogroup"] > div > label:has(input:checked){ border-color:#1CB0F6; background:#DDF4FF; box-shadow:0 3px 0 #1CB0F6; }
[class*="st-key-qcard_"] [role="radiogroup"] > div > label p{ font-weight:600; }
@media (max-width: 760px){ [class*="st-key-qcard_"] [role="radiogroup"]{ grid-template-columns:1fr; } }
.q-ok, .q-bad{ border-radius:14px; padding:.8rem 1rem; margin:.4rem 0 .6rem; font-weight:700; }
.q-ok{ background:var(--green-soft); color:#3C7A04; border:2px solid #B7E68A; }
.q-bad{ background:var(--red-soft); color:var(--red-dark); border:2px solid #F7B9BB; }
.q-ok .big, .q-bad .big{ font-family:'Nunito',sans-serif; font-weight:900; font-size:1.15rem; display:block; }
.quiz-progress{ display:flex; align-items:center; gap:.8rem; margin:.4rem 0 1rem; }
.quiz-progress .bar{ flex:1; height:16px; border-radius:999px; background:#E5E5E5; overflow:hidden; }
.quiz-progress .bar > div{ height:100%; border-radius:999px; background:var(--green);
  box-shadow:inset 0 -4px 0 rgba(0,0,0,.08), inset 0 4px 0 rgba(255,255,255,.25); transition:width .4s; }
.quiz-progress .t{ font-family:'Nunito',sans-serif; font-weight:900; color:var(--muted); white-space:nowrap; }
.quiz-done{ border-radius:18px; padding:1.1rem 1.3rem; margin-bottom:1rem; background:linear-gradient(120deg,#E9F8DC,#F4FCEB);
  border:2px solid #B7E68A; }
.quiz-done .big{ font-family:'Nunito',sans-serif; font-weight:900; font-size:1.4rem; color:#3C7A04; }

/* ---------- Interactive Learning ---------- */
.st-key-buddy_fab, .st-key-buddy_fab_open{
  position:fixed !important; right:.8rem; bottom:1.2rem; z-index:1000; width:auto !important;  /* sits in the page's right margin, clear of the chat input */
}
/* the bot picture itself is set as background-image by views/interactive.py */
.st-key-buddy_fab button, .st-key-buddy_fab_open button{
  width:68px; height:68px; min-height:0; border-radius:50% !important; padding:0 !important;
  border:0 !important; background-color:transparent !important;
  background-size:cover !important; background-position:center !important; background-repeat:no-repeat !important;
  box-shadow:0 8px 22px rgba(24,101,242,.28), 0 0 0 0 rgba(24,101,242,.45);
  animation:buddy-pulse 2.6s infinite; transition:transform .18s !important;
}
.st-key-buddy_fab button p, .st-key-buddy_fab_open button p{ opacity:0; font-size:1px !important; }
.st-key-buddy_fab button:hover, .st-key-buddy_fab_open button:hover{ transform:translateY(-3px) scale(1.05) !important; background-color:transparent !important; }
.st-key-buddy_fab_open button{ animation:none;
  box-shadow:0 0 0 4px rgba(24,101,242,.3), 0 8px 22px rgba(24,101,242,.28); }
@keyframes buddy-pulse{
  0%{ box-shadow:0 8px 22px rgba(24,101,242,.28), 0 0 0 0 rgba(24,101,242,.45); }
  70%{ box-shadow:0 8px 22px rgba(24,101,242,.28), 0 0 0 16px rgba(24,101,242,0); }
  100%{ box-shadow:0 8px 22px rgba(24,101,242,.28), 0 0 0 0 rgba(24,101,242,0); }
}
.buddy-title img{ width:30px; height:30px; vertical-align:middle; margin-right:.35rem; margin-top:-3px; }
.st-key-buddy_panel{ background:var(--panel2); position:sticky; top:3.5rem; }
.st-key-buddy_panel [data-testid="stChatMessage"]{ padding:.7rem .9rem; border-radius:14px; }
.buddy-title{ font-weight:800; font-size:1.15rem; }
.buddy-sub{ color:var(--muted); font-size:.78rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.buddy-empty{ color:var(--muted); font-size:.92rem; padding:1rem; border:1px dashed #CBD2DC; border-radius:14px; background:#fff; }
.doc-page{ white-space:pre-wrap; line-height:1.65; font-size:.95rem; color:var(--text);
  background:#fff; border:1px solid var(--line); border-radius:12px; padding:1.4rem 1.6rem; box-shadow:var(--shadow); }
.doc-page-title{ font-weight:700; color:var(--accent); margin-bottom:.8rem; font-size:.85rem;
  letter-spacing:.08em; text-transform:uppercase; white-space:normal; }

/* ---------- "beyond the course material" box (general knowledge / reasoning) ---------- */
[class*="st-key-beyond_"]{ background:#F3F0FF; border:1px solid #D9D0FF; border-left:4px solid #7C5CFA;
  border-radius:12px; padding:.75rem 1rem .4rem; margin-top:.6rem; }
.beyond-head{ color:#5B3FD9; font-weight:700; font-size:.9rem; }
.beyond-note{ color:var(--muted); font-size:.76rem; margin:-.1rem 0 .3rem; }

/* teaching board (core/board.py) on the light theme */
[class*="st-key-say_"]{ background:var(--accent-soft) !important; border-color:#C9DAFB !important; border-left-color:var(--accent) !important; }
[class*="st-key-given_"]{ background:#F8FAFF !important; border-color:#AFC6F3 !important; }
</style>
"""

def inject():
    st.markdown(CSS, unsafe_allow_html=True)

def kicker(t):
    st.markdown(f'<div class="kicker">{t}</div>', unsafe_allow_html=True)


def beyond_box(text: str, key: str, source: str = "course materials") -> None:
    """The part of an answer that does NOT come from the course material, in a light-blue box."""
    import streamlit as st
    from core.board import fix_markdown   # LaTeX \( \) / \[ \] -> $ / $$ so the formulas render
    with st.container(key=f"beyond_{key}"):
        st.markdown('<div class="beyond-head"><span class="mi">lightbulb</span> Beyond the course material</div>'
                    f'<div class="beyond-note">General knowledge and reasoning, not taken from your '
                    f'{source}. Double-check it with the lecture notes or your instructor.</div>',
                    unsafe_allow_html=True)
        st.markdown(fix_markdown(text))

