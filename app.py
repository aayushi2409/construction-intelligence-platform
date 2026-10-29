"""Multimodal Construction Intelligence Platform
Skill Wallet Capstone - Project 1
Run:  streamlit run app.py
"""
import json
import re
import urllib.parse

import streamlit as st
from PIL import Image

try:
    import google.generativeai as genai
except ImportError:  # app still runs in demo mode
    genai = None

st.set_page_config(page_title="Construction Intelligence Platform", page_icon="🏗️", layout="wide")

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
.hero{background:linear-gradient(120deg,#0f2027,#2c5364);padding:1.4rem 1.8rem;border-radius:14px;color:#fff;margin-bottom:1rem}
.hero h1{margin:0;font-size:1.9rem}.hero p{margin:.3rem 0 0;opacity:.85}
.card{background:#f7f9fc;border:1px solid #e1e6ef;border-radius:12px;padding:1rem;text-align:center;color:#1b2430}
.card .v{font-size:1.7rem;font-weight:700}.card .l{font-size:.8rem;color:#5a6b80;text-transform:uppercase}
.pill{display:inline-block;padding:.2rem .8rem;border-radius:999px;font-weight:600;font-size:.85rem;color:#fff}
.alert{border-left:6px solid #e67e22;background:#fff6ec;color:#3a2a14;padding:.8rem 1rem;border-radius:8px;margin:.4rem 0}
</style>""", unsafe_allow_html=True)

SEV_COLOR = {"Low": "#27ae60", "Medium": "#f39c12", "High": "#e67e22", "Critical": "#c0392b",
             "COMPLIANT": "#27ae60", "NEEDS IMPROVEMENT": "#f39c12", "NON-COMPLIANT": "#c0392b"}


def pill(text):
    return f"<span class='pill' style='background:{SEV_COLOR.get(str(text), '#2c5364')}'>{text}</span>"


def card(label, value):
    return f"<div class='card'><div class='v'>{value}</div><div class='l'>{label}</div></div>"


def inr(v):
    """Format rupees as Lakhs / Crores."""
    return f"₹{v/1e7:,.2f} Cr" if v >= 1e7 else f"₹{v/1e5:,.2f} Lakhs"


# ---------------------------------------------------------------- AI layer
def get_model():
    key = st.session_state.get("api_key", "").strip()
    if not key or genai is None:
        return None
    genai.configure(api_key=key)
    return genai.GenerativeModel(st.session_state.get("model_name", "gemini-2.0-flash"))


def extract_json(text):
    text = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def analyze_image(prompt, image, demo_result):
    """Multimodal call with graceful fallback to demo data."""
    model = get_model()
    if model is None:
        return demo_result, True
    try:
        resp = model.generate_content([prompt, image])
        return extract_json(resp.text), False
    except Exception as e:  # network, quota, bad JSON ...
        st.warning(f"AI call failed ({e}). Showing demonstration output instead.")
        return demo_result, True


SAFETY_PROMPT = """You are a certified construction safety officer. Audit this site photo for PPE compliance
under OSHA 1926 and Indian standards (IS 456 / BOCW Act). Check helmets, high-vis vests, steel-toe boots, safety harnesses.
Return ONLY valid JSON:
{"safety_score":0-100,"overall_status":"COMPLIANT|NEEDS IMPROVEMENT|NON-COMPLIANT",
"workers_detected":int,
"ppe_detection":{"helmet":bool,"high_vis_vest":bool,"steel_toe_boots":bool,"safety_harness":bool},
"hazards":[{"hazard":str,"severity":"Low|Medium|High|Critical","remedy":str}],
"summary":str}"""

CRACK_PROMPT = """You are a structural engineer. Analyse this concrete/wall surface photo.
Classify defect_type as one of: Hairline, Settlement, Structural Shear, Thermal Expansion, Spalling.
Return ONLY valid JSON:
{"defect_type":str,"estimated_width_mm":float,"severity_score":0-100,"urgency":"Low|Medium|High|Critical",
"root_causes":[str],"recommended_repairs":[str],"monitoring_advice":str,"summary":str}"""

DEMO_SAFETY = {"safety_score": 62, "overall_status": "NEEDS IMPROVEMENT", "workers_detected": 3,
    "ppe_detection": {"helmet": True, "high_vis_vest": True, "steel_toe_boots": False, "safety_harness": False},
    "hazards": [
        {"hazard": "Work at height without harness", "severity": "Critical", "remedy": "Stop work; issue full-body harness anchored to a rated point (OSHA 1926.502)."},
        {"hazard": "Workers in sandals / non-safety footwear", "severity": "High", "remedy": "Mandate IS 15298 steel-toe safety boots at site entry."},
        {"hazard": "Loose rebar and debris on walkway", "severity": "Medium", "remedy": "Daily housekeeping; cap exposed rebar ends."}],
    "summary": "DEMO MODE: partial PPE compliance. Fall-protection is the top priority."}

DEMO_CRACK = {"defect_type": "Settlement", "estimated_width_mm": 3.5, "severity_score": 68, "urgency": "High",
    "root_causes": ["Differential foundation settlement", "Poor soil compaction", "Water seepage near footing"],
    "recommended_repairs": ["Install crack gauges / tell-tales", "Epoxy injection once movement stabilises",
                            "Structural re-grouting and underpinning if movement continues"],
    "monitoring_advice": "Record width weekly for 8 weeks before sealing.",
    "summary": "DEMO MODE: diagonal stepped crack typical of settlement; needs engineer inspection."}

DEMO_CHAT = {
    "curing": "Cure concrete for at least 7 days (OPC) and 10+ days for blended cement per IS 456 - keep surfaces continuously moist.",
    "cement": "Typical M25 mix ratio is 1:1:2 (cement:sand:aggregate). Use ~0.4-0.45 water-cement ratio.",
    "steel": "Fe500D TMT is standard in India. Provide min. cover of 25 mm for slabs, 40 mm for columns (IS 456).",
    "safety": "Mandatory PPE: helmet, high-vis vest, safety boots, gloves, and full-body harness above 2 m.",
}


def chat_reply(history, user_msg):
    model = get_model()
    if model is None:
        for k, v in DEMO_CHAT.items():
            if k in user_msg.lower():
                return v + "  _(demo mode)_"
        return "Demo mode: ask about curing, cement, steel or safety, or add a Gemini API key for full answers."
    sys = ("You are a senior Indian civil engineering site assistant. Be concise and practical; cite IS codes "
           "(IS 456, IS 800, IS 875) where relevant. Costs in INR.\n")
    convo = "\n".join(f"{r.upper()}: {c}" for r, c in history[-10:])
    try:
        return model.generate_content(f"{sys}{convo}\nUSER: {user_msg}\nASSISTANT:").text
    except Exception as e:
        return f"AI error: {e}"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🔐 Configuration")
    st.session_state["api_key"] = st.text_input("Google Gemini API key", type="password",
                                                help="Leave blank for demonstration mode.")
    st.session_state["model_name"] = st.selectbox("Model", ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"])
    if st.session_state["api_key"]:
        st.success("Live AI mode")
    else:
        st.info("Demonstration mode")
    st.markdown("---")
    st.subheader("🧭 Architecture")
    st.markdown("""
1. **Streamlit UI** - 4 feature tabs  
2. **Pillow** - image handling  
3. **Gemini multimodal** - JSON contracts  
4. **Rule engine** - cost & valuation (INR)  
5. **Pollinations** - concept renders  

No database; fully stateless for Streamlit Cloud.""")

st.markdown("<div class='hero'><h1>🏗️ Multimodal Construction Intelligence Platform</h1>"
            "<p>PPE safety audit • Crack diagnostics • Cost & valuation • Weather-aware site assistant</p></div>",
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🦺 Site Safety & PPE", "🧱 Crack Diagnostics", "💰 Planning & Valuation", "🌦️ Site Assistant"])

# ---------------------------------------------------------------- tab 1
with tab1:
    up = st.file_uploader("Upload construction site photo", type=["jpg", "jpeg", "png"], key="safe")
    if up:
        img = Image.open(up).convert("RGB")
        c1, c2 = st.columns([1, 1.3])
        c1.image(img, use_container_width=True)
        if c2.button("Run PPE & Hazard Audit", type="primary"):
            with st.spinner("Analysing site..."):
                st.session_state["safety"], demo = analyze_image(SAFETY_PROMPT, img, DEMO_SAFETY)
    r = st.session_state.get("safety")
    if r:
        m = st.columns(3)
        m[0].markdown(card("Safety Score", f"{r.get('safety_score', 0)}/100"), unsafe_allow_html=True)
        m[1].markdown(card("Workers Detected", r.get("workers_detected", "-")), unsafe_allow_html=True)
        m[2].markdown(card("Status", pill(r.get("overall_status", "-"))), unsafe_allow_html=True)
        st.progress(min(max(int(r.get("safety_score", 0)), 0), 100) / 100)
        st.subheader("PPE Checklist")
        cols = st.columns(4)
        for col, (k, v) in zip(cols, r.get("ppe_detection", {}).items()):
            col.markdown(f"{'✅' if v else '❌'} **{k.replace('_', ' ').title()}**")
        st.subheader("Identified Hazards")
        for h in r.get("hazards", []):
            st.markdown(f"{pill(h['severity'])} **{h['hazard']}**<br>🔧 {h['remedy']}", unsafe_allow_html=True)
            st.write("")
        st.info(r.get("summary", ""))
        st.download_button("⬇️ Download JSON report", json.dumps(r, indent=2), "safety_report.json", "application/json")

# ---------------------------------------------------------------- tab 2
with tab2:
    up = st.file_uploader("Upload concrete / wall surface photo", type=["jpg", "jpeg", "png"], key="crack")
    if up:
        img = Image.open(up).convert("RGB")
        c1, c2 = st.columns([1, 1.3])
        c1.image(img, use_container_width=True)
        if c2.button("Diagnose Defect", type="primary"):
            with st.spinner("Running structural diagnosis..."):
                st.session_state["crack"], _ = analyze_image(CRACK_PROMPT, img, DEMO_CRACK)
    r = st.session_state.get("crack")
    if r:
        m = st.columns(4)
        m[0].markdown(card("Defect Type", r.get("defect_type", "-")), unsafe_allow_html=True)
        m[1].markdown(card("Crack Width", f"{r.get('estimated_width_mm', '-')} mm"), unsafe_allow_html=True)
        m[2].markdown(card("Severity Score", f"{r.get('severity_score', '-')}/100"), unsafe_allow_html=True)
        m[3].markdown(card("Urgency", pill(r.get("urgency", "-"))), unsafe_allow_html=True)
        a, b = st.columns(2)
        a.subheader("Root Causes")
        for x in r.get("root_causes", []):
            a.markdown(f"- {x}")
        b.subheader("Recommended Repairs")
        for x in r.get("recommended_repairs", []):
            b.markdown(f"- {x}")
        st.warning(f"**Monitoring:** {r.get('monitoring_advice', '')}")
        st.info(r.get("summary", ""))
        st.download_button("⬇️ Download JSON report", json.dumps(r, indent=2), "crack_report.json", "application/json")

# ---------------------------------------------------------------- tab 3
RATES = {"Residential Villa": 2400, "Apartment Block": 2100, "Commercial Building": 2800, "Industrial Warehouse": 1600}
MARKET = {"Residential Villa": 7500, "Apartment Block": 6800, "Commercial Building": 9500, "Industrial Warehouse": 4200}
TIER = {"Tier 1 (Mumbai, Delhi, Bengaluru)": (1.35, 2.2), "Tier 2 (Indore, Jaipur, Pune)": (1.0, 1.0),
        "Tier 3 (Smaller cities & towns)": (0.82, 0.55)}
MILESTONES = [(10, "Site prep & excavation"), (25, "Foundation"), (45, "Plinth & columns"), (65, "Slabs & structure"),
              (80, "Masonry & MEP"), (95, "Finishing"), (100, "Handover")]

with tab3:
    c1, c2 = st.columns(2)
    stype = c1.selectbox("Structure type", list(RATES))
    area = c1.number_input("Built-up area (sq ft)", 200, 500000, 2000, step=100)
    tier = c2.selectbox("Location tier", list(TIER))
    pct = c2.slider("Completion %", 0, 100, 40)
    cost_mult, val_mult = TIER[tier]
    budget = area * RATES[stype] * cost_mult
    capex = budget * pct / 100
    market_full = area * MARKET[stype] * val_mult
    current_val = market_full * (0.25 + 0.75 * pct / 100) if pct else market_full * 0.25  # land + built value
    steel_kg, cement_bags = area * 4.0, area * 0.45
    m = st.columns(4)
    m[0].markdown(card("Total Budget", inr(budget)), unsafe_allow_html=True)
    m[1].markdown(card("CapEx Incurred", inr(capex)), unsafe_allow_html=True)
    m[2].markdown(card("Current Asset Value", inr(current_val)), unsafe_allow_html=True)
    m[3].markdown(card("Value on Completion", inr(market_full)), unsafe_allow_html=True)
    n = st.columns(2)
    n[0].markdown(card("Steel Required", f"{steel_kg/1000:,.1f} tonnes"), unsafe_allow_html=True)
    n[1].markdown(card("Cement Required", f"{cement_bags:,.0f} bags"), unsafe_allow_html=True)
    st.write("")
    st.progress(pct / 100, text=f"Project progress: {pct}%")
    done = [name for p, name in MILESTONES if pct >= p]
    nxt = next((name for p, name in MILESTONES if pct < p), "Completed")
    st.caption(f"✅ Done: {', '.join(done) or 'None yet'}  |  ⏭️ Next: {nxt}")
    st.caption("Estimates use indicative rate cards; verify with local quantity surveyors.")

    st.subheader("🎨 Conceptual 3D Rendering (Pollinations AI)")
    style = st.text_input("Architectural style", "modern Indian contemporary, glass facade, landscaped garden")
    prompt = f"photorealistic 3D architectural rendering of a {stype.lower()}, {style}, golden hour, 8k"
    st.code(prompt, language=None)
    if st.button("Generate render"):
        url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}?width=1024&height=640&nologo=true"
        st.image(url, caption="AI concept render (illustrative)", use_container_width=True)

# ---------------------------------------------------------------- tab 4
ALERTS = {
    "Normal": [("Standard operations", "Maintain routine toolbox talk and PPE checks.")],
    "High Wind": [("Suspend crane operations", "Stop tower/mobile crane lifts above 36 km/h wind speed."),
                  ("Secure loose materials", "Tie down sheeting, scaffolding boards and formwork."),
                  ("Halt work at height", "No work on scaffolds or edges during gusts.")],
    "Heavy Rain": [("Electrical hazard mitigation", "De-energise temporary power; use ELCB/RCCB-protected, IP-rated connections."),
                   ("Concrete protection", "Cover fresh pours with tarpaulin; avoid placing concrete in rain."),
                   ("Excavation safety", "Check trench shoring and pump out water.")],
    "Extreme Heat": [("Concrete curing protocol", "Pour early morning; use chilled water, wet-hessian curing and 3x/day sprinkling per IS 456."),
                     ("Worker heat stress", "Mandatory hydration breaks every 45 min; shaded rest area; ORS available."),
                     ("Limit heavy labour", "Avoid outdoor strenuous work 12:00-16:00.")],
}

with tab4:
    st.subheader("Weather-Aware Daily Protocol")
    w = st.radio("Today's condition", list(ALERTS), horizontal=True)
    for t, d in ALERTS[w]:
        st.markdown(f"<div class='alert'><b>⚠️ {t}</b><br>{d}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("💬 Construction Engineering Assistant")
    if "chat" not in st.session_state:
        st.session_state["chat"] = [("assistant", "Hello! Ask me about mixes, curing, rebar, safety or costing.")]
    for role, text in st.session_state["chat"]:
        st.chat_message(role).write(text)
    if q := st.chat_input("Ask a site engineering question..."):
        st.session_state["chat"].append(("user", f"[Weather: {w}] {q}"))
        st.chat_message("user").write(q)
        with st.spinner("Thinking..."):
            ans = chat_reply(st.session_state["chat"], q)
        st.session_state["chat"].append(("assistant", ans))
        st.chat_message("assistant").write(ans)
