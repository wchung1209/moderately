# code/app.py
import sys
import os
import streamlit as st

# Path to inference.py
sys.path.append(os.path.dirname(__file__))

# Import predict function
from inference import predict, rewrite

# -------- Streamlit UI -----------------------
# Paths & Config
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project_root/
LOGO_PATH = os.path.join(BASE_DIR, "misc", "logo.png")

st.set_page_config(page_title="Moderately – Political Bias & Factuality Checker", layout="wide")

# CSS Style 
st.markdown(
    """
    <style>
      :root {
        --max-width: 1100px;
        --text-color: #1f2937;
        --muted-text: #6b7280;
        --card-bg: #ffffff;
        --card-border: #e5e7eb;
        --input-bg: #ffffff;
        --btn-bg: #2563eb;
        --btn-fg: #ffffff;
        --btn-bg-hover: #1e40af;
        --footer-bg: #fafafa;
        --footer-border: #e5e7eb;
        }
        @media (prefers-color-scheme: dark) {
        :root {
            --text-color: #e5e7eb;
            --muted-text: #a3a3a3;
            --card-bg: #111827;
            --card-border: #374151;
            --input-bg: #0b1220;
            --btn-bg: #3b82f6;
            --btn-fg: #0b1220;
            --btn-bg-hover: #60a5fa;
            --footer-bg: #0b1220;
            --footer-border: #1f2937;
        }
        }

        /* Layout */
        .block-container {
        max-width: var(--max-width);
        padding-top: 3.0rem !important;  
        padding-bottom: 6rem;
        color: var(--text-color);
        }
        .center { text-align: center; }

        /* Headings */
        .h1like {
        font-weight: 800; line-height: 1.1; margin: .25rem 0 .25rem 0;
        font-size: clamp(1.8rem, 2.6vw + 1rem, 2.7rem); color: var(--text-color);
        }
        .tagline { font-size: 1rem; color: var(--muted-text); margin: .25rem 0 1.1rem 0; }

        /* Tabs (top navigation, left-aligned, visible labels) */
        .stTabs { margin-top: .25rem; }
        .stTabs [role="tablist"] {
        display: flex; justify-content: flex-start; gap: 1.25rem;
        }
        .stTabs [role="tab"] {
        background: transparent; border: 0; padding: .4rem .2rem;
        }
        .stTabs [role="tab"] * {
        /* Ensure label text is actually painted */
        color: var(--text-color) !important;
        -webkit-text-fill-color: var(--text-color) !important;
        text-shadow: none; opacity: 1;
        font-weight: 700;
        margin: 0;
        }
        .stTabs [role="tab"][aria-selected="true"] {
        border-bottom: none !important;
        }
        .stTabs [role="tab"][aria-selected="true"] * {
        color: var(--btn-bg) !important;
        -webkit-text-fill-color: var(--btn-bg) !important;
        }
        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="highlight"] {
        background: var(--btn-bg) !important;  
        height: 2px !important;                
        bottom: 0 !important;           
        opacity: 1 !important;
        z-index: 2 !important;
        }

        /* Cards & content */
        .card {
        border: 1px solid var(--card-border); border-radius: 12px; padding: 1rem 1.1rem;
        background: var(--card-bg); box-shadow: 0 1px 2px rgba(0,0,0,.04); margin: .5rem 0 0;
        }
        .card.emphasis {
        border-color: rgba(37,99,235,.35);
        background: linear-gradient(0deg, rgba(37,99,235,.07), rgba(37,99,235,.07)), var(--card-bg);
        }
        .analysis span { font-weight: 700; }

        /* Textarea */
        textarea {
        background: var(--input-bg) !important; color: var(--text-color) !important;
        border-radius: 10px !important; border: 1px solid var(--card-border) !important;
        }

        /* Buttons */
        .stButton > button {
        background: var(--btn-bg) !important; color: var(--btn-fg) !important; border: none !important;
        border-radius: 10px !important; font-weight: 700 !important; padding: .62rem 1.2rem !important;
        box-shadow: 0 2px 6px rgba(0,0,0,.08) !important;
        }
        .stButton > button:hover {
        background: var(--btn-bg-hover) !important; transition: background .15s ease-in-out;
        }

        /* Hide the "View fullscreen" button on images (prevents enlarging the logo) */
        button[title="View fullscreen"] { display: none !important; }

        /* Footer */
        .footer {
        position: fixed; left: 0; right: 0; bottom: 0; width: 100%; background: var(--footer-bg);
        border-top: 1px solid var(--footer-border); padding: .6rem 1rem; font-size: 0.85rem;
        color: var(--muted-text); text-align: center; z-index: 999;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper functions
def render_header(title: str, subtitle_html: str):
    col = st.columns([1, 2, 1])[1]  # centered mid column
    if os.path.exists(LOGO_PATH):
        col.image(LOGO_PATH, use_container_width=False)
    col.markdown(f"<div class='center h1like'>{title}</div>", unsafe_allow_html=True)
    if subtitle_html:
        col.markdown(f"<div class='center tagline'>{subtitle_html}</div>", unsafe_allow_html=True)

def require_openai_key_ui() -> str:
    key = st.secrets.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")
    if not key:
        st.error("OpenAI API key is not set. Add it to Streamlit secrets or set OPENAI_API_KEY in your environment.")
        st.stop()
    return key

# Ensure our session_state keys exist
if "result" not in st.session_state:
    st.session_state.result = None
if "rewrite_text" not in st.session_state:
    st.session_state.rewrite_text = None


home_tab, about_tab = st.tabs(["Home", "About Moderately"])

# ---------- HOME ----------
with home_tab:
    render_header(
        "Political Bias & Factuality Checker",
        "Enter 1–4 sentences and click <b>Analyze Political and Factuality Bias</b> to see the political bias and factuality assessment. Then, click <b>Rewrite Text with No Bias!</b> to rewrite the text to a neutral, factual form.",
    )

    st.markdown("**Your Text**")
    user_input = st.text_area(" ", height=160, label_visibility="collapsed",
                              placeholder="Paste your 1–4 sentences here…")

    if st.button("Analyze Political and Factuality Bias"):
        if not user_input.strip():
            st.error("Please enter some text first.")
        else:
            st.session_state.result = predict(user_input)
            st.session_state.rewrite_text = None

    if st.session_state.result is not None:
        res = st.session_state.result
        st.markdown(
            f"<div class='card emphasis analysis'>{res['analysis_html']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        if st.button("Rewrite Text with No Bias!"):
            with st.spinner("Rewriting…"):
                st.session_state.rewrite_text = rewrite(
                    text=user_input,
                    ideology=res["ideology"],
                    factuality=res["factuality"]
                )

        if st.session_state.rewrite_text:
            st.subheader("🔄 Neutral, Factual Rewrite")
            st.markdown(
                f"<div class='card'>{st.session_state.rewrite_text}</div>",
                unsafe_allow_html=True,
            )

# ---------- ABOUT ----------
with about_tab:
    render_header("About Moderately", "")
    st.markdown(
        """
        <div class="card">
          <div style="font-size: 1.25rem; font-weight: 700; margin: 0 0 .5rem;">About Moderately</div>
          <p>
            <!-- Replace this with your own blurb -->
            Moderately analyzes short passages (1–4 sentences) for political ideology bias and factuality,
            then optionally rewrites the text to a neutral, factual form. Use this page to describe your
            mission, data sources, limitations, and privacy approach.
          </p>
          <ul>
            <li>What it does well</li>
            <li>Where it may err (limitations)</li>
            <li>How your data is used</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Footer ----------
st.markdown(
    """
    <div class="footer">
      <em>*Moderately can make mistakes. Moderately does not check the facts of the written text, but only analyzes whether the text is attempting to make a non-opinion-based statement. Use the results at your own discretion.</em>
    </div>
    """,
    unsafe_allow_html=True,
)