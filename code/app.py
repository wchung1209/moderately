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
            <!-- About Moderately -->
            Hi, my name is <b>Woojae Chung</b>, the creator of Moderately. Thank you for visiting! 
            <br><br>
            The motivation behind Moderately is clear. Today's world suffers from polarization and misinformation. 
            Internet and social media algorithms are designed to put you in an echo chamber where you are constantly exposed to similarly biased content and slowly polarizing your views. 
            And no, credible news organizations aren't free from blame either. This issue is most prominent in the context of politics.
            <br><br>
            Moderately aims to help users identify political bias and factuality in text. 
            As the user, you can paste a body of text, and Moderately will analyze it for political bias and factuality. 
            It will then rewrite the text to a neutral, factual form.
            <br><br>
            This is currently a working prototype and a passion project of mine. My goal is to expand its capabilities and make the model much more robust over time.
          </p>
          <div style="font-size: 1.25rem; font-weight: 700; margin: 0 0 .5rem;">Data</div>
          <p>
            <!-- Data -->
            The data I use come from two sources: <a href="https://huggingface.co/datasets/mediabiasgroup/BABE" target="_blank">BABE</a> dataset and the <a href="https://huggingface.co/datasets/cajcodes/political-bias" target="_blank">Political Bias</a> (short "PoliBias") dataset.
            The two datasets combined provide around 5,000 short text samples. The BABE dataset contains binary labels for the existence of political leaning and an ordinal factuality scale. The PoliBias dataset contains a 5-scale ideological bias only.
            <br><br>
            For consistency, I implemented a new labeling scheme that labels the political ideology bias in a 5-scale ordinal format and the factuality in a 3-scale ordinal format for the combined dataset.
            In this procedure, I use three "graders" to independently label each text sample, on top of the existing labels provided by the datasets for those that have them (BABE has factuality label, PoliBias has the ideology label).
            The graders include two prompt-engineered GPT-4 models that are given altered instructions albeit with the same goal, the original label from the dataset, and myself, a human grader. 
            In the case where all three graders agree, I accept the label. 
            In the case where there is a disagreement but only by a small margin (e.g. 1 vs 2 vs 2), I accept the majority vote. 
            In the case where all three graders disagree, my human grade overrides the others.
            <br><br>
            The final dataset is then split into train/val/test, and then the train/val dataset is fed into the model for the modeling stage.
            <br><br>
          </p>
          <div style="font-size: 1.25rem; font-weight: 700; margin: 0 0 .5rem;">Model</div>
          <p>
            <!-- Model -->
            The baseline model used is a distilBERT model. While other BERT-based models such as base BERT and RoBERTa were experimented with, I chose the distilBERT model due to its computational efficiency. 
            <br><br>
            A Multi-Head Classification model is built on top of the baseline model, which learns label-encoded ideology and factuality labels from the training data. This baseline model is then fine-tuned on the combined dataset.
            Parameters such as learning rate, batch size, and number of epochs were tuned using the Optuna library, and the best-performing model was selected based on the validation set performance.
            <br><br>
            You can find the comparison between the baseline distilBERT model and the fine-tuned model below. Note that the results shown below are from the validation set, as I am still keen on improving the model before evaluating on the test set.
          </p>
          
          <b>Ideology - Per Class and Overall F1 Scores</b>
            <table>
            <thead>
                <tr><th>Class</th><th>Baseline</th><th>Fine-tuned</th></tr>
            </thead>
            <tbody>
                <tr><td>0</td><td>0.503</td><td>0.525</td></tr>
                <tr><td>1</td><td>0.387</td><td>0.383</td></tr>
                <tr><td>2</td><td>0.745</td><td>0.753</td></tr>
                <tr><td>3</td><td>0.411</td><td>0.435</td></tr>
                <tr><td>4</td><td>0.557</td><td>0.578</td></tr>
                <tr><td><strong>Overall</strong></td><td><strong>0.520</strong></td><td><strong>0.533</strong></td></tr>
            </tbody>
            </table>

          <b>Factuality - Per Class and Overall F1 Scores</b>
            <table>
            <thead>
                <tr><th>Class</th><th>Baseline</th><th>Fine-tuned</th></tr>
            </thead>
            <tbody>
                <tr><td>0</td><td>0.823</td><td>0.826</td></tr>
                <tr><td>1</td><td>0.663</td><td>0.639</td></tr>
                <tr><td>2</td><td>0.784</td><td>0.777</td></tr>
                <tr><td><strong>Overall</strong></td><td><strong>0.756</strong></td><td><strong>0.748</strong></td></tr>
            </tbody>
            </table>  

          <div style="font-size: 1.25rem; font-weight: 700; margin: 0 0 .5rem;">Tech</div>
          <p>
            <!-- Tech -->
            <ul>
              <li>Frontend: Streamlit, Custom CSS</li>
              <li>Classifier Model: PyTorch, HuggingFace Transformers (DistilBERT)</li>
              <li>LLM (Rewrite): OpenAI Python SDK</li>
              <li>CI/Testing: Pytest</li>
              <li>Deployment: Streamlit Cloud</li>
            </ul>
          </p>

          <div style="font-size: 1.25rem; font-weight: 700; margin: 0 0 .5rem;">Next Steps</div>
          <p>
            <!-- Next Steps -->
            Importantly, Moderately is still a work in progress.
            I am actively working on improving the model's performance, expanding the dataset, and enhancing the capabilities and user experience within Moderately.
            <br><br>
            On the roadmap, I am planning to:
            <ul>
              <li>Bring in additional data. This is, in my opinion, the biggest bottleneck. Moderately is only trained from short 1-2 sentence text and with only a few thousand examples. I have some datasets that are much more comprehensive that I would like to train on.</li>
              <li>Improve model performance with a more robust model architecture. In particular, the model can use improvement on distinguishing the "Strongly" and "Somewhat" examples. Perhaps ordinal data + label encoding is not the most ideal choice. I would also like to compare with more language models.</li>
              <li>Add in new features! I would like to add support for web scraping API to support calling from a live web page.</li>
            </ul>
          </p>

          <p>
            The GitHub repository for Moderately is available at <a href="https://github.com/wchung1209/moderately" target="_blank">github.com/wchung1209/moderately</a>.
          </p>
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