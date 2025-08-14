#!/usr/bin/env python3
# inference.py

import os
import pickle
import torch
import importlib.util
from transformers import DistilBertTokenizerFast, DistilBertConfig
import sys
import numpy as np
from functools import lru_cache
from openai import OpenAI
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None 

sys.modules['numpy._core'] = np.core
if hasattr(np.core, '_multiarray_umath'):
    sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath

# ─── CONFIGURE THESE PATHS ────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(__file__))   # project_root/
ARTIFACT_DIR  = os.path.join(BASE_DIR, "moderately_artifacts")
MODEL_DIR         = os.path.join(ARTIFACT_DIR, "model")
TOKENIZER_DIR     = os.path.join(ARTIFACT_DIR, "tokenizer")
IDEO_ENCODER_PATH = os.path.join(ARTIFACT_DIR, "ideo_encoder.pkl")
FACT_ENCODER_PATH = os.path.join(ARTIFACT_DIR, "fact_encoder.pkl")
TRAIN_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "2_Model_Training.py")

# ─── 1) Load LabelEncoders ─────────────────────────────────────────────────
with open(IDEO_ENCODER_PATH, "rb") as f:
    ideology_enc = pickle.load(f)
with open(FACT_ENCODER_PATH, "rb") as f:
    factuality_enc = pickle.load(f)

# ─── 2) Load Tokenizer & Config ────────────────────────────────────────────
tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_DIR)
config    = DistilBertConfig.from_pretrained(MODEL_DIR)

# ─── 3) Import MultiTaskDistilBert Class Dynamically ───────────────────────
spec = importlib.util.spec_from_file_location("model_training", TRAIN_SCRIPT_PATH)
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
MultiTaskDistilBert = mod.MultiTaskDistilBert  # the class definition

# ─── 4) Instantiate Model & Load Weights ──────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_ideo_labels = len(ideology_enc.classes_)
num_fact_labels = len(factuality_enc.classes_)

# We only need dummy class-weights for instantiation; they won't be used at inference
class_weights_ideo = torch.ones(num_ideo_labels)
class_weights_fact = torch.ones(num_fact_labels)

model = MultiTaskDistilBert(
    num_ideology_labels   = num_ideo_labels,
    num_factuality_labels = num_fact_labels,
    class_weights_ideo    = class_weights_ideo,
    class_weights_fact    = class_weights_fact,
    alpha = 0.5848983419474759,         # same α/β as training
    beta  = 1.0 - 0.5848983419474759
).to(device)

# Overwrite with fine-tuned weights
ckpt = torch.load(os.path.join(MODEL_DIR, "model_state_dict.pt"), map_location=device)
model.load_state_dict(ckpt)
model.eval()

# ─── 5) Inference Function ────────────────────────────────────────────────
IDEOLOGY_DESCRIPTIONS = {
    0: "strong conservative bias",
    1: "somewhat conservative bias",
    2: "moderate/neutral/no bias",
    3: "somewhat liberal bias",
    4: "strongly liberal bias"
}
IDEOLOGY_COLORS = {
    0: "red",
    1: "red",
    2: "grey",
    3: "blue",
    4: "blue"
}

FACTUALITY_DESCRIPTIONS = {
    0: "entirely factual claims",
    1: "mix of factual and opinionated claims",
    2: "entirely opinionated claims"
}
FACTUALITY_COLORS = {
    0: "green",
    1: "grey",
    2: "orange"
}

def predict(text: str):
    """
    Given a string of text, returns a dict:
      { "ideology": <label_str>, "factuality": <label_str> }
    """
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out = model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask
        )

    ideo_id = out["ideology_logits"].argmax(dim=-1).cpu().item()
    fact_id = out["factuality_logits"].argmax(dim=-1).cpu().item()

    ideo_lbl   = IDEOLOGY_DESCRIPTIONS[ideo_id]
    ideo_col   = IDEOLOGY_COLORS[ideo_id]
    fact_lbl   = FACTUALITY_DESCRIPTIONS[fact_id]
    fact_col   = FACTUALITY_COLORS[fact_id]

    sentence_html = (
        f"This text may contain "
        f"<span style='color:{ideo_col}'>{ideo_lbl}</span> "
        f"and is made up of <span style='color:{fact_col}'>{fact_lbl}</span>."
    )

    return {
        "ideology_id":      ideo_id,
        "factuality_id":    fact_id,
        "ideology":         ideo_lbl,
        "factuality":       fact_lbl,
        "ideology_color":   ideo_col,
        "factuality_color": fact_col,
        "analysis_html":    sentence_html
    }

# ─── 6) Quick Test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = "I believe climate change is a serious threat and the government should do more to enforce climate tax laws and penalize corporations."
    print("Input:", sample)
    print("Prediction:", predict(sample))


# ─── 7) Rewrite Function ───────────────────────────────────────────────────────────

# OPENAI_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_KEY:
#     raise RuntimeError("OPENAI_API_KEY not set. See README for setup.")
# client = OpenAI(api_key=OPENAI_KEY)

MODEL_REPO = os.getenv("HF_REPO", "wchung1209/moderately-bias-model")
MODEL_FILE = os.getenv("MODEL_FILENAME", "model_state_dict.pt")

def _resolve_weights_path() -> str:
    # 1) If present next to the code (dev/local), use it
    here = os.path.dirname(__file__)
    local_candidate = os.path.join(here, MODEL_FILE)
    if os.path.exists(local_candidate):
        return local_candidate

    # 2) Otherwise fetch from HF Hub into a writable cache dir
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub not installed; cannot fetch model weights.")

    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        token=os.getenv("HF_TOKEN"), 
        local_dir=os.path.expanduser("~/.cache/moderately"),
        local_dir_use_symlinks=False,
    )
    return path

@lru_cache(maxsize=1)
def load_model(device: torch.device):
    weights_path = _resolve_weights_path()
    state = torch.load(weights_path, map_location=device)
    model = build_model()  # your constructor that defines the same architecture
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model

@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    # Try Streamlit Secrets first, then env var
    key = None
    if st and hasattr(st, "secrets"):
        # supports either flat or namespaced secrets
        key = st.secrets.get("OPENAI_API_KEY") or \
              (st.secrets.get("openai", {}).get("api_key") if isinstance(st.secrets.get("openai"), dict) else None)
    key = key or os.getenv("OPENAI_API_KEY")

    if not key:
        # Raise a clear error only when you actually try to use the client
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it in Streamlit Cloud → Manage app → Settings → Secrets.\n"
            "Supported keys: OPENAI_API_KEY or [openai].api_key"
        )
    return OpenAI(api_key=key)

SYSTEM_PROMPT = """
You are an assistant that rewrites user‐supplied text to remove political bias and opinionated language while preserving only the factual claims. Do NOT add new facts or change meaning. Keep length, tone, and style as close as possible to the original.
"""

USER_TEMPLATE = """
Original text:
\"\"\"{text}\"\"\"

Detected political ideology bias: {ideology}
Detected factuality label: {factuality}

Please output only the rewritten, neutral, factual text.
"""

# def rewrite(text, ideology, factuality):
#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role":   "user", "content": USER_TEMPLATE.format(
#             text=text,
#             ideology=ideology,
#             factuality=factuality
#         )}
#     ]
#     resp = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=messages,
#         temperature=0.0,
#     )
#     return resp.choices[0].message.content.strip()

def rewrite(text: str, ideology: str, factuality: str) -> str:
    client = get_openai_client()
    resp = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                text=text, ideology=ideology, factuality=factuality
            )},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()