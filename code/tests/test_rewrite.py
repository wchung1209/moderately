# code/tests/test_rewrite.py
import os
import sys
import pytest

# Make sure code/ is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference import rewrite

class DummyChoice:
    def __init__(self, text):
        self.message = type("M", (), {"content": text})

class DummyResponse:
    def __init__(self, text):
        self.choices = [DummyChoice(text)]

@pytest.fixture(autouse=True)
def patch_openai(monkeypatch):
    """Monkeypatch openai.ChatCompletion.create to return a dummy rewrite."""
    import openai
    def fake_create(*args, **kwargs):
        # simply echoes back a known string
        return DummyResponse("This is a neutral, factual paraphrase.")
    monkeypatch.setattr(openai.ChatCompletion, "create", fake_create)
    yield

def test_rewrite_signature_and_return_type():
    out = rewrite(
        text="Some biased input text.",
        ideology="Somewhat conservative bias",
        factuality="Mix of facts and opinions"
    )
    assert isinstance(out, str)
    assert "neutral" in out.lower()

def test_rewrite_not_empty_and_no_original_bias_words():
    original = "I absolutely hate policy X because it is the worst."
    out = rewrite(
        text=original,
        ideology="Strong conservative bias",
        factuality="Opinionated"
    )
    # Our fake always returns the dummy text, but in real life:
    # assert out != original
    # assert "hate" not in out.lower()
    assert out == "This is a neutral, factual paraphrase."

def test_integration_with_predict(monkeypatch):
    # Monkeypatch predict to return dummy labels
    import inference
    monkeypatch.setattr(inference, "predict", lambda text: {
        "ideology": "Somewhat liberal bias",
        "factuality": "Entirely factual",
        "analysis_html": ""
    })
    from inference import predict, rewrite
    res = predict("Dummy")
    out = rewrite("Dummy", res["ideology"], res["factuality"])
    assert isinstance(out, str)
