# code/tests/test_inference.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from inference import (
    predict,
    IDEOLOGY_DESCRIPTIONS,
    FACTUALITY_DESCRIPTIONS,
    IDEOLOGY_COLORS,
    FACTUALITY_COLORS
)

@pytest.fixture(scope="module")
def short_text():
    return "The economy grew by 3% last quarter."

def test_predict_returns_dict(short_text):
    res = predict(short_text)
    assert isinstance(res, dict)

def test_keys_present(short_text):
    res = predict(short_text)
    for key in ("ideology", "factuality", "analysis_html"):
        assert key in res

def test_label_values(short_text):
    res = predict(short_text)
    # ideology should be one of the mapped strings
    assert res["ideology"] in IDEOLOGY_DESCRIPTIONS.values()
    # factuality should be one of the mapped strings
    assert res["factuality"] in FACTUALITY_DESCRIPTIONS.values()

def test_html_contains_spans(short_text):
    res = predict(short_text)
    html = res["analysis_html"]
    # must contain exactly one ideology span and one factuality span
    assert html.count("<span") >= 2
    assert "</span>" in html

def test_color_classes(short_text):
    res = predict(short_text)
    # extract colors from returned dict
    ideo_color = res.get("ideology_color", None)
    fact_color = res.get("factuality_color", None)
    # colors should match your mapping
    assert ideo_color in IDEOLOGY_COLORS.values()
    assert fact_color in FACTUALITY_COLORS.values()
