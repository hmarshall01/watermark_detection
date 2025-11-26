from __future__ import annotations
import os
import re
from typing import List, Tuple, Literal

Label = Literal["human", "ai_no_wm", "ai_wm"]

CONNECTORS = ["however", "therefore", "moreover"]
FORMAL = ["kindly", "your cooperation", "in order to"]
INTRO_STARTS = ["as a reminder", "to prevent", "as part of"]
AIISH_PHRASES = {
    "urgent action required", "verify your account", "click the link below",
    "within 24 hours", "kindly act without delay", "for security reasons",
    "confirm your identity", "immediate action required", "restricted access",
}
URL_RE = re.compile(r"(https?://|www\.)", re.I)

def _count_connectors(t: str) -> int:
    t = t.lower()
    return sum(t.count(w) for w in CONNECTORS)

def _count_punc(t: str) -> Tuple[int, int]:
    return t.count(","), t.count(":")

def _count_formal(t: str) -> int:
    t = t.lower()
    return sum(1 for p in FORMAL if p in t)

def _count_intro(t: str) -> int:
    t = t.lower()
    sentences = [s.strip() for s in t.split(".")]
    return sum(1 for s in sentences for start in INTRO_STARTS if s.startswith(start))

def _features(t: str) -> List[int]:
    com, col = _count_punc(t)
    return [_count_connectors(t), com, col, _count_formal(t), _count_intro(t)]

_model = None  

def _load_labeled_emails():
    base = "emails"
    triples = []
    for sub, lab in [("human", "human"), ("ai_no_wm", "ai_no_wm"), ("ai_wm", "ai_wm")]:
        path = os.path.join(base, sub)
        if not os.path.isdir(path):
            continue
        for fn in os.listdir(path):
            if fn.endswith(".txt"):
                full = os.path.join(path, fn)
                try:
                    with open(full, "r", encoding="utf-8") as f:
                        txt = f.read().strip()
                    if txt:
                        triples.append((fn, lab, txt))
                except Exception:
                    pass
    return triples

def _maybe_train():
    global _model
    emails = _load_labeled_emails()
    if len(emails) < 6:
        _model = None
        return
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        _model = None
        return
    inv = {"human": 0, "ai_no_wm": 1, "ai_wm": 2}
    X = [_features(txt) for (_, lbl, txt) in emails if lbl in inv]
    y = [inv[lbl] for (_, lbl, txt) in emails if lbl in inv]
    if not X:
        _model = None
        return
    _model = LogisticRegression(max_iter=2000)
    _model.fit(X, y)

_maybe_train()

def _heuristic(text: str) -> Label:
    s = (text or "").strip()
    if not s:
        return "human"
    low = s.lower()
    aiish = any(p in low for p in AIISH_PHRASES)
    conns = _count_connectors(s)
    has_url = bool(URL_RE.search(s))
    com, col = _count_punc(s)
    many_punct = (com + col) >= 3
    if aiish or conns >= 3:
        return "ai_wm" if (has_url or many_punct) else "ai_no_wm"
    return "human"


def classify_input(text: str) -> Label:
    s = (text or "").strip()
    if not s:
        return "human"
    if _model is not None:
        try:
            pred = int(_model.predict([_features(s)])[0])
            return {0: "human", 1: "ai_no_wm", 2: "ai_wm"}[pred]
        except Exception:
            pass
    return _heuristic(s)

if __name__ == "__main__":
    
    sample = "However, please verify your account within 24 hours via https://example.com"
    print("demo:", classify_input(sample))