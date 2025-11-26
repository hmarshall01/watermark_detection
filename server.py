from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List
from flask import Flask, request, jsonify, send_from_directory
import math

# Soft imports
try:
    from detector import classify_input as classify_input_logreg
except Exception:
    classify_input_logreg = None

try:
    from run_bert_model import classify_input_bert
except Exception:
    classify_input_bert = None

try:
    from rf_detector import classify_input_rf
except Exception:
    classify_input_rf = None

app = Flask(__name__, static_folder=".", static_url_path="")

RF_THRESHOLD = 0.60  # stricter than 0.50

def _ai_votes(text: str) -> Dict[str, Optional[int]]:
    """
    Return advisory votes: 1=AI, 0=Human/neutral, None=unavailable.
    - LogReg: only 'ai_wm' counts as AI (ai_no_wm and human -> 0).
    - BERT: 'AI-generated' -> 1.
    - RF: prob_ai >= RF_THRESHOLD -> 1, else 0.
    """
    votes: Dict[str, Optional[int]] = {"logreg": None, "bert": None, "rf": None}

    # Logistic (advisory)
    if classify_input_logreg:
        try:
            lbl = classify_input_logreg(text)  # "human" | "ai_no_wm" | "ai_wm"
            votes["logreg"] = 1.0 if lbl == "ai_wm" else 0.0
        except Exception:
            votes["logreg"] = None

    # BERT (advisory)
    if classify_input_bert:
        try:
            lbl = classify_input_bert(text)    # "AI-generated" | "Human-written"
            votes["bert"] = 1.0 if lbl == "AI-generated" else 0.0
        except Exception:
            votes["bert"] = None

    # RF (advisory, stricter threshold)
    if classify_input_rf:
        try:
            res: Optional[Tuple[str, float]] = classify_input_rf(
                text, model_path="rf_email_detector.pkl", threshold=RF_THRESHOLD
            )
            if res is not None:
                lbl, prob_ai = res
                votes["rf"] = 1.0 if lbl == "AI-generated" else 0.0
            else:
                votes["rf"] = None
        except Exception:
            votes["rf"] = None

    return votes

def _advisory_consensus_score(votes: Dict[str, Optional[int]]) -> int:
    """
    Advisory rule:
      - If <2 models present -> 0.
      - Else majority: ai_votes >= floor(n/2)+1 -> 100 else 0.
    """
    present: List[int] = [v for v in votes.values() if isinstance(v, int)]
    n = len(present)
    if n < 2:
        return 0
    ai_votes = sum(present)
    needed = math.floor(n / 2) + 1
    return 100 if ai_votes >= needed else 0

def _verdict(score: int) -> str:
    return "Likely AI-generated" if score >= 100 else "Unlikely AI-generated"

# weighed based on reliability
def score_text(text: str) -> Dict[str, Any]:
    votes = _ai_votes(text)

    weights = {"bert":0.50, "logreg":0.40, "rf":0.10}
    total_weight = 0.0
    weighted_sum = 0.0

    for model_name, vote in votes.items():
        if vote is not None:
            w = weights[model_name]
            weighted_sum += vote * w
            total_weight += w

    prob = weighted_sum / total_weight if total_weight > 0 else  0.0

    verdict = "Likely AI-generated" if prob >= 0.6 else "Unlikely AI-generated"

    return {"score": prob * 100, "message": verdict}

# -------- Routes --------
@app.get("/")
def home():
    return send_from_directory(".", "index.html")

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    return jsonify(score_text(text))

if __name__ == "__main__":
    print("[server] http://127.0.0.1:5000/  mode=ADVISORY (2-of-N, LogReg=wm-only, RF thr=0.60)")
    app.run("127.0.0.1", 5000, debug=True)