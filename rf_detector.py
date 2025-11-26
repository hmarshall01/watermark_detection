from __future__ import annotations
from typing import Optional, Tuple
from pathlib import Path

def classify_input_rf(
    text: str,
    model_path: str = "rf_email_detector.pkl",
    threshold: float = 0.60,  
) -> Optional[Tuple[str, float]]:
    """
    Return ("AI-generated"|"Human-written", prob_ai) or None if model missing.
    - Expects joblib bundle: {"word": TfidfVectorizer, "char": TfidfVectorizer, "rf": RandomForestClassifier}
    - Training labels: 1 = AI, 0 = Human
    - threshold applies to P(AI)
    """
    import joblib
    from scipy.sparse import hstack

    mp = Path(model_path)
    if not mp.exists():
        return None

    bundle = joblib.load(mp)
    word = bundle.get("word")
    char = bundle.get("char")
    rf   = bundle.get("rf")
    if word is None or char is None or rf is None:
        return None

    text = (text or "").strip()
    if not text:
        return ("Human-written", 0.0)

    Xw = word.transform([text])
    Xc = char.transform([text])
    X  = hstack([Xw, Xc]).tocsr()

    prob_ai = float(rf.predict_proba(X)[0, 1])
    label = "AI-generated" if prob_ai >= threshold else "Human-written"
    return (label, prob_ai)
