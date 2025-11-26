# app.py — PyScript-safe, matches your index.html (py-click on input#button)
import asyncio
import js  # DOM access in PyScript

# --- Lazy import helpers (avoid crashes if something isn't available) ---
def _try_import_logistic():
    try:
        from detector import classify_input
        return classify_input
    except Exception:
        return None

def _try_import_bert():
    try:
        from run_bert_model import classify_input_bert
        return classify_input_bert
    except Exception:
        return None

def _try_import_rf():
    try:
        from rf_detector import classify_input_rf
        return classify_input_rf
    except Exception:
        return None

def update_ui(event=None):
    # Read textarea
    ta = js.document.querySelector("#email_content")
    text = ta.value if ta else ""

    # --- Logistic (existing, from detector.py) ---
    classify_input = _try_import_logistic()
    if classify_input is not None:
        try:
            logistic = classify_input(text)
            log_num = 100 if logistic in ("ai_no_wm", "ai_wm") else 0
        except Exception:
            logistic, log_num = None, 0
    else:
        logistic, log_num = None, 0

    # --- BERT (optional; skip if torch/transformers missing in browser) ---
    classify_input_bert = _try_import_bert()
    if classify_input_bert is not None:
        try:
            bert_label = classify_input_bert(text)
            b_num = 100 if bert_label == "AI-generated" else 0
        except Exception:
            bert_label, b_num = None, 0
    else:
        bert_label, b_num = None, 0

    # --- RF (optional; skip if rf_email_detector.pkl not present) ---
    classify_input_rf = _try_import_rf()
    rf_num = None
    if classify_input_rf is not None:
        try:
            rf_out = classify_input_rf(text, model_path="rf_email_detector.pkl")
            if rf_out is not None:
                rf_label, _rf_prob = rf_out
                rf_num = 100 if rf_label == "AI-generated" else 0
        except Exception:
            rf_num = None

    # Average only detectors that produced a number
    votes = [log_num, b_num] + ([rf_num] if rf_num is not None else [])
    score = sum(votes) / len(votes) if votes else 0

    # Update gauge (ignore if helper missing)
    try:
        js.updateGauge()
    except Exception:
        pass

    # Result text (your thresholds/phrasing)
    if score <= 50:
        result_text = "It is unlikely the text entered is AI generated"
    elif score <= 80:
        result_text = "It is likely the text entered is AI generated"
    else:
        result_text = "AI generated text found"

    # Watermark notes (keep your behavior)
    if b_num == 100:
        result_text += " Sematic sentence structure watermark found. "
    if isinstance(logistic, str) and ("ai_wm" in logistic):
        result_text += " Connecting words/punctuation watermark found. "

    # Optional status to debug what's active
    status = []
    status.append("LogReg ✓" if classify_input else "LogReg ✕")
    status.append("BERT ✓" if classify_input_bert else "BERT ✕")
    status.append("RF ✓" if classify_input_rf and rf_num is not None else "RF ✕")
    result_text += " [" + "  ".join(status) + "]"

    # Write to page
    res = js.document.querySelector("#result")
    if res:
        res.innerText = result_text

# Note: NO manual addEventListener here
