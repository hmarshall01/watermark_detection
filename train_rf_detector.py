from __future__ import annotations
from pathlib import Path
import argparse
import sys
from typing import Tuple, List, Any, Dict

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.sparse import hstack

AI_POSITIVE = {
    "1","ai","ai_text","ai-generated","ai_generated","ai gen","ai-gen",
    "aiwm","ai_wm","ai-wm","nowm","ai-no-wm","ai_no_wm","ai nowm",
    "ai/no_wm","ai/nowm","ai/ai_wm","ai/aiwm","ai phishing","ai_email",
    "ai generated","ai-generated text","ai written","yes","true","y","t"
}
HUMAN_NEGATIVE = {
    "0","human","human_written","human-written","human text","human email",
    "human-written text","no","false","n","f"
}

def normalize_label(x: Any) -> int:
    """
    Map many common string labels to 0/1. Falls back to int cast when possible.
    Raises ValueError for unknowns to avoid silent mistakes.
    """
    if pd.isna(x):
        raise ValueError("Empty label encountered")
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        v = int(x)
        if v in (0, 1):
            return v
    s = str(x).strip().lower()
    if s in AI_POSITIVE:
        return 1
    if s in HUMAN_NEGATIVE:
        return 0
    # try clean int
    try:
        v = int(s)
        if v in (0, 1):
            return v
    except Exception:
        pass
    raise ValueError(f"Unrecognized label value: {x!r}. Expected one of {sorted(list(AI_POSITIVE|HUMAN_NEGATIVE))} or 0/1.")

def load_data(csv_path: Path, label_col: str) -> Tuple[List[str], List[int]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # allow alternative column names
    if "text" not in df.columns:
        raise ValueError(f"CSV must have a 'text' column. Found: {list(df.columns)}")
    if label_col not in df.columns:
        # try common fallbacks
        for alt in ["label", "labels", "target", "y", "class"]:
            if alt in df.columns:
                label_col = alt
                break
        else:
            raise ValueError(f"Label column '{label_col}' not found. Available: {list(df.columns)}")

    texts = df["text"].astype(str).tolist()
    raw_labels = df[label_col].tolist()
    labels = [normalize_label(v) for v in raw_labels]

    # small report
    import collections
    ctr = collections.Counter(labels)
    print(f"Label counts -> human(0): {ctr.get(0,0)}, ai(1): {ctr.get(1,0)}")
    return texts, labels

def train_and_save(train_csv: Path, out_path: Path, label_col: str = "label") -> None:
    X, y = load_data(train_csv, label_col)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    tfidf_word = TfidfVectorizer(
        ngram_range=(1,2), min_df=2, strip_accents="unicode",
        lowercase=True, max_features=25000
    )
    tfidf_char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3,5), min_df=2,
        strip_accents="unicode", lowercase=True, max_features=12000
    )

    Xtr_w = tfidf_word.fit_transform(Xtr)
    Xte_w = tfidf_word.transform(Xte)
    Xtr_c = tfidf_char.fit_transform(Xtr)
    Xte_c = tfidf_char.transform(Xte)

    Xtr_all = hstack([Xtr_w, Xtr_c]).tocsr()
    Xte_all = hstack([Xte_w, Xte_c]).tocsr()

    rf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )
    rf.fit(Xtr_all, ytr)

    print("\nValidation report:")
    print(classification_report(yte, rf.predict(Xte_all), digits=3))

    bundle: Dict[str, Any] = {"word": tfidf_word, "char": tfidf_char, "rf": rf}
    joblib.dump(bundle, out_path)
    print(f"\nSaved model -> {out_path.resolve()}")

def main():
    ap = argparse.ArgumentParser(description="Train RF on email CSV.")
    ap.add_argument("--train_csv", type=Path, required=True, help="CSV with columns: text,label (string or 0/1)")
    ap.add_argument("--out", type=Path, default=Path("rf_email_detector.pkl"))
    ap.add_argument("--label_col", type=str, default="label", help="Name of label column (default: label)")
    args = ap.parse_args()

    try:
        train_and_save(args.train_csv, args.out, args.label_col)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        print("Tips:", file=sys.stderr)
        print(" - Ensure CSV has columns: text,label (label can be human/ai or 0/1).", file=sys.stderr)
        print(" - Use --label_col to choose a different label column name.", file=sys.stderr)
        raise

if __name__ == "__main__":
    main()
