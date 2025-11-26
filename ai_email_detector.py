# AI vs Human Email Detector

# pip install pandas scikit-learn numpy

import pandas as pd # type: ignore
import numpy as np # type: ignore
import re
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import classification_report  # type: ignore
import joblib  # type: ignore

print("\n=== AI PHISHING EMAIL DETECTOR ===\n")

# 2. Load Training Data
print("Loading training dataset...")

train_df = pd.read_csv("training_data.csv")   # Must contain: text,label

# 3. Feature Engineering Functions

def politeness_score(text):
    polite_words = [
        "please", "kindly", "thank you", "regards", "sincerely",
        "appreciate", "request", "cooperate", "assistance"
    ]
    text = text.lower()
    return sum(text.count(w) for w in polite_words)


def robotic_phrase_score(text):
    robotic_phrases = [
        "as an ai", "i am unable", "i apologize", "it appears that",
        "furthermore", "moreover", "additionally", "in accordance",
        "as per your request"
    ]
    text = text.lower()
    return sum(text.count(p) for p in robotic_phrases)


def extract_features(text):
    return {
        "politeness": politeness_score(text),
        "robotic_score": robotic_phrase_score(text),
        "length": len(text),
        "avg_word_length": np.mean([len(w) for w in text.split()]) if text.split() else 0
    }

print("Extracting training features...")

feature_rows = []
for t in train_df["text"]:
    feature_rows.append(extract_features(t))

feature_df = pd.DataFrame(feature_rows)

# 4. Combine TF-IDF + Custom Features

tfidf = TfidfVectorizer(stop_words="english", max_features=300)
tfidf_matrix = tfidf.fit_transform(train_df["text"])

final_features = np.hstack([tfidf_matrix.toarray(), feature_df.values])
labels = train_df["label"]

# 5. Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    final_features, labels, test_size=0.2, random_state=42
)

print("Training model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate Model

print("\nEvaluating...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Save Model + Vectorizer

joblib.dump(model, "ai_detector_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\nModel saved successfully!")

# 8. Load Another File and Predict
print("\nLoading test emails for prediction...")

test_df = pd.read_csv("test_emails.csv")   # This file should only contain a column called "text"

print("Extracting features from test emails...")

test_feature_rows = []
for t in test_df["text"]:
    test_feature_rows.append(extract_features(t))

test_feature_df = pd.DataFrame(test_feature_rows)

# TF-IDF transform (using previously trained vectorizer)
tfidf_loaded = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix_test = tfidf_loaded.transform(test_df["text"])

# Combine Features
final_test_features = np.hstack([tfidf_matrix_test.toarray(), test_feature_df.values])

# Load model
model_loaded = joblib.load("ai_detector_model.pkl")

print("Predicting...")
predictions = model_loaded.predict(final_test_features)

# Add predictions to file
test_df["prediction"] = predictions
test_df.to_csv("predictions_output.csv", index=False)

print("\n===== PREDICTIONS COMPLETE =====")
print("Results saved to: predictions_output.csv")
print("Done!")
