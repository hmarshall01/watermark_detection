from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# load pretrained model
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)


model_path = "./ai_detector_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
device = "cpu"
model.to(device)
model.eval()

# application portion

# from files
def classify_folder(text_file):
    try:
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except Exception as e:
        return f"Error reading file: {e}"

    if not text:
        return "File is empty."
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    outputs = model(**inputs)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()

    return "AI-generated" if pred == 1 else "Human-written"

print(classify_folder("emails_to_check/test_ai_wm1.txt"))
print(classify_folder("emails_to_check/test_human1.txt"))
print(classify_folder("emails_to_check/test_ai_wm2.txt"))

# from user input
def classify_input_bert(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    outputs = model(**inputs)
    logits = outputs.logits

    pred = torch.argmax(logits, dim=1).item()

    return "AI-generated" if pred == 1 else "Human-written"