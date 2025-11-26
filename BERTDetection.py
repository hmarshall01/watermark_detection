from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

import torch
import evaluate
import accelerate
import numpy as np
from transformers import DataCollatorWithPadding
from datasets import Dataset

# PURPOSE: BERT is a very strong transformer model that this class will
#          finetune the pretrained model to continue to become more accurate
#           the more this program is used. It investigates sentence structure


# prevent my device from crashing during training
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# defining the pretrained model used
model = "distilbert-base-uncased"

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model)
datamodel = AutoModelForSequenceClassification.from_pretrained(model, num_labels=2)

#checkpoint
datamodel.gradient_checkpointing_enable()

# loading large preset database from hugging face and prepping dataset
# human = 0, ai-generated = 1
datasets = load_dataset("dmitva/human_ai_generated_text")
print(datasets)

all_texts = []
for row in datasets['train']:
    all_texts.append({'text': row['human_text'], 'label': 0})
    all_texts.append({'text': row['ai_text'], 'label': 1})

features = Dataset.from_list(all_texts[:20000])  # 20k samples

split = features.train_test_split(test_size=0.1)  # 10% test
train_dataset = split['train']
test_dataset = split['test']

print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

# tokenizing data
def tok_data(ex):
    return tokenizer(ex['text'], truncation=True, max_length=128)

tokenized_train = train_dataset.map(tok_data, batched=True)
tokenized_test = test_dataset.map(tok_data, batched=True)

tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_train.set_format("torch")

tokenized_test = tokenized_test.rename_column("label", "labels")
tokenized_test.set_format("torch")

# data loaders
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# optimization and training settings

training_args = TrainingArguments(
    output_dir="./ai_detector_model",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    dataloader_num_workers=0,
    no_cuda=False
)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return metric.compute(predictions=preds, references=labels)

# actual training happens here
trainer = Trainer(
    model=datamodel,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# evaluation
results = trainer.evaluate()
print(results)

# saving the model
trainer.save_model("./ai_detector_model")
tokenizer.save_pretrained("./ai_detector_model")

# again to make sure my computer doesn't break and run in eval mode
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
datamodel.to(device)
datamodel.eval()

model_path = "./ai_detector_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
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
    inputs = tokenizer(text_file, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = datamodel(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()

    return "AI-generated" if pred == 1 else "Human-written"

classify_folder("emails_to_check")

# from user input

