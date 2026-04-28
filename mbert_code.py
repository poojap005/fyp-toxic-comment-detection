import pandas as pd
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

#set random seeds to improve reproducibility across runs
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

#load dataset
df = pd.read_csv("Toxic_Comment_Dataset.csv")

print(df.head())
print(df["Label"].value_counts())

#separate comments and labels
X = df["Comments"]
y = df["Label"]

#split dataset into training and testing sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y #ensures class distribution is similar in both sets
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

#tokenization (mBERT)
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")

#tokenizing training data
train_encodings = tokenizer(
    list(X_train),
    truncation=True, #avoids overflow
    padding=True, #ensures texts are of same length
    max_length=128
)

#tokenizing test data
test_encodings = tokenizer(
    list(X_test),
    truncation=True,
    padding=True,
    max_length=128
)

print("Tokenization Done")

#dataset class
import torch

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __getitem__(self, idx):
        data = {}
        for key in self.encodings:
            data[key] = torch.tensor(self.encodings[key][idx])

        data["labels"] = torch.tensor(self.labels[idx])
        return data
    
    def __len__(self):
        return len(self.labels)
    
train_dataset = MyDataset(train_encodings, y_train)
test_dataset = MyDataset(test_encodings, y_test)

#implementing the model
model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-multilingual-cased",
    num_labels=2
)

print("Model Loaded")

#metrics for trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision":precision,
        "recall":recall,
        "f1": f1
    }

#training arguments
training_args = TrainingArguments(
    output_dir="./mbert_results",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./mbert_logs",
    save_total_limit=2,
    seed=42
)

#training the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


trainer.train()

#getting predictions
predictions = trainer.predict(test_dataset)

#predicted labels
y_pred = np.argmax(predictions.predictions, axis=1)

#true labels
y_true = y_test.to_numpy()

#metrics
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary", zero_division=0
)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

#confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

print("Unique Predictions:", np.unique(y_pred, return_counts=True))

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("mBERT_confusion_matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("mBERT_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()