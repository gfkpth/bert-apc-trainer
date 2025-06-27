from datasets import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import classification_report

# Example data
examples = [
    {"tokens": ["Wir", "Linguisten", "arbeiten", "."],
     "labels": [0, 1, 2, 2]}  # 0=B-CONSTR, 1=I-CONSTR, 2=O
]

# Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-german-cased")
model = BertForTokenClassification.from_pretrained("bert-base-german-cased", num_labels=3)

# Tokenize and align labels
def tokenize_and_align(ex):
    tokenized = tokenizer(ex["tokens"], is_split_into_words=True, truncation=True, padding="max_length", max_length=32)
    word_ids = tokenized.word_ids()
    label_ids = []
    for word_idx in word_ids:
        label_ids.append(ex["labels"][word_idx] if word_idx is not None else -100)  # -100 to ignore in loss
    tokenized["labels"] = label_ids
    return tokenized

# Prepare dataset
dataset = Dataset.from_list(examples)
tokenized_dataset = dataset.map(tokenize_and_align)

# Training setup
training_args = TrainingArguments(
    output_dir="./bert-tokenclass",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,
    evaluation_strategy="no"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)