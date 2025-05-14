
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import torch

model_path = "C:\\sj\\models\\Mistral-7B-Instruct-v0.3"
output_path = "C:\\sj\\models\\finetuned\\classification"
# Example dataset (replace with your own if you want)
data_dict = {
    "text": [
        "  The staff was very kind and attentive to my needs!!!  ",
        "The waiting time was too long, and the staff was rude. Visit us at http://hospitalreviews.com",
        "The doctor answered all my questions...but the facility was outdated.   ",
        "The nurse was compassionate & made me feel comfortable!! :) ",
        "I had to wait over an hour before being seen.  Unacceptable service! #frustrated",
        "The check-in process was smooth, but the doctor seemed rushed. Visit https://feedback.com",
        "Everyone I interacted with was professional and helpful.  "
    ],
    "label": ["positive", "negative", "neutral", "positive", "negative", "neutral", "positive"]
}

# Convert to pandas DataFrame
data = pd.DataFrame(data_dict)

# Clean the text
import re

def clean_text(text):
    text = text.lower().strip()  # Convert to lowercase and remove extra spaces
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    return text

# Apply text cleaning
data["cleaned_text"] = data["text"].apply(clean_text)


# Convert labels to numerical values
data["label"] = data["label"].astype("category").cat.codes  # Converts ["positive", "negative", "neutral"] to [0, 1, 2]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# Apply tokenization with padding
def tokenize_function(text):
    return tokenizer(text, truncation=True, padding="max_length", max_length=128)

# Apply tokenization
data["tokenized"] = data["cleaned_text"].apply(tokenize_function)

# Extract tokenized features
data["input_ids"] = data["tokenized"].apply(lambda x: x["input_ids"])
data["attention_mask"] = data["tokenized"].apply(lambda x: x["attention_mask"])

# Drop old tokenized column
data = data.drop(columns=["tokenized"])

print(data.head())


# Split into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

# Remove unnecessary columns
train_dataset = train_dataset.remove_columns(["text", "cleaned_text"])
test_dataset = test_dataset.remove_columns(["text", "cleaned_text"])

#print(train_dataset)

# Enable dynamic padding for batches
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    output_dir=output_path,
    logging_dir=output_path,
    report_to="none",  
    save_strategy="epoch",  
    evaluation_strategy="epoch",  
)
# Load pre-trained BERT model (3-class classification)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3, device_map='cuda', torch_dtype=torch.float16)
model.config.pad_token_id = model.config.eos_token_id
# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,  
)

# Train the model
trainer.train()

from sklearn.metrics import accuracy_score, f1_score

# Generate predictions
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
labels = test_dataset['label']

# Calculate metrics
accuracy = accuracy_score(labels, preds)
f1 = f1_score(labels, preds, average='weighted')

print(f"Accuracy: {accuracy}, F1 Score: {f1}")