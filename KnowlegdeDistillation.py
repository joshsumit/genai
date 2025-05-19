import torch
from torch.utils.data import DataLoader
from transformers import (
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    BertTokenizer,
    AdamW,
    get_scheduler,
)
from datasets import load_dataset
import torch.nn.functional as F

# Load dataset
dataset = load_dataset("imdb", split="train[:1%]")  # small subset for demo
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Load models
teacher = BertForSequenceClassification.from_pretrained("bert-base-uncased").eval()
student = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Optimizer and scheduler
optimizer = AdamW(student.parameters(), lr=5e-5)
num_training_steps = len(dataloader) * 3
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher.to(device)
student.to(device)

temperature = 2.0
alpha = 0.5  # balance between distillation loss and CE loss

for epoch in range(3):
    student.train()
    total_loss = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Teacher predictions
        with torch.no_grad():
            teacher_logits = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            soft_targets = F.softmax(teacher_logits / temperature, dim=-1)

        # Student predictions
        student_logits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits

        # Losses
        loss_ce = F.cross_entropy(student_logits, batch["label"])
        loss_kd = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            soft_targets,
            reduction="batchmean"
        ) * (temperature ** 2)

        loss = alpha * loss_kd + (1 - alpha) * loss_ce

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {total_loss / len(dataloader):.4f}")
