import torch
import torch.nn as nn
from transformers import BertModel

# Load a pre-trained model
model = BertModel.from_pretrained('bert-base-uncased')

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Add a trainable classification head
class CustomModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Linear(base_model.config.hidden_size, 2)  # Example: binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.classifier(cls_output)

custom_model = CustomModel(model)
