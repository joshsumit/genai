import tensorflow as tf
from transformers import TFBertModel

# Load a pre-trained model
base_model = TFBertModel.from_pretrained('bert-base-uncased')

# Freeze all layers
for layer in base_model.layers:
    layer.trainable = False

# Add a trainable classification head
input_ids = tf.keras.Input(shape=(128,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.Input(shape=(128,), dtype=tf.int32, name="attention_mask")

outputs = base_model(input_ids, attention_mask=attention_mask)
cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
logits = tf.keras.layers.Dense(2, activation='softmax')(cls_output)

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=logits)
