# text_generation_gpt2.py

import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Make sure to install:
# pip install transformers datasets

# Set your model and data path
model_name = "gpt2"
train_file_path = "data/train.txt" 

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Tokenize dataset
def load_dataset(train_file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=train_file_path,
        block_size=block_size
    )

train_dataset = load_dataset(train_file_path, tokenizer)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Setup training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
    prediction_loss_only=True,
    fp16=torch.cuda.is_available(),
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

# Start training
trainer.train()

# Save model
trainer.save_model("./gpt2_finetuned")
tokenizer.save_pretrained("./gpt2_finetuned")

print("✅ Fine-tuning complete and model saved!")
