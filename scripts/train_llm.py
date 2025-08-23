import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Load dataset
dataset = load_dataset("json", data_files="data/waste_tips.jsonl", split="train")

# Tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Preprocess
def tokenize_fn(examples):
    return tokenizer(examples["input"] + " " + tokenizer.eos_token + " " + examples["output"], truncation=True, max_length=128)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["input", "output"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training
training_args = TrainingArguments(
    output_dir="models/waste_llm",
    num_train_epochs=20,
    per_device_train_batch_size=4,
    save_steps=50,
    save_total_limit=2,
    logging_steps=10,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
trainer.save_model("models/waste_llm")
tokenizer.save_pretrained("models/waste_llm")
