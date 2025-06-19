import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from peft import get_peft_model, LoraConfig, TaskType
import evaluate

# Step 1: Load dataset
print("Loading CNN/DailyMail dataset...")
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Step 2: Load tokenizer and model
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Step 3: Apply LoRA
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

# Step 4: Preprocess the data
def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Step 5: Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./task_11_outputs",
    # evaluation_strategy="epoch",
    learning_rate=5e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    # save_total_limit=2,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=10
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Step 6: Define ROUGE metric
rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 2) for k, v in result.items()}

# Step 7: Setup trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].select(range(1000)),
    eval_dataset=tokenized_dataset["validation"].select(range(100)),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Step 8: Train model
print("\nTraining model...")
trainer.train()

# Step 9: Save model
print("\nSaving model...")
model.save_pretrained("./task_11_outputs/final_model")
tokenizer.save_pretrained("./task_11_outputs/final_model")

# Step 10: Inference example
print("\nTesting inference...")
example = dataset["test"][0]["article"]
input_text = "summarize: " + example
inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
summary_ids = model.generate(**inputs, max_length=128)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("\nGenerated Summary:\n", summary)
