import os
import jsonlines
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training

import warnings
warnings.filterwarnings("ignore")

BASE_MODEL = "microsoft/phi-1_5"
DATA_DIR = "data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
OUTPUT_DIR = "models/adapters/qlora_adapter"

BATCH_SIZE = 1
GRAD_ACCUM = 4
EPOCHS = 10
LR = 2e-4
MAX_LENGTH = 128

def load_training_dataset():
    return load_dataset("json",data_files=TRAIN_PATH)

def tokenize(example, tokenizer):
    prompt = f"Instruction:{example['instruction']}\nResponse:{example['output']}"
    
    tokens = tokenizer(
        prompt,
        truncation = True,
        max_length = MAX_LENGTH,
        padding = "max_length"
    )
    
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens

def main():
    print("\n===== Loading Base Model =====")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n=== Loading 4-bit quantized base model (qLoRA)… ===")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"       # <-- add this
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto" #GPU USAGE
    )
    
    # REQUIRED for QLoRA
    model = prepare_model_for_kbit_training(model)

    print("\n===== Load Dataset =====\n")
    dataset = load_training_dataset()
   
    print("\n===== Tokenizing Dataset =====\n")
    tokenized = dataset.map(
        lambda ex: tokenize(ex, tokenizer),
        batched=False
    )
    
    print("\n===== QLoRA Fine Tuning Configuration =====\n")
    lora_configs = LoraConfig(
        r = 8,
        lora_alpha=128,
        target_modules=["Wqkv", "out_proj",'fc1','fc2'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_configs)
    model.print_trainable_parameters()
    
    print("\n===== Starting Training =====\n")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=False,
        bf16=False,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        optim="paged_adamw_8bit"
    )
    
    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=tokenized['train'],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.gradient_checkpointing_enable()
    trainer.train()
    
    print("\n===== Saving QLoRA Adapter =====\n")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n===== LoRA Fine Tuning Completed =====\n")

if __name__=='__main__':
    main()