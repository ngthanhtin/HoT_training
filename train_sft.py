import os
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import (
    LoraConfig,
    PeftModel,
    PeftConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, DPOTrainer, DPOConfig, SFTConfig, GRPOConfig, GRPOTrainer

import re
from tqdm import tqdm
import wandb
import random

from data_utils import (
    load_and_preprocess_dataset,
    qa_formatter,
    process_examples_for_localization_training,
    process_examples_for_repeating_training)


# Define constants
MODEL_NAME = "Qwen/Qwen2.5-1.5B"  # or "meta-llama/Llama-3-8B" if you have access
OUTPUT_DIR = "./sft_localization"
NUM_EPOCHS = 20
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize wandb for experiment tracking
wandb.init(project="factual-span-tagging-sft")

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# Main function for SFT training
def train_with_sft(data_type='localization'):
    print("Loading dataset...")
    train_ds, val_ds, test_ds = load_and_preprocess_dataset(no_test_split=True)
    print(f"Dataset loaded: {len(train_ds)} train, {len(val_ds)} val test samples")
    # Process the dataset
    print("Processing dataset...")
    if data_type == 'localization':
        processed_train = process_examples_for_localization_training(train_ds)
        processed_val = process_examples_for_localization_training(val_ds)
    elif data_type == 'repeating':
        processed_train = process_examples_for_repeating_training(train_ds)
        processed_val = process_examples_for_repeating_training(val_ds)
    
    processed_train = Dataset.from_list(processed_train)
    processed_val = Dataset.from_list(processed_val)
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure QLoRA for efficient fine-tuning
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Adjust for your specific model
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # load_in_8bit=True,
        device_map="auto", #balanced
        torch_dtype=torch.float16,
        # device_map={"": torch.cuda.current_device()}  # pin to one GPU
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Prepare model for QLoRA fine-tuning
    # model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    print(f"Model loaded: {MODEL_NAME}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Set up training arguments
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        # evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        fp16=True,
        report_to="wandb",
        max_seq_length=1024,
        # dataset_text_field="input",
        packing=False,
    )
    
    # Initialize SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        # tokenizer=tokenizer,
        train_dataset=processed_train,
        eval_dataset=processed_val,
        formatting_func=qa_formatter, 
    )
    
    # Start training
    print("Starting SFT training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    
    return model, tokenizer

if __name__ == "__main__":
    data_type = 'localization'  # or 'repeating'
    sft_model, tokenizer = train_with_sft(data_type=data_type)
    
    print("Training SFT completed successfully!")
    wandb.finish()