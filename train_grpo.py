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
from trl import GRPOConfig, GRPOTrainer, VLLMSampler

import re
from tqdm import tqdm
import wandb
import random

from data_utils import load_and_preprocess_dataset, qa_formatter, process_examples_for_localization_training, process_examples_for_repeating_training

# Define constants
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

def load_pretrained_model(pretrained_model_path):
    # Load the base model and tokenizer first
    base_model_name = "Qwen/Qwen2.5-1.5B"  # or "meta-llama/Llama-3-8B" if you have access
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load the PEFT configuration and adapter weights
    peft_config = PeftConfig.from_pretrained(pretrained_model_path)
    model = PeftModel.from_pretrained(model, pretrained_model_path)
    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"Model loaded successfully from {pretrained_model_path}")
    return model, tokenizer

# ---------------------------------------------------------------------
# Reward Functions
# ---------------------------------------------------------------------
import re
from typing import List

# ---------- utilities ---------------------------------------------------------

QUESTION_RE = re.compile(r"<question>(.*?)</question>", re.S)
ANSWER_RE   = re.compile(r"<answer>(.*?)</answer>",   re.S)
FACT_RE     = re.compile(
    r"<fact(\d+) x1=(\d+) x2=(\d+)>(.*?)</fact\1>", re.S | re.I
)
FINAL_ANS_RE = re.compile(r"\{(.*?)\}")      # finds “… {10} …”

def extract_sections(text: str):
    """return question, answer-body (no tags)"""
    q = QUESTION_RE.search(text)
    a = ANSWER_RE.search(text)
    return (q.group(1).strip() if q else ""), (a.group(1).strip() if a else "")

def extract_final_answer(ans_body: str) -> str:
    m = FINAL_ANS_RE.search(ans_body)
    return m.group(1).strip() if m else ""

# ---------- 1. correctness (final numeric / MC choice) ------------------------
def correctness_reward(prompts, completions, ground_truth, **kw):

    outs = []
    for c in completions:
        # _, body = extract_sections(c[0]["content"])
        outs.append(extract_final_answer(c))

    # compare element-wise
    scores = []
    for pred, gold in zip(outs, ground_truth):
        # numeric answers → compare numerically
        if pred.strip().isdigit() and gold.strip().isdigit():
            score = 2.0 if float(pred) == float(gold) else 0.0
        # otherwise do a case-insensitive string match
        else:
            score = 2.0 if pred.strip().lower() == gold.strip().lower() else 0.0
        scores.append(score)

    return scores          # list[float]  length == batch_size

# ---------- 2. tag-template reward  (soft) ------------------------------------

TAG_PAT = re.compile(
    r"<answer>.*?</answer>", re.S | re.I
)

def format_reward(completions, **kw):
    return [0.5 if TAG_PAT.search(c) else 0.0
            for c in completions]

# ---------- 3. xml-count / early-stop reward ---------------------------------

def xmlcount_reward(completions, **kw):
    scores = []
    for c in completions:
        txt = c
        score = 0.0
        # one & only one pair of wrappers?
        score += 0.25 if txt.count("<answer>")    == 1 else 0
        score += 0.25 if txt.count("</answer>")   == 1 else 0
        # discourage trailing junk after </answer>
        tail = txt.split("</answer>")[-1]
        score -= len(tail) * 0.001
        scores.append(max(score, 0.0))
    return scores
# ---------- 4. span-tagging reward (hard) ------------------------------------
TOKEN_RE = re.compile(r"\w+|\$?\d+")

def toks(s: str) -> set[str]:
    """Lower-case & tokenize; return a *set* for fast Jaccard."""
    return set(TOKEN_RE.findall(s.lower()))

def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union)

# from sentence_transformers import SentenceTransformer
# from numpy import dot
# from numpy.linalg import norm

# model = SentenceTransformer("all-MiniLM-L6-v2")

# def cosine(a, b):
#     return dot(a, b) / (norm(a) * norm(b))

# def span_sim(txt, gold):
#     v1, v2 = model.encode([txt, gold])
#     return cosine(v1, v2)        # threshold maybe 0.85


def span_reward(prompts, completions, **kw) -> list[float]:

    rewards = []
    θ = 0.7                       # similarity threshold

    for prompt, comp in zip(prompts, completions):
        # ----- fetch question tokens ------------------------------------
        q_raw, _ = extract_sections(prompt)
        q_tokens = TOKEN_RE.findall(q_raw)          # list[str]
        # ----- evaluate ONE completion (first candidate) ----------------
        body         = extract_sections(comp)[1]
        ok, tot = 0, 0
        for fid, x1, x2, txt in FACT_RE.findall(body):
            tot += 1
            x1, x2     = int(x1), int(x2)
            gold_txt   = " ".join(q_tokens[x1:x2+1])
            sim        = jaccard(toks(txt), toks(gold_txt))
            if sim >= θ:
                ok += 1
        # avoid divide-by-zero when no facts are present
        rewards.append(ok / tot if tot else 0.0)

    return rewards

def make_prompt_dataset(data_type="localization"):
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
    
    train_rows = [{"prompt": f"<question>{q['input']}</question>", 'ground_truth': q['gt']}
            for q in processed_train]
    val_rows = [{"prompt": f"<question>{q['input']}</question>", 'ground_truth': q['gt']}
            for q in processed_val]     

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)
    
    return train_ds, val_ds
    
# Main function for GRPO training
def train_with_grpo(sft_model, tokenizer, data_type="localization"):
    train_ds, eval_ds = make_prompt_dataset(data_type=data_type)

    grpo_cfg = GRPOConfig(
        output_dir=f"./grpo_{data_type}_checkpoints",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=1e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        beta=0.04,                       # GRPO default (≈ KL weight)
        max_grad_norm = 0.1,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = 'cosine',
        fp16=False,
        max_prompt_length = 256,
        max_completion_length = 512,
        num_generations=4, 
        
    )

    # ---------------------------------------------------------------------
    # 5.  Trainer
    # ---------------------------------------------------------------------
    trainer = GRPOTrainer(
        model=sft_model,
        reward_funcs=[correctness_reward, format_reward, xmlcount_reward, span_reward],     # <-- *must* be given
        args=grpo_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,      # tokenizer param in GRPOTrainer
    )

    trainer.train()
    trainer.save_model("./output/grpo_final_model")

# Main execution flow
if __name__ == "__main__":
    data_type = 'localization'  # or 'repeating'
    sft_model, tokenizer = load_pretrained_model(f"./sft_{data_type}/final_model")
    final_model = train_with_grpo(sft_model, tokenizer, data_type=data_type)
    
    print("Training completed successfully!")
    wandb.finish()