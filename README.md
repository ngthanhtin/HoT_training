---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B
tags:
  - peft
  - lora
  - qwen
  - causal-lm
  - instruction-tuning
inference: false
---

# 🔮 Fine-Tuned Qwen2.5-1.5B with PEFT (LoRA)

This repository contains PEFT adapter weights fine-tuned from the base model [`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B) using [LoRA (Low-Rank Adaptation)](https://github.com/huggingface/peft). Only the adapter weights are included here.

## 🧾 Model Details

- **Base Model**: Qwen/Qwen2.5-1.5B
- **Fine-Tuning Method**: LoRA (via 🤗 PEFT)
- **Library Versions**:
  - `transformers >= 4.36.0`
  - `peft >= 0.10.0`
  - `accelerate >= 0.21.0`
- **Use Case**: Instruction following / downstream adaptation

## 💡 Usage

To load and use this adapter:

```python
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

# Paths
base_model = "Qwen/Qwen2.5-1.5B"
adapter_path = "your-username/your-repo-name"  # replace with your repo

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter_path)

# Inference
prompt = "<question>Once upon a time,</question>"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

If you use this model, please cite:
```
@article{nguyen2025hot,
  title={HoT: Highlighted Chain of Thought for Referencing Supporting Facts from Inputs},
  author={Nguyen, Tin and Bolton, Logan and Taesiri, Mohammad Reza and Nguyen, Anh Totti},
  journal={arXiv preprint arXiv:2503.02003},
  year={2025}
}
```
