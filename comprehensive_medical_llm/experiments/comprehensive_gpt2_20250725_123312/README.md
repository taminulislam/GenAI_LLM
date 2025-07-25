---
base_model: gpt2
library_name: transformers
model_name: comprehensive_gpt2_20250725_123312
tags:
- generated_from_trainer
- trl
- sft
licence: license
---

# Model Card for comprehensive_gpt2_20250725_123312

This model is a fine-tuned version of [gpt2](https://huggingface.co/gpt2).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/taminul/medical-llm-finetuning/runs/wi74b18b) 


This model was trained with SFT.

### Framework versions

- TRL: 0.19.1
- Transformers: 4.53.3
- Pytorch: 2.5.1+cu121
- Datasets: 4.0.0
- Tokenizers: 0.21.2

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```