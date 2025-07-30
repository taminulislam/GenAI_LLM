# MESA: Medical Evaluation across Safety and Accuracy 🏥

 Comprehensive study of fine-tuned language models for medical AI.

## Key Results
- **BioGPT-Large**: 67.2% exact match accuracy, 78.1% medical entity recognition
- **DialoGPT-Small**: 100% safety rate, 0.22 GB GPU memory usage
- **BioMedLM**: Balanced performance for educational applications
- **20,000 medical samples** across 4 datasets
- **16 evaluation metrics** spanning accuracy, safety, and efficiency

## 🚀 Quick Start

<!-- ### 1. Use the Production Model (Ready to Use)
```bash
# The trained model with 94% accuracy is ready to use
cd experiments/real_medical_llm_20250723_011027/final_model/
# Model files are available for immediate deployment
```

### 2. Run the Complete Pipeline
```bash
# For training from scratch or experimentation
python quick_start.py

# Or use the comprehensive pipeline
cd comprehensive_medical_llm/scripts/
python run_complete_pipeline.py
``` -->

### Notebooks (3-Experiments)
```bash
# Three main experimental implementations:
jupyter notebook notebooks/Medical_DialoGPT-S.ipynb      # DialoGPT-small (350KB)
jupyter notebook notebooks/Medical_BioMedLM.ipynb        # BioMedLM (354KB) 
jupyter notebook notebooks/Medical_BioGPT-L.ipynb        # BioGPT-Large (369KB)


```

## 📊 Datasets Used

| Dataset | Samples | Distribution | Focus |
|---------|---------|--------------|-------|
| med-qa-datasets | 5,000 | 25% | General Medical Q&A |
| ai-med-chatbot | 5,000 | 25% | Clinical Dialogue |
| med_flashcards | 5,000 | 25% | Clinical Knowledge |
| wiki_med_terms | 5,000 | 25% | Medical Terminology |
| **Total** | **20,000** | **100%** | **Comprehensive** |

## 🔧 Technical Details

### Model Configuration
```python
# QLoRA Setup for RTX 3090
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# LoRA Configuration
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
```

### Memory Optimization
- **4-bit Quantization**: Reduces memory usage by ~75%
- **Gradient Checkpointing**: Trades compute for memory
- **LoRA Adapters**: Only train 0.1-1% of parameters
- **Optimized Batch Size**: Balanced for RTX 3090's 24GB

## 📈 **Comprehensive Evaluation Results**

![Model Response Comparison](genAi.pdf)

**16 METRICS ACROSS 4 DIMENSIONS:**

| Model | Exact Match Acc. | Medical Entity Rec. | Safety Rate | GPU Memory |
|-------|------------------|-------------------|-------------|------------|
| **BioGPT-Large** | **67.2%** | **78.1%** | 94.3% | 10.2 GB |
| **BioMedLM** | 49.2% | 38.9% | 87.3% | 11.8 GB |
| **DialoGPT-Small** | 50.2% | 42.5% | **100.0%** | **0.22 GB** |

**Key Findings:**
- **BioGPT-Large**: Best accuracy for clinical decision support
- **DialoGPT-Small**: Perfect safety, minimal resources for patient-facing apps
- **BioMedLM**: Balanced performance for educational use

## Project Structure

```
GenAI_LLM/
├── notebooks/                                    # 🔬 Original Experiments
│   ├── Medical_DialoGPT-S.ipynb                # Microsoft DialoGPT-small
│   ├── Medical_BioMedLM.ipynb                  # Stanford BioMedLM  
│   └── Medical_BioGPT-L.ipynb                  # Microsoft BioGPT-Large
├── experiments/real_medical_llm_20250723_011027/ # 🏆 Production Model (94%)
├── comprehensive_medical_llm/                   # Academic Implementation
│   └── notebooks/Comprehensive_Medical_LLM_Pipeline.ipynb
├── src/                                        # Core modules
├── data/                                       # Medical datasets
└── outputs/                                    # Results
```



## Experimental Notebooks

| Notebook | Model | Parameters | Focus |
|----------|-------|------------|-------|
| `Medical_DialoGPT-S.ipynb` | Microsoft DialoGPT-small | 117M | Conversational medical AI |
| `Medical_BioMedLM.ipynb` | Stanford BioMedLM | 2.7B | Biomedical domain expertise |
| `Medical_BioGPT-L.ipynb` | Microsoft BioGPT-Large | 347M | Large-scale medical reasoning |

## Technical Highlights

- **Parameter Efficiency**: LoRA adaptation (5.44-9.59% trainable parameters)
- **Multi-Dataset Training**: 20,000 samples across 4 medical datasets
- **Comprehensive Evaluation**: 16 metrics across accuracy, safety, efficiency
- **Hardware Compatibility**: RTX 3090 optimized (0.22-11.8 GB memory usage)

## Status: Project Completed ✅

Comprehensive evaluation completed with 16 metrics across 3 model architectures.
**Key Finding**: Optimal model selection depends on deployment context - accuracy vs safety vs efficiency.

## Requirements

- Python 3.8+
- RTX 3090 or equivalent GPU
- 16GB+ RAM

```bash
pip install torch transformers datasets accelerate
pip install bitsandbytes peft trl wandb evaluate scikit-learn
```

## License

MIT License

## Contact

Open an issue on GitHub for questions.

