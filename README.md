# Medical LLM Fine-Tuning for Clinical Decision Support

A research project for fine-tuning large language models on medical datasets using parameter-efficient techniques optimized for RTX 3090.

## 🎯 Project Overview

This project demonstrates how to achieve competitive medical AI performance using smaller, efficiently fine-tuned models rather than massive foundation models. Perfect for researchers with consumer-grade hardware who want to contribute to medical AI research.

### Key Features
- **Parameter-Efficient**: Uses QLoRA for 4-bit quantization and LoRA adapters
- **RTX 3090 Optimized**: Carefully tuned for 24GB VRAM constraints  
- **Multi-Dataset Training**: Combines multiple medical datasets for robust performance
- **Publication-Ready**: Structured for academic research and paper submission

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone this repository
git clone <your-repo-url>
cd medical-llm-finetuning

# Install dependencies
pip install torch transformers datasets accelerate
pip install bitsandbytes peft trl wandb
pip install evaluate scikit-learn
```

### 2. Run Training
```bash
python quick_start.py
```

This will:
- Load a medical QA dataset (239k samples)
- Fine-tune a model using QLoRA
- Save the trained model
- Run a quick test

## 📊 Datasets Used

| Dataset | Size | Description |
|---------|------|-------------|
| `lavita/medical-qa-datasets` | 239k | Comprehensive medical QA |
| `BI55/MedText` | 1.4k | High-quality clinical scenarios |
| `AdaptLLM/medicine-LLM` | Various | ChemProt, RCT, specialized datasets |

### Evaluation Benchmarks
- **MedQA**: Medical exam questions
- **MedMCQA**: Multiple choice medical questions  
- **PubMedQA**: Biomedical literature QA

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

## 📈 Expected Results

Based on current research trends, you should expect:

| Metric | Target | SOTA Comparison |
|--------|--------|-----------------|
| MedQA Accuracy | 75-85% | GPT-4: ~86% |
| MedMCQA Accuracy | 70-80% | GPT-4: ~75% |
| Training Time | 6-12 hours | - |
| Memory Usage | <20GB | Full FT: >80GB |

## 🏗️ Project Structure

```
medical-llm-finetuning/
├── project_setup.md       # Detailed project documentation
├── quick_start.py         # Main training script
├── README.md              # This file
├── experiments/           # Experimental results and logs
├── models/                # Saved models and checkpoints
├── data/                  # Dataset cache and preprocessing
└── evaluation/            # Benchmark evaluation scripts
```

## 📚 Research Contributions

### 1. Parameter Efficiency Analysis
- Demonstrate 3B models competing with 70B+ models
- Cost-benefit analysis of QLoRA vs full fine-tuning
- Memory optimization for consumer hardware

### 2. Multi-Dataset Clinical Reasoning
- Performance across medical specialties
- Knowledge transfer between datasets
- Specialty-specific evaluation metrics

### 3. Safety and Hallucination Detection
- Uncertainty quantification methods
- Medical fact-checking mechanisms
- Clinical safety evaluation

## 🎓 Publication Strategy

### Target Venues
- **Primary**: EMNLP, ACL, NAACL (NLP conferences)
- **Secondary**: CHIL, JAMIA (medical informatics)
- **Workshops**: Clinical NLP, AI for Healthcare

### Paper Outline
1. **Introduction**: Clinical AI + parameter efficiency challenges
2. **Related Work**: Medical LLMs + efficient fine-tuning
3. **Methodology**: Multi-dataset QLoRA approach
4. **Experiments**: Comprehensive benchmark evaluation
5. **Results**: Performance analysis + clinical insights
6. **Discussion**: Safety, limitations, future work

## 📋 Timeline (16 weeks)

| Phase | Duration | Tasks |
|-------|----------|-------|
| **Setup** | Weeks 1-2 | Environment, data preprocessing, baselines |
| **Training** | Weeks 3-8 | Single/multi-dataset training, optimization |
| **Evaluation** | Weeks 9-12 | Benchmark testing, safety evaluation |
| **Research** | Weeks 13-16 | Analysis, paper writing, submission |

## 💰 Budget Considerations

### Computational Costs
- **Training**: ~50-100 GPU hours on RTX 3090
- **Evaluation**: ~20-30 GPU hours
- **Total Hardware Cost**: $0 (using existing RTX 3090)

### Optional Expenses
- **API Costs**: ~$100 for GPT-4 baseline comparisons
- **Storage**: Cloud storage for large datasets (~$20/month)

## ⚠️ Important Considerations

### Technical Limitations
- **Memory Constraints**: Carefully tune batch sizes
- **Training Stability**: Monitor for gradient explosions
- **Evaluation Bias**: Ensure diverse test sets

### Ethical Considerations
- **Medical Accuracy**: Never use for actual medical advice
- **Bias Detection**: Test across demographic groups
- **Privacy**: Ensure all datasets are de-identified

## 🔍 Troubleshooting

### Common Issues

#### Out of Memory (OOM)
```python
# Reduce batch size
per_device_train_batch_size=2

# Enable gradient checkpointing
gradient_checkpointing=True

# Reduce sequence length
max_seq_length=512
```

#### Slow Training
```python
# Enable mixed precision
fp16=True

# Increase batch size with accumulation
gradient_accumulation_steps=4
```

#### Poor Performance
- Check data preprocessing
- Verify prompt formatting
- Increase training epochs
- Try different learning rates

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for transformers and datasets
- **Microsoft** for DeepSpeed optimizations
- **Medical dataset contributors** for open-source datasets
- **PEFT library** for parameter-efficient fine-tuning

## 📞 Contact

For questions about this research project:
- Open an issue on GitHub
- Email: [your-email@domain.com]
- Twitter: [@your-handle]

---

**🎯 Ready to start your medical AI research journey?**

Run `python quick_start.py` and begin fine-tuning your first medical LLM!

**📖 Next Steps:**
1. Review `project_setup.md` for detailed methodology
2. Experiment with different model sizes and datasets
3. Evaluate on medical benchmarks
4. Compare with baseline models
5. Write and submit your research paper

*Good luck with your research! 🚀*

