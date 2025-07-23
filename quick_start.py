#!/usr/bin/env python3
"""
Medical LLM Fine-tuning Starter Script
RTX 3090 Optimized with QLoRA

This script provides a quick start for fine-tuning LLMs on medical datasets
using parameter-efficient techniques suitable for RTX 3090.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
import wandb

def setup_model_and_tokenizer(model_name="microsoft/DialoGPT-small"):
    """
    Setup model and tokenizer with 4-bit quantization for RTX 3090
    """
    print(f"Loading model: {model_name}")
    
    # 4-bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization and use safetensors
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_safetensors=True,  # Force use of safetensors
    )
    
    return model, tokenizer

def setup_lora_config():
    """
    Configure LoRA for efficient fine-tuning
    """
    return LoraConfig(
        r=32,                           # Reduced rank for smaller model
        lora_alpha=16,                  # Alpha parameter for LoRA scaling
        target_modules=[                # Target modules for LoRA (adjusted for GPT-2 style)
            "c_attn", "c_proj", "c_fc"
        ],
        lora_dropout=0.1,               # Dropout probability for LoRA layers
        bias="none",                    # Bias type
        task_type="CAUSAL_LM",         # Task type
    )

def load_medical_dataset():
    """
    Load and preprocess medical datasets
    """
    print("Loading medical datasets...")
    
    # Create a small dummy dataset for testing
    print("Creating dummy medical dataset for testing...")
    
    # Create a small dummy dataset for testing
    dummy_data = {
        "text": [
            "Instruction: Diagnose the following symptoms.\nInput: Patient has fever, cough, and fatigue.\nResponse: Based on the symptoms, this could indicate a viral infection such as influenza or COVID-19. Recommend rest, hydration, and monitoring symptoms.",
            "Instruction: What is the treatment for hypertension?\nResponse: Treatment for hypertension typically includes lifestyle changes (diet, exercise, weight management) and may require antihypertensive medications such as ACE inhibitors, beta-blockers, or diuretics.",
            "Instruction: Explain the risk factors for diabetes.\nResponse: Risk factors for diabetes include family history, obesity, sedentary lifestyle, age over 45, high blood pressure, and gestational diabetes history.",
            "Instruction: What are the symptoms of pneumonia?\nResponse: Pneumonia symptoms include persistent cough, fever, chills, shortness of breath, chest pain, and fatigue. Severe cases may require immediate medical attention.",
            "Instruction: How to manage chronic pain?\nResponse: Chronic pain management includes a combination of medications, physical therapy, lifestyle modifications, stress management, and sometimes psychological support.",
            "Instruction: What causes migraine headaches?\nResponse: Migraines can be triggered by stress, certain foods, hormonal changes, lack of sleep, bright lights, or strong smells. Treatment involves identifying triggers and preventive medications.",
            "Instruction: Explain heart attack symptoms.\nResponse: Heart attack symptoms include chest pain or pressure, shortness of breath, nausea, sweating, and pain radiating to arms, neck, or jaw. Seek immediate emergency care.",
            "Instruction: What is Type 2 diabetes?\nResponse: Type 2 diabetes is a chronic condition where the body becomes resistant to insulin or doesn't produce enough insulin, leading to elevated blood sugar levels.",
            "Instruction: How to prevent stroke?\nResponse: Stroke prevention includes controlling blood pressure, managing cholesterol, staying physically active, eating a healthy diet, avoiding smoking, and limiting alcohol.",
            "Instruction: What causes asthma?\nResponse: Asthma is caused by inflammation and narrowing of airways, often triggered by allergens, respiratory infections, exercise, cold air, or stress."
        ]
    }
    
    from datasets import Dataset
    return Dataset.from_dict(dummy_data)

def setup_training_args():
    """
    Configure training arguments optimized for RTX 3090
    """
    return TrainingArguments(
        output_dir="./medical-llm-results",
        num_train_epochs=2,                 # Reduced for quick testing
        per_device_train_batch_size=2,      # Smaller batch size for safety
        gradient_accumulation_steps=4,       # Effective batch size = 2*4 = 8
        learning_rate=2e-4,
        fp16=True,                          # Use mixed precision
        logging_steps=5,                    # More frequent logging
        save_strategy="epoch",
        eval_strategy="no",                 # Fixed parameter name
        warmup_ratio=0.03,
        weight_decay=0.001,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        report_to="wandb" if wandb.api.api_key else "none",
        run_name="medical-llm-finetune",
        gradient_checkpointing=True,        # Enable to save memory
    )

def main():
    """
    Main training function
    """
    print("🚀 Starting Medical LLM Fine-tuning")
    print("=" * 50)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This script requires a GPU.")
        return False
    
    print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
    print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Setup model and tokenizer
    model_name = "microsoft/DialoGPT-small"  # Smaller model for testing
    
    try:
        model, tokenizer = setup_model_and_tokenizer(model_name)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Trying alternative model...")
        # Try alternative model
        model_name = "distilgpt2"
        model, tokenizer = setup_model_and_tokenizer(model_name)
        print("✅ Alternative model loaded successfully!")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    print(f"📈 Trainable parameters: {model.num_parameters():,}")
    print(f"🔒 Total parameters: {model.base_model.num_parameters():,}")
    print(f"📊 Trainable %: {100 * model.num_parameters() / model.base_model.num_parameters():.2f}%")
    
    # Load dataset
    dataset = load_medical_dataset()
    print(f"📚 Dataset size: {len(dataset)} samples")
    
    # Setup training arguments
    training_args = setup_training_args()
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        text_field="text",              # Updated parameter name
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=512,             # Reduced sequence length
        packing=False,
    )
    
    print("\n🎯 Starting training...")
    print("=" * 50)
    
    # Start training
    try:
        trainer.train()
        print("✅ Training completed successfully!")
        
        # Save the final model
        trainer.save_model("./medical-llm-final")
        print("💾 Model saved to ./medical-llm-final")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Try reducing batch_size or max_seq_length if you encounter OOM errors")
        return False
    
    return True

def test_model():
    """
    Quick test function to verify the model works
    """
    print("\n🧪 Testing model...")
    
    # Load the fine-tuned model
    try:
        from peft import PeftModel
        
        base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-small",
            device_map="auto",
            use_safetensors=True
        )
        model = PeftModel.from_pretrained(base_model, "./medical-llm-final")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test prompt
        test_prompt = "Instruction: What are the symptoms of diabetes?\nResponse:"
        
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🔍 Test Response:\n{response}")
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("🔧 Initializing Medical LLM Training Environment...")
    
    # Test imports first
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported")
        import transformers
        print(f"✅ Transformers {transformers.__version__} imported")
        import peft
        print(f"✅ PEFT {peft.__version__} imported")
        import trl
        print(f"✅ TRL imported")
    except Exception as e:
        print(f"❌ Import error: {e}")
        exit(1)
    
    # Initialize wandb (optional)
    try:
        wandb.login()
        print("✅ Weights & Biases initialized")
    except:
        print("⚠️  Weights & Biases not configured (optional)")
    
    # Run training
    success = main()
    
    if success:
        # Test the model
        test_model()
        
        print("\n🎉 All done! Check ./medical-llm-final for your fine-tuned model")
        print("📖 Next steps:")
        print("   1. Load real medical datasets (lavita/medical-qa-datasets)")
        print("   2. Evaluate on medical benchmarks (MedQA, MedMCQA)")
        print("   3. Test with different hyperparameters")
        print("   4. Scale up with larger models")
        print("   5. Compare with baseline models")
    else:
        print("\n❌ Training failed. Please check the error messages above.")
        print("💡 Common solutions:")
        print("   1. Reduce batch size in setup_training_args()")
        print("   2. Reduce max_seq_length in the trainer")
        print("   3. Enable gradient_checkpointing")
        print("   4. Try a smaller model") 