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

def setup_model_and_tokenizer(model_name="microsoft/DialoGPT-medium"):
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
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    return model, tokenizer

def setup_lora_config():
    """
    Configure LoRA for efficient fine-tuning
    """
    return LoraConfig(
        r=64,                           # Rank of adaptation
        lora_alpha=16,                  # Alpha parameter for LoRA scaling
        target_modules=[                # Target modules for LoRA
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
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
    
    # Load primary medical QA dataset
    try:
        dataset = load_dataset("lavita/medical-qa-datasets", "all-processed")
        print(f"Loaded dataset with {len(dataset['train'])} samples")
        
        # Basic preprocessing
        def format_medical_qa(example):
            """Format medical QA into instruction-following format"""
            instruction = example.get('instruction', '')
            input_text = example.get('input', '')
            output_text = example.get('output', '')
            
            # Combine instruction and input
            if input_text:
                prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
            else:
                prompt = f"Instruction: {instruction}\nResponse:"
            
            # Create the full text for training
            full_text = f"{prompt} {output_text}"
            
            return {
                "text": full_text,
                "prompt": prompt,
                "response": output_text
            }
        
        # Apply formatting
        dataset = dataset.map(format_medical_qa)
        
        # Split dataset (use small subset for quick testing)
        train_size = min(10000, len(dataset['train']))  # Start with 10k samples
        dataset = dataset['train'].select(range(train_size))
        
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating dummy dataset for testing...")
        
        # Create a small dummy dataset for testing
        dummy_data = {
            "text": [
                "Instruction: Diagnose the following symptoms.\nInput: Patient has fever, cough, and fatigue.\nResponse: Based on the symptoms, this could indicate a viral infection such as influenza or COVID-19. Recommend rest, hydration, and monitoring symptoms.",
                "Instruction: What is the treatment for hypertension?\nResponse: Treatment for hypertension typically includes lifestyle changes (diet, exercise, weight management) and may require antihypertensive medications such as ACE inhibitors, beta-blockers, or diuretics.",
                "Instruction: Explain the risk factors for diabetes.\nResponse: Risk factors for diabetes include family history, obesity, sedentary lifestyle, age over 45, high blood pressure, and gestational diabetes history."
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
        num_train_epochs=3,
        per_device_train_batch_size=4,      # Adjust based on memory
        gradient_accumulation_steps=2,       # Effective batch size = 4*2 = 8
        learning_rate=2e-4,
        fp16=True,                          # Use mixed precision
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="no",           # Disable eval for quick start
        warmup_ratio=0.03,
        weight_decay=0.001,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        report_to="wandb" if wandb.api.api_key else "none",
        run_name="medical-llm-finetune",
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
        return
    
    print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
    print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Setup model and tokenizer
    model_name = "microsoft/DialoGPT-medium"  # Start with smaller model for testing
    # For production, use: "meta-llama/Llama-2-7b-chat-hf" or "microsoft/DialoGPT-large"
    
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
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
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=1024,
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

def test_model():
    """
    Quick test function to verify the model works
    """
    print("\n🧪 Testing model...")
    
    # Load the fine-tuned model
    try:
        from peft import PeftModel
        
        base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-medium",
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, "./medical-llm-final")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        
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

if __name__ == "__main__":
    # Initialize wandb (optional)
    try:
        wandb.login()
        print("✅ Weights & Biases initialized")
    except:
        print("⚠️  Weights & Biases not configured (optional)")
    
    # Run training
    main()
    
    # Test the model
    test_model()
    
    print("\n🎉 All done! Check ./medical-llm-final for your fine-tuned model")
    print("📖 Next steps:")
    print("   1. Evaluate on medical benchmarks (MedQA, MedMCQA)")
    print("   2. Test with different hyperparameters")
    print("   3. Scale up with larger datasets")
    print("   4. Compare with baseline models") 