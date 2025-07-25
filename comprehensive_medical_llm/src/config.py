"""
Medical LLM Configuration Module
Contains all hyperparameters, model settings, and training configurations
"""

import torch
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """Model configuration settings"""
    # Model selection
    base_model_name: str = "microsoft/DialoGPT-small"
    fallback_model_name: str = "distilgpt2"
    use_safetensors: bool = True
    trust_remote_code: bool = True
    
    # Quantization settings
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    bnb_4bit_use_double_quant: bool = True

@dataclass
class LoRAConfig:
    """LoRA fine-tuning configuration"""
    r: int = 32                           # Rank of adaptation
    lora_alpha: int = 16                  # Alpha parameter for LoRA scaling
    target_modules: List[str] = None      # Will be set based on model type
    lora_dropout: float = 0.1             # Dropout probability for LoRA layers
    bias: str = "none"                    # Bias type
    task_type: str = "CAUSAL_LM"          # Task type
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for GPT-2 style models
            self.target_modules = ["c_attn", "c_proj", "c_fc"]

@dataclass
class TrainingConfig:
    """Training configuration settings"""
    # Training parameters
    output_dir: str = "./medical-llm-results"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    
    # Memory optimization
    fp16: bool = True
    gradient_checkpointing: bool = True
    
    # Logging and saving
    logging_steps: int = 5
    save_strategy: str = "epoch"
    eval_strategy: str = "no"
    
    # Sequence settings
    max_seq_length: int = 512
    packing: bool = False
    
    # Monitoring
    report_to: str = "wandb"  # or "none"
    run_name: str = "medical-llm-finetune"
    remove_unused_columns: bool = False

@dataclass
class DataConfig:
    """Dataset configuration"""
    # Primary dataset
    dataset_name: str = "lavita/medical-qa-datasets"
    dataset_config: str = "all-processed"
    text_field: str = "text"
    
    # Preprocessing
    max_samples: int = 10000  # For testing, increase for full training
    train_split_ratio: float = 0.8
    
    # Dummy data for testing
    use_dummy_data: bool = True
    dummy_data_size: int = 10

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Test settings
    max_new_tokens: int = 100
    temperature: float = 0.7
    do_sample: bool = True
    
    # Evaluation datasets
    eval_datasets: List[str] = None
    
    def __post_init__(self):
        if self.eval_datasets is None:
            self.eval_datasets = [
                "MedQA",
                "MedMCQA", 
                "PubMedQA"
            ]

@dataclass
class SystemConfig:
    """System and hardware configuration"""
    # Hardware settings
    device: str = "auto"
    cuda_available: bool = torch.cuda.is_available()
    
    # Memory management
    max_memory_gb: float = 20.0  # RTX 3090 safe limit
    
    # Paths
    models_dir: str = "./models"
    data_dir: str = "./data"
    logs_dir: str = "./logs"
    experiments_dir: str = "./experiments"
    evaluation_dir: str = "./evaluation"

class MedicalLLMConfig:
    """Main configuration class that combines all configs"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.lora = LoRAConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.evaluation = EvaluationConfig()
        self.system = SystemConfig()
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings"""
        # Check CUDA availability
        if not self.system.cuda_available:
            print("⚠️  Warning: CUDA not available, training will be slow")
            self.training.fp16 = False
        
        # Adjust batch size based on available memory
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if total_memory < 16:
                self.training.per_device_train_batch_size = 1
                print(f"⚠️  Low GPU memory ({total_memory:.1f}GB), reducing batch size to 1")
    
    def get_model_config_dict(self):
        """Get model configuration as dictionary for BitsAndBytesConfig"""
        return {
            "load_in_4bit": self.model.load_in_4bit,
            "bnb_4bit_quant_type": self.model.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": self.model.bnb_4bit_compute_dtype,
            "bnb_4bit_use_double_quant": self.model.bnb_4bit_use_double_quant,
        }
    
    def get_lora_config_dict(self):
        """Get LoRA configuration as dictionary"""
        return {
            "r": self.lora.r,
            "lora_alpha": self.lora.lora_alpha,
            "target_modules": self.lora.target_modules,
            "lora_dropout": self.lora.lora_dropout,
            "bias": self.lora.bias,
            "task_type": self.lora.task_type,
        }
    
    def get_training_args_dict(self):
        """Get training arguments as dictionary"""
        return {
            "output_dir": self.training.output_dir,
            "num_train_epochs": self.training.num_train_epochs,
            "per_device_train_batch_size": self.training.per_device_train_batch_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "learning_rate": self.training.learning_rate,
            "weight_decay": self.training.weight_decay,
            "warmup_ratio": self.training.warmup_ratio,
            "max_grad_norm": self.training.max_grad_norm,
            "fp16": self.training.fp16,
            "gradient_checkpointing": self.training.gradient_checkpointing,
            "logging_steps": self.training.logging_steps,
            "save_strategy": self.training.save_strategy,
            "eval_strategy": self.training.eval_strategy,
            "report_to": self.training.report_to if self.training.report_to != "none" else [],
            "run_name": self.training.run_name,
            "remove_unused_columns": self.training.remove_unused_columns,
        }
    
    def print_config(self):
        """Print current configuration"""
        print("🔧 Medical LLM Configuration:")
        print("=" * 50)
        print(f"📱 Model: {self.model.base_model_name}")
        print(f"🎯 LoRA Rank: {self.lora.r}")
        print(f"🔄 Epochs: {self.training.num_train_epochs}")
        print(f"📦 Batch Size: {self.training.per_device_train_batch_size}")
        print(f"📚 Max Samples: {self.data.max_samples}")
        print(f"💾 GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "CPU Only")
        print("=" * 50)

# Global configuration instance
config = MedicalLLMConfig() 