"""
Medical LLM Fine-tuning Package
Author: Your Name
Version: 1.0.0
"""

from .config import config, MedicalLLMConfig
from .model_setup import ModelManager, setup_model_and_tokenizer, setup_lora_model, get_model_memory_usage
from .data_loader import MedicalDataLoader, load_medical_data, preprocess_medical_data
from .trainer import MedicalLLMTrainer, quick_train, setup_training_environment
from .evaluator import MedicalLLMEvaluator, quick_evaluate, compare_models

__version__ = "1.0.0"
__author__ = "Medical LLM Research Team"

__all__ = [
    'config',
    'MedicalLLMConfig', 
    'ModelManager',
    'setup_model_and_tokenizer',
    'setup_lora_model',
    'get_model_memory_usage',
    'MedicalDataLoader',
    'load_medical_data',
    'preprocess_medical_data',
    'MedicalLLMTrainer',
    'quick_train',
    'setup_training_environment',
    'MedicalLLMEvaluator',
    'quick_evaluate',
    'compare_models'
] 