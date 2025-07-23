"""
Medical LLM Model Setup Module
Handles model loading, quantization, and LoRA configuration
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Tuple, Optional
import logging

from .config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Manages model loading, tokenizer setup, and LoRA configuration"""
    
    def __init__(self, cfg=None):
        """
        Initialize ModelManager with configuration
        
        Args:
            cfg: Configuration object, defaults to global config
        """
        self.cfg = cfg if cfg else config
        self.model = None
        self.tokenizer = None
        self.is_quantized = False
        
    def setup_model_and_tokenizer(self, model_name: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Setup model and tokenizer with 4-bit quantization for RTX 3090
        
        Args:
            model_name: Name of the model to load, defaults to config
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if model_name is None:
            model_name = self.cfg.model.base_model_name
            
        logger.info(f"Loading model: {model_name}")
        
        try:
            # Setup quantization configuration
            bnb_config = self._create_quantization_config()
            
            # Load tokenizer
            tokenizer = self._load_tokenizer(model_name)
            
            # Load model with quantization
            model = self._load_model(model_name, bnb_config)
            
            # Store for later use
            self.model = model
            self.tokenizer = tokenizer
            self.is_quantized = True
            
            logger.info("✅ Model loaded successfully!")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            logger.info("💡 Trying fallback model...")
            
            # Try fallback model
            fallback_name = self.cfg.model.fallback_model_name
            return self._load_fallback_model(fallback_name)
    
    def _create_quantization_config(self) -> BitsAndBytesConfig:
        """Create 4-bit quantization configuration"""
        return BitsAndBytesConfig(**self.cfg.get_model_config_dict())
    
    def _load_tokenizer(self, model_name: str) -> AutoTokenizer:
        """Load and configure tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not available
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
            
        return tokenizer
    
    def _load_model(self, model_name: str, bnb_config: BitsAndBytesConfig) -> AutoModelForCausalLM:
        """Load model with quantization configuration"""
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=self.cfg.system.device,
            trust_remote_code=self.cfg.model.trust_remote_code,
            use_safetensors=self.cfg.model.use_safetensors,
        )
    
    def _load_fallback_model(self, fallback_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load fallback model if primary model fails"""
        try:
            bnb_config = self._create_quantization_config()
            tokenizer = self._load_tokenizer(fallback_name)
            model = self._load_model(fallback_name, bnb_config)
            
            # Store for later use
            self.model = model
            self.tokenizer = tokenizer
            self.is_quantized = True
            
            logger.info(f"✅ Fallback model ({fallback_name}) loaded successfully!")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Fallback model also failed: {e}")
            raise RuntimeError("Failed to load both primary and fallback models")
    
    def setup_lora_model(self, model: Optional[AutoModelForCausalLM] = None) -> AutoModelForCausalLM:
        """
        Setup LoRA configuration and prepare model for training
        
        Args:
            model: Model to apply LoRA to, defaults to stored model
            
        Returns:
            Model with LoRA applied
        """
        if model is None:
            model = self.model
            
        if model is None:
            raise ValueError("No model available. Call setup_model_and_tokenizer first.")
        
        logger.info("Setting up LoRA configuration...")
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Create LoRA configuration
        lora_config = self._create_lora_config()
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        # Log parameter information
        self._log_parameter_info(model)
        
        # Update stored model
        self.model = model
        
        return model
    
    def _create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration"""
        return LoraConfig(**self.cfg.get_lora_config_dict())
    
    def _log_parameter_info(self, model: AutoModelForCausalLM):
        """Log information about model parameters"""
        trainable_params = model.num_parameters()
        total_params = model.base_model.num_parameters()
        trainable_percentage = 100 * trainable_params / total_params
        
        logger.info(f"📈 Trainable parameters: {trainable_params:,}")
        logger.info(f"🔒 Total parameters: {total_params:,}")
        logger.info(f"📊 Trainable %: {trainable_percentage:.2f}%")
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        if self.model is None:
            return {"status": "No model loaded"}
        
        try:
            trainable_params = self.model.num_parameters()
            total_params = self.model.base_model.num_parameters()
            
            return {
                "status": "Model loaded",
                "quantized": self.is_quantized,
                "trainable_parameters": trainable_params,
                "total_parameters": total_params,
                "trainable_percentage": 100 * trainable_params / total_params,
                "model_size_mb": total_params * 4 / (1024 * 1024),  # Approximate
            }
        except Exception as e:
            return {"status": f"Error getting model info: {e}"}
    
    def save_model(self, save_path: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        try:
            self.model.save_pretrained(save_path)
            if self.tokenizer:
                self.tokenizer.save_pretrained(save_path)
            logger.info(f"💾 Model saved to {save_path}")
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            raise
    
    def load_trained_model(self, model_path: str, base_model_name: Optional[str] = None):
        """
        Load a previously trained model
        
        Args:
            model_path: Path to the saved model
            base_model_name: Base model name, defaults to config
        """
        from peft import PeftModel
        
        if base_model_name is None:
            base_model_name = self.cfg.model.base_model_name
        
        try:
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map=self.cfg.system.device,
                use_safetensors=self.cfg.model.use_safetensors
            )
            
            # Load PEFT model
            model = PeftModel.from_pretrained(base_model, model_path)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.model = model
            self.tokenizer = tokenizer
            
            logger.info(f"✅ Trained model loaded from {model_path}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Error loading trained model: {e}")
            raise

# Convenience functions for backward compatibility
def setup_model_and_tokenizer(model_name: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Convenience function to setup model and tokenizer
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Tuple of (model, tokenizer)
    """
    manager = ModelManager()
    return manager.setup_model_and_tokenizer(model_name)

def setup_lora_model(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """
    Convenience function to setup LoRA on a model
    
    Args:
        model: Model to apply LoRA to
        
    Returns:
        Model with LoRA applied
    """
    manager = ModelManager()
    return manager.setup_lora_model(model)

def get_model_memory_usage() -> dict:
    """Get current GPU memory usage information"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    try:
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3      # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        return {
            "allocated_gb": round(allocated, 2),
            "cached_gb": round(cached, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - cached, 2),
            "utilization_percent": round((cached / total) * 100, 1)
        }
    except Exception as e:
        return {"error": f"Error getting memory info: {e}"} 