"""
Medical LLM Trainer Module
Handles training logic, SFTTrainer setup, and training monitoring
"""

import os
import logging
import torch
import wandb
from typing import Optional, Dict, Any, Tuple
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
from datasets import Dataset
import json
from datetime import datetime

try:
    from .config import config
    from .model_setup import ModelManager
    from .data_loader import MedicalDataLoader
except ImportError:
    from config import config
    from model_setup import ModelManager
    from data_loader import MedicalDataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalLLMTrainer:
    """Handles medical LLM training with SFTTrainer"""
    
    def __init__(self, cfg=None):
        """
        Initialize trainer with configuration
        
        Args:
            cfg: Configuration object, defaults to global config
        """
        self.cfg = cfg if cfg else config
        self.model_manager = None
        self.data_loader = None
        self.trainer = None
        self.training_stats = {}
        
    def setup_training_arguments(self, output_dir: str = None) -> TrainingArguments:
        """
        Setup training arguments with optimal settings for RTX 3090
        
        Args:
            output_dir: Output directory for training results
            
        Returns:
            TrainingArguments object
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./experiments/medical_llm_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            # Output settings
            output_dir=output_dir,
            run_name=f"medical-llm-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            
            # Training parameters
            num_train_epochs=self.cfg.training.num_train_epochs,
            per_device_train_batch_size=self.cfg.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.cfg.training.per_device_train_batch_size,
            gradient_accumulation_steps=self.cfg.training.gradient_accumulation_steps,
            learning_rate=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
            max_grad_norm=self.cfg.training.max_grad_norm,
            warmup_ratio=self.cfg.training.warmup_ratio,
            
            # Memory optimization
            fp16=self.cfg.training.fp16,
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            
            # Logging and saving
            logging_dir=f"{output_dir}/logs",
            logging_steps=self.cfg.training.logging_steps,
            save_strategy="epoch",
            eval_strategy=self.cfg.training.eval_strategy,
            save_total_limit=3,
            load_best_model_at_end=True if self.cfg.training.eval_strategy != "no" else False,
            metric_for_best_model="eval_loss" if self.cfg.training.eval_strategy != "no" else None,
            
            # Monitoring
            report_to="wandb" if self._check_wandb() else "none",
            remove_unused_columns=False,
            

        )
        
        logger.info(f"Training arguments configured for output: {output_dir}")
        return training_args
    
    def _check_wandb(self) -> bool:
        """Check if wandb is available and configured"""
        try:
            return wandb.api.api_key is not None
        except:
            return False
    
    def setup_trainer(self, 
                     model_manager: ModelManager,
                     data_loader: MedicalDataLoader,
                     training_args: TrainingArguments) -> SFTTrainer:
        """
        Setup SFTTrainer with model, data, and training arguments
        
        Args:
            model_manager: ModelManager instance with loaded model
            data_loader: MedicalDataLoader instance with processed data
            training_args: TrainingArguments object
            
        Returns:
            Configured SFTTrainer
        """
        if not model_manager.model or not model_manager.tokenizer:
            raise ValueError("Model manager must have loaded model and tokenizer")
        
        if not data_loader.processed_dataset:
            raise ValueError("Data loader must have processed dataset")
        
        # Split dataset for training and evaluation
        train_dataset = data_loader.processed_dataset
        eval_dataset = None
        
        if self.cfg.training.eval_strategy != "no" and len(train_dataset) > 100:
            # Use 10% for evaluation
            split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
            logger.info(f"Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval")
        
        # Setup trainer
        trainer = SFTTrainer(
            model=model_manager.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )
        
        self.trainer = trainer
        self.model_manager = model_manager
        self.data_loader = data_loader
        
        logger.info("SFTTrainer configured successfully")
        return trainer
    
    def train(self, 
              model_manager: ModelManager = None,
              data_loader: MedicalDataLoader = None,
              output_dir: str = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Args:
            model_manager: ModelManager instance (optional, will create if None)
            data_loader: MedicalDataLoader instance (optional, will create if None)
            output_dir: Output directory for results
            
        Returns:
            Dictionary containing training statistics and results
        """
        logger.info("Starting Medical LLM Training Pipeline...")
        
        # Initialize components if not provided
        if model_manager is None:
            model_manager = ModelManager(self.cfg)
            model_manager.setup_model_and_tokenizer()
            model_manager.setup_lora_model()
        
        if data_loader is None:
            data_loader = MedicalDataLoader(self.cfg)
            dataset = data_loader.load_medical_dataset()
            data_loader.dataset = dataset
            data_loader.processed_dataset = data_loader.preprocess_dataset(dataset)
        
        # Setup training
        training_args = self.setup_training_arguments(output_dir)
        trainer = self.setup_trainer(model_manager, data_loader, training_args)
        
        # Initialize wandb if available
        if self._check_wandb():
            wandb.init(
                project="medical-llm-finetuning",
                name=training_args.run_name,
                config={
                    "model_name": self.cfg.model.base_model_name,
                    "dataset_size": len(data_loader.processed_dataset),
                    "batch_size": self.cfg.training.per_device_train_batch_size,
                    "learning_rate": self.cfg.training.learning_rate,
                    "lora_r": self.cfg.lora.r,
                    "lora_alpha": self.cfg.lora.lora_alpha,
                }
            )
        
        # Run training
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save model and tokenizer
        logger.info("Saving trained model...")
        final_model_path = os.path.join(training_args.output_dir, "final_model")
        model_manager.save_model(final_model_path)
        
        # Collect training statistics
        self.training_stats = {
            "train_loss": train_result.training_loss,
            "train_steps": train_result.global_step,
            "epochs_trained": getattr(train_result, 'epoch', self.cfg.training.num_train_epochs),
            "output_dir": training_args.output_dir,
            "final_model_path": final_model_path,
            "dataset_size": len(data_loader.processed_dataset),
            "model_name": self.cfg.model.base_model_name,
        }
        
        # Add evaluation metrics if available
        if hasattr(train_result, 'metrics'):
            self.training_stats.update(train_result.metrics)
        
        # Save training statistics
        stats_file = os.path.join(training_args.output_dir, "training_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(self.training_stats, f, indent=2, default=str)
        
        logger.info(f"Training completed! Results saved to: {training_args.output_dir}")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        
        return self.training_stats
    
    def evaluate(self, eval_dataset: Dataset = None) -> Dict[str, float]:
        """
        Evaluate the trained model
        
        Args:
            eval_dataset: Dataset for evaluation (optional)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.trainer:
            raise ValueError("Trainer not initialized. Run train() first.")
        
        logger.info("Running model evaluation...")
        
        if eval_dataset:
            eval_results = self.trainer.evaluate(eval_dataset=eval_dataset)
        else:
            eval_results = self.trainer.evaluate()
        
        logger.info(f"Evaluation results: {eval_results}")
        return eval_results
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage
        
        Returns:
            Dictionary with memory usage statistics
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "max_memory_gb": max_memory,
                "device": torch.cuda.get_device_name()
            }
        else:
            return {"status": "CUDA not available"}

# Convenience functions
def quick_train(output_dir: str = None, 
                use_dummy_data: bool = True,
                num_epochs: int = 2) -> Dict[str, Any]:
    """
    Quick training function for testing and development
    
    Args:
        output_dir: Output directory for results
        use_dummy_data: Whether to use dummy data for quick testing
        num_epochs: Number of training epochs
        
    Returns:
        Training statistics dictionary
    """
    # Temporarily modify config for quick training
    original_epochs = config.training.num_epochs
    original_use_dummy = config.data.use_dummy_data
    
    config.training.num_epochs = num_epochs
    config.data.use_dummy_data = use_dummy_data
    
    try:
        trainer = MedicalLLMTrainer()
        results = trainer.train(output_dir=output_dir)
        return results
    finally:
        # Restore original config
        config.training.num_epochs = original_epochs
        config.data.use_dummy_data = original_use_dummy

def setup_training_environment():
    """Setup and verify training environment"""
    logger.info("Setting up Medical LLM Training Environment...")
    
    # Check CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✅ CUDA Device: {device_name} ({memory_gb:.1f}GB)")
    else:
        logger.warning("⚠️  CUDA not available - training will be slow")
    
    # Check required packages
    try:
        import transformers, peft, trl, datasets
        logger.info("✅ All required packages imported successfully")
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    
    # Create necessary directories
    for dir_name in ["experiments", "models", "data", "logs"]:
        os.makedirs(dir_name, exist_ok=True)
    
    logger.info("✅ Training environment ready!")
    return True 