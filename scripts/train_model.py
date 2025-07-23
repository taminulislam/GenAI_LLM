#!/usr/bin/env python3
"""
Medical LLM Training Script
Trains medical language models using the modular training pipeline
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import config, MedicalLLMConfig
from src.model_setup import ModelManager
from src.data_loader import MedicalDataLoader
from src.trainer import MedicalLLMTrainer, setup_training_environment
from src.evaluator import MedicalLLMEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalLLMTrainingPipeline:
    """Complete training pipeline for medical LLMs"""
    
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg else config
        self.model_manager = None
        self.data_loader = None
        self.trainer = None
        self.evaluator = None
        self.training_results = {}
        
    def setup_experiment_directory(self, experiment_name: str = None) -> str:
        """Setup experiment directory with timestamp"""
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"medical_llm_{timestamp}"
        
        experiment_dir = Path("experiments") / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (experiment_dir / "checkpoints").mkdir(exist_ok=True)
        (experiment_dir / "logs").mkdir(exist_ok=True)
        (experiment_dir / "models").mkdir(exist_ok=True)
        (experiment_dir / "evaluation").mkdir(exist_ok=True)
        
        logger.info(f"📁 Experiment directory created: {experiment_dir}")
        return str(experiment_dir)
    
    def save_experiment_config(self, experiment_dir: str, args: argparse.Namespace):
        """Save experiment configuration"""
        config_data = {
            "experiment_info": {
                "name": Path(experiment_dir).name,
                "start_time": datetime.now().isoformat(),
                "command_line_args": vars(args)
            },
            "model_config": {
                "base_model": self.cfg.model.base_model_name,
                "use_quantization": self.cfg.model.load_in_4bit,
                "lora_r": self.cfg.lora.r,
                "lora_alpha": self.cfg.lora.lora_alpha,
                "lora_dropout": self.cfg.lora.lora_dropout
            },
            "training_config": {
                "num_epochs": self.cfg.training.num_epochs,
                "batch_size": self.cfg.training.batch_size,
                "learning_rate": self.cfg.training.learning_rate,
                "max_seq_length": self.cfg.training.max_seq_length,
                "use_fp16": self.cfg.training.use_fp16
            },
            "data_config": {
                "use_dummy_data": self.cfg.data.use_dummy_data,
                "max_samples": self.cfg.data.max_samples
            }
        }
        
        config_file = Path(experiment_dir) / "experiment_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"💾 Experiment config saved to {config_file}")
    
    def load_datasets(self, data_dir: str = "data", dataset_name: str = None):
        """Load and prepare datasets"""
        logger.info("📚 Loading datasets...")
        
        self.data_loader = MedicalDataLoader(self.cfg)
        
        if dataset_name:
            # Load specific dataset
            dataset_path = Path(data_dir) / f"{dataset_name}_processed"
            if dataset_path.exists():
                logger.info(f"Loading processed dataset: {dataset_path}")
                self.data_loader.load_processed_dataset(str(dataset_path))
            else:
                logger.info(f"Processing dataset: {dataset_name}")
                self.data_loader.load_dataset(dataset_name)
                self.data_loader.preprocess_dataset()
        else:
            # Load default dataset
            self.data_loader.load_dataset()
            self.data_loader.preprocess_dataset()
        
        logger.info(f"✅ Dataset loaded: {len(self.data_loader.processed_dataset)} samples")
    
    def setup_model(self, model_name: str = None):
        """Setup model and tokenizer"""
        logger.info("🤖 Setting up model...")
        
        if model_name:
            # Override default model
            original_model = self.cfg.model.base_model_name
            self.cfg.model.base_model_name = model_name
            logger.info(f"Using custom model: {model_name}")
        
        self.model_manager = ModelManager(self.cfg)
        self.model_manager.setup_model_and_tokenizer()
        self.model_manager.setup_lora_model()
        
        # Log model info
        memory_usage = self.model_manager.get_model_memory_usage()
        logger.info(f"✅ Model ready: {memory_usage}")
    
    def run_training(self, experiment_dir: str) -> Dict[str, Any]:
        """Run the training process"""
        logger.info("🚀 Starting training...")
        
        # Setup trainer
        self.trainer = MedicalLLMTrainer(self.cfg)
        
        # Run training
        training_results = self.trainer.train(
            model_manager=self.model_manager,
            data_loader=self.data_loader,
            output_dir=experiment_dir
        )
        
        self.training_results = training_results
        
        # Save training summary
        summary_file = Path(experiment_dir) / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        logger.info(f"✅ Training completed!")
        logger.info(f"   Final loss: {training_results.get('train_loss', 'N/A'):.4f}")
        logger.info(f"   Model saved to: {training_results.get('final_model_path', 'N/A')}")
        
        return training_results
    
    def run_evaluation(self, experiment_dir: str) -> Dict[str, Any]:
        """Run post-training evaluation"""
        logger.info("📊 Running evaluation...")
        
        self.evaluator = MedicalLLMEvaluator(self.cfg)
        
        # Use the trained model from training
        trained_model_path = self.training_results.get('final_model_path')
        if trained_model_path and Path(trained_model_path).exists():
            evaluation_results = self.evaluator.run_comprehensive_evaluation(trained_model_path)
        else:
            # Fallback to using the model manager directly
            evaluation_results = self.evaluator.run_comprehensive_evaluation()
        
        # Save evaluation results to experiment directory
        eval_file = Path(experiment_dir) / "evaluation_results.json"
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Create evaluation report
        report = self.evaluator.create_evaluation_report(evaluation_results)
        report_file = Path(experiment_dir) / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"✅ Evaluation completed!")
        overall_accuracy = evaluation_results.get('summary', {}).get('overall_accuracy', 0)
        logger.info(f"   Overall accuracy: {overall_accuracy:.3f}")
        
        return evaluation_results
    
    def run_complete_pipeline(self, 
                            experiment_name: str = None,
                            model_name: str = None,
                            dataset_name: str = None,
                            data_dir: str = "data",
                            skip_evaluation: bool = False) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        logger.info("🏥 Medical LLM Training Pipeline")
        logger.info("=" * 60)
        
        # Setup experiment
        experiment_dir = self.setup_experiment_directory(experiment_name)
        
        # Setup components
        self.load_datasets(data_dir, dataset_name)
        self.setup_model(model_name)
        
        # Save experiment configuration
        args = argparse.Namespace(
            experiment_name=experiment_name,
            model_name=model_name,
            dataset_name=dataset_name,
            data_dir=data_dir,
            skip_evaluation=skip_evaluation
        )
        self.save_experiment_config(experiment_dir, args)
        
        # Run training
        training_results = self.run_training(experiment_dir)
        
        # Run evaluation (optional)
        evaluation_results = {}
        if not skip_evaluation:
            evaluation_results = self.run_evaluation(experiment_dir)
        
        # Combine results
        complete_results = {
            "experiment_dir": experiment_dir,
            "training_results": training_results,
            "evaluation_results": evaluation_results,
            "success": True
        }
        
        logger.info("🎉 Pipeline completed successfully!")
        logger.info(f"📁 Results saved to: {experiment_dir}")
        
        return complete_results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train medical LLM models")
    
    # Experiment settings
    parser.add_argument("--experiment-name", type=str, help="Name for this experiment")
    parser.add_argument("--model", type=str, help="Model to use (overrides config)")
    parser.add_argument("--dataset", type=str, help="Specific dataset to use")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing datasets")
    
    # Training settings
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--max-length", type=int, help="Maximum sequence length")
    
    # Mode settings
    parser.add_argument("--quick-test", action="store_true", help="Run quick test with dummy data")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip post-training evaluation")
    parser.add_argument("--dummy-data", action="store_true", help="Use dummy data for training")
    
    # System settings
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU-only training")
    
    return parser.parse_args()

def apply_cli_overrides(args: argparse.Namespace):
    """Apply command line argument overrides to config"""
    if args.epochs:
        config.training.num_epochs = args.epochs
        logger.info(f"Override: epochs = {args.epochs}")
    
    if args.batch_size:
        config.training.batch_size = args.batch_size
        logger.info(f"Override: batch_size = {args.batch_size}")
    
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
        logger.info(f"Override: learning_rate = {args.learning_rate}")
    
    if args.max_length:
        config.training.max_seq_length = args.max_length
        logger.info(f"Override: max_seq_length = {args.max_length}")
    
    if args.dummy_data or args.quick_test:
        config.data.use_dummy_data = True
        logger.info("Override: using dummy data")
    
    if args.quick_test:
        # Quick test settings
        config.training.num_epochs = 1
        config.data.max_samples = 100
        config.training.logging_steps = 5
        logger.info("Quick test mode enabled")
    
    if args.cpu_only:
        # Force CPU settings (note: this might not work with all models)
        config.model.load_in_4bit = False
        config.training.use_fp16 = False
        logger.info("CPU-only mode enabled")

def main():
    """Main training function"""
    args = parse_arguments()
    
    # Setup environment
    logger.info("🔧 Setting up training environment...")
    if not setup_training_environment():
        logger.error("❌ Environment setup failed!")
        sys.exit(1)
    
    # Apply CLI overrides
    apply_cli_overrides(args)
    
    try:
        # Initialize pipeline
        pipeline = MedicalLLMTrainingPipeline()
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            experiment_name=args.experiment_name,
            model_name=args.model,
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            skip_evaluation=args.skip_evaluation
        )
        
        # Print final summary
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        
        training_results = results["training_results"]
        logger.info(f"Experiment: {Path(results['experiment_dir']).name}")
        logger.info(f"Model: {config.model.base_model_name}")
        logger.info(f"Training Loss: {training_results.get('train_loss', 'N/A')}")
        logger.info(f"Training Steps: {training_results.get('train_steps', 'N/A')}")
        logger.info(f"Dataset Size: {training_results.get('dataset_size', 'N/A')}")
        
        if results["evaluation_results"]:
            eval_summary = results["evaluation_results"].get("summary", {})
            logger.info(f"Evaluation Accuracy: {eval_summary.get('overall_accuracy', 'N/A'):.3f}")
        
        logger.info(f"Results Directory: {results['experiment_dir']}")
        logger.info("=" * 60)
        
        # Next steps
        logger.info("\n🎯 Next Steps:")
        logger.info("1. Review training logs and evaluation results")
        logger.info("2. Run 'python scripts/evaluate_model.py' for detailed evaluation")
        logger.info("3. Use the trained model for inference")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 