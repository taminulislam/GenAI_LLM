#!/usr/bin/env python3
"""
Medical LLM Environment Setup Script
Sets up and verifies the training environment for medical LLM fine-tuning
"""

import os
import sys
import subprocess
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src import setup_training_environment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu():
    """Check GPU availability and specs"""
    logger.info("🔍 Checking GPU Configuration...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"✅ GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Test CUDA operations
        try:
            test_tensor = torch.randn(100, 100).cuda()
            test_result = torch.mm(test_tensor, test_tensor.t())
            logger.info("✅ CUDA operations working correctly")
            del test_tensor, test_result
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"❌ CUDA test failed: {e}")
            return False
        
        return True
    else:
        logger.warning("⚠️ CUDA not available - training will be slow")
        return False

def check_packages():
    """Check if all required packages are installed"""
    logger.info("🔍 Checking Required Packages...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'accelerate', 
        'bitsandbytes', 'peft', 'trl', 'wandb', 'evaluate', 
        'scikit-learn', 'numpy', 'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            logger.error(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Install missing packages with:")
        logger.info(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_directories():
    """Check and create necessary directories"""
    logger.info("🔍 Checking Project Directories...")
    
    required_dirs = [
        "data", "models", "experiments", "evaluation", "logs",
        "src", "scripts", "notebooks"
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            logger.info(f"✅ {dir_name}/ directory exists")
        else:
            logger.info(f"📁 Creating {dir_name}/ directory...")
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ {dir_name}/ directory created")
    
    return True

def install_requirements():
    """Install required packages"""
    logger.info("📦 Installing Required Packages...")
    
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.10.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.39.0",
        "peft>=0.4.0",
        "trl>=0.4.0",
        "wandb>=0.15.0",
        "evaluate>=0.4.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0"
    ]
    
    try:
        for requirement in requirements:
            logger.info(f"Installing {requirement}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", requirement, "--quiet"
            ])
        
        logger.info("✅ All packages installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Package installation failed: {e}")
        return False

def setup_wandb():
    """Setup Weights & Biases (optional)"""
    logger.info("🔧 Setting up Weights & Biases...")
    
    try:
        import wandb
        
        # Check if already logged in
        if wandb.api.api_key:
            logger.info("✅ Weights & Biases already configured")
            return True
        
        # Prompt for setup
        response = input("Would you like to set up Weights & Biases for experiment tracking? (y/n): ")
        if response.lower() in ['y', 'yes']:
            logger.info("Please run 'wandb login' in your terminal to set up W&B")
            logger.info("You can get your API key from: https://wandb.ai/settings")
        else:
            logger.info("⚠️ Weights & Biases setup skipped")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ W&B setup issue: {e}")
        return True  # Not critical

def run_system_test():
    """Run a quick system test"""
    logger.info("🧪 Running System Test...")
    
    try:
        # Test imports
        from src.config import config
        from src.model_setup import ModelManager
        from src.data_loader import MedicalDataLoader
        from src.trainer import MedicalLLMTrainer
        from src.evaluator import MedicalLLMEvaluator
        
        logger.info("✅ All modules imported successfully")
        
        # Test configuration
        logger.info(f"✅ Default model: {config.model.base_model_name}")
        logger.info(f"✅ Training epochs: {config.training.num_epochs}")
        logger.info(f"✅ Batch size: {config.training.batch_size}")
        
        # Test memory availability
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✅ GPU Memory: {total_memory:.1f}GB available")
            
            if total_memory < 10:
                logger.warning("⚠️ Low GPU memory - consider reducing batch size")
        
        logger.info("🎉 System test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ System test failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("🚀 Medical LLM Environment Setup")
    logger.info("=" * 50)
    
    success = True
    
    # Step 1: Check directories
    if not check_directories():
        success = False
    
    # Step 2: Check packages
    if not check_packages():
        install_choice = input("Install missing packages? (y/n): ")
        if install_choice.lower() in ['y', 'yes']:
            if not install_requirements():
                success = False
        else:
            success = False
    
    # Step 3: Check GPU
    gpu_available = check_gpu()
    if not gpu_available:
        logger.warning("Training will be slower without GPU acceleration")
    
    # Step 4: Setup W&B
    setup_wandb()
    
    # Step 5: Run system test
    if not run_system_test():
        success = False
    
    # Step 6: Setup training environment
    if not setup_training_environment():
        success = False
    
    # Final status
    logger.info("=" * 50)
    if success:
        logger.info("🎉 Environment setup completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Run 'python scripts/download_data.py' to download datasets")
        logger.info("2. Run 'python scripts/train_model.py' to start training")
        logger.info("3. Run 'python scripts/evaluate_model.py' to evaluate models")
    else:
        logger.error("❌ Environment setup failed!")
        logger.error("Please fix the issues above and run again")
        sys.exit(1)

if __name__ == "__main__":
    main() 