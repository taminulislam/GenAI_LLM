#!/usr/bin/env python3
"""
Comprehensive Medical LLM Project - Step 1: Environment Setup
Sets up the environment and verifies all dependencies are working correctly.
"""

import sys
import os
import torch
import datetime

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    """Setup environment and verify dependencies."""
    print("=== Comprehensive Medical LLM Project - Environment Setup ===")
    print(f"Started at: {datetime.datetime.now()}")
    
    results = []
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"\n1. Python Version: {python_version}")
    results.append(f"Python Version: {python_version}")
    
    # Check PyTorch and CUDA
    print(f"\n2. PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    results.append(f"PyTorch Version: {torch.__version__}")
    results.append(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            print(f"   Device {i}: {device_name}")
            results.append(f"CUDA Device {i}: {device_name}")
    
    # Check required packages
    required_packages = [
        'transformers', 'datasets', 'torch', 'numpy', 'pandas', 
        'sklearn', 'accelerate', 'peft', 'bitsandbytes', 'evaluate'
    ]
    
    print(f"\n3. Checking Required Packages:")
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✓ {package}")
            results.append(f"✓ {package}")
        except ImportError:
            print(f"   ✗ {package} - Missing!")
            results.append(f"✗ {package} - Missing!")
    
    # Test imports from our modules
    print(f"\n4. Testing Project Modules:")
    try:
        from config import ComprehensiveConfig
        print("   ✓ config.ComprehensiveConfig")
        results.append("✓ config.ComprehensiveConfig")
    except Exception as e:
        print(f"   ✗ config.ComprehensiveConfig - {e}")
        results.append(f"✗ config.ComprehensiveConfig - {e}")
    
    try:
        from data_loader import ComprehensiveDataLoader
        print("   ✓ data_loader.ComprehensiveDataLoader")
        results.append("✓ data_loader.ComprehensiveDataLoader")
    except Exception as e:
        print(f"   ✗ data_loader.ComprehensiveDataLoader - {e}")
        results.append(f"✗ data_loader.ComprehensiveDataLoader - {e}")
    
    try:
        from model_setup import ComprehensiveModelSetup
        print("   ✓ model_setup.ComprehensiveModelSetup")
        results.append("✓ model_setup.ComprehensiveModelSetup")
    except Exception as e:
        print(f"   ✗ model_setup.ComprehensiveModelSetup - {e}")
        results.append(f"✗ model_setup.ComprehensiveModelSetup - {e}")
    
    try:
        from trainer import ComprehensiveTrainer
        print("   ✓ trainer.ComprehensiveTrainer")
        results.append("✓ trainer.ComprehensiveTrainer")
    except Exception as e:
        print(f"   ✗ trainer.ComprehensiveTrainer - {e}")
        results.append(f"✗ trainer.ComprehensiveTrainer - {e}")
        
    try:
        from evaluator import ComprehensiveEvaluator
        print("   ✓ evaluator.ComprehensiveEvaluator")
        results.append("✓ evaluator.ComprehensiveEvaluator")
    except Exception as e:
        print(f"   ✗ evaluator.ComprehensiveEvaluator - {e}")
        results.append(f"✗ evaluator.ComprehensiveEvaluator - {e}")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"step1_environment_setup_{timestamp}.txt")
    
    with open(output_file, 'w') as f:
        f.write("=== Comprehensive Medical LLM Project - Environment Setup Results ===\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n\n")
        for result in results:
            f.write(f"{result}\n")
    
    print(f"\n=== Environment Setup Complete ===")
    print(f"Results saved to: {output_file}")
    print(f"Completed at: {datetime.datetime.now()}")
    
    return output_file

if __name__ == "__main__":
    main() 