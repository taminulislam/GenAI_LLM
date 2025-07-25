#!/usr/bin/env python3
"""
Comprehensive Medical LLM Project - Step 3: Model Setup
Sets up multiple models with different configurations for comprehensive training.
"""

import sys
import os
import datetime
import torch

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    """Set up multiple models for training."""
    print("=== Comprehensive Medical LLM Project - Model Setup ===")
    print(f"Started at: {datetime.datetime.now()}")
    
    results = []
    
    try:
        from config import ComprehensiveConfig
        from model_setup import ComprehensiveModelSetup
        
        # Initialize configuration
        config = ComprehensiveConfig()
        print(f"\n1. Configuration loaded successfully")
        print(f"   Models to setup: {', '.join(config.model_configs.keys())}")
        results.append(f"Configuration loaded: {len(config.model_configs)} models configured")
        
        # Initialize model setup
        model_setup = ComprehensiveModelSetup(config)
        print(f"\n2. Model setup initialized")
        results.append("Model setup initialized successfully")
        
        # Check GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\n3. GPU Memory Available: {gpu_memory:.2f} GB")
            results.append(f"GPU Memory: {gpu_memory:.2f} GB")
        
        # Setup each model
        print(f"\n4. Setting up models...")
        model_info = {}
        
        for model_name in config.model_configs.keys():
            print(f"\n   Setting up {model_name}...")
            try:
                model, tokenizer = model_setup.setup_model(model_name)
                
                # Get model statistics
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                print(f"     ✓ {model_name} loaded successfully")
                print(f"     Total parameters: {total_params:,}")
                print(f"     Trainable parameters: {trainable_params:,}")
                print(f"     Trainable percentage: {100 * trainable_params / total_params:.2f}%")
                
                model_info[model_name] = {
                    'total_params': total_params,
                    'trainable_params': trainable_params,
                    'trainable_percentage': 100 * trainable_params / total_params
                }
                
                results.append(f"✓ {model_name}: {total_params:,} total, {trainable_params:,} trainable ({100 * trainable_params / total_params:.2f}%)")
                
                # Clean up memory for next model
                del model, tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"     ✗ {model_name} failed: {e}")
                results.append(f"✗ {model_name} failed: {e}")
        
        # Memory usage summary
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\n5. Final GPU Memory Usage:")
            print(f"   Allocated: {memory_allocated:.2f} GB")
            print(f"   Reserved: {memory_reserved:.2f} GB")
            results.append(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
        
        success = len(model_info) > 0
        results.append(f"✓ Successfully set up {len(model_info)} models")
        
    except Exception as e:
        print(f"\n✗ Error during model setup: {e}")
        results.append(f"✗ Error: {e}")
        success = False
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    status = "SUCCESS" if success else "ERROR"
    output_file = os.path.join(output_dir, f"step3_model_setup_{status}_{timestamp}.txt")
    
    with open(output_file, 'w') as f:
        f.write("=== Comprehensive Medical LLM Project - Model Setup Results ===\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"Status: {status}\n\n")
        for result in results:
            f.write(f"{result}\n")
    
    print(f"\n=== Model Setup Complete ===")
    print(f"Status: {status}")
    print(f"Results saved to: {output_file}")
    print(f"Completed at: {datetime.datetime.now()}")
    
    return output_file, success

if __name__ == "__main__":
    main() 