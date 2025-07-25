#!/usr/bin/env python3
"""
Comprehensive Medical LLM Project - Step 4: Model Training
Trains multiple models on medical datasets with comprehensive logging.
"""

import sys
import os
import datetime
import torch

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    """Train multiple models on medical datasets."""
    print("=== Comprehensive Medical LLM Project - Model Training ===")
    print(f"Started at: {datetime.datetime.now()}")
    
    results = []
    training_results = {}
    
    try:
        from config import ComprehensiveConfig
        from data_loader import ComprehensiveDataLoader
        from trainer import ComprehensiveTrainer
        
        # Initialize configuration
        config = ComprehensiveConfig()
        print(f"\n1. Configuration loaded successfully")
        results.append(f"Configuration loaded: {len(config.model_configs)} models to train")
        
        # Load datasets
        print(f"\n2. Loading datasets...")
        data_loader = ComprehensiveDataLoader(config)
        datasets = data_loader.load_all_datasets()
        combined_datasets = data_loader.create_combined_datasets(datasets)
        
        train_size = len(combined_datasets['train']) if combined_datasets['train'] else 0
        val_size = len(combined_datasets['validation']) if combined_datasets['validation'] else 0
        
        print(f"   Training samples: {train_size:,}")
        print(f"   Validation samples: {val_size:,}")
        results.append(f"Datasets loaded - Train: {train_size:,}, Val: {val_size:,}")
        
        # Initialize trainer
        trainer = ComprehensiveTrainer(config)
        print(f"\n3. Trainer initialized")
        results.append("Trainer initialized successfully")
        
        # Train each model
        print(f"\n4. Training models...")
        for model_name in config.model_configs.keys():
            print(f"\n   Training {model_name}...")
            try:
                # Train the model
                training_start = datetime.datetime.now()
                training_result = trainer.train_model(
                    model_name=model_name,
                    train_dataset=combined_datasets['train'],
                    validation_dataset=combined_datasets['validation']
                )
                training_end = datetime.datetime.now()
                training_duration = training_end - training_start
                
                print(f"     ✓ {model_name} training completed")
                print(f"     Training time: {training_duration}")
                print(f"     Final train loss: {training_result.get('train_loss', 'N/A')}")
                print(f"     Final eval loss: {training_result.get('eval_loss', 'N/A')}")
                
                training_results[model_name] = {
                    'duration': str(training_duration),
                    'train_loss': training_result.get('train_loss', 'N/A'),
                    'eval_loss': training_result.get('eval_loss', 'N/A'),
                    'status': 'success'
                }
                
                results.append(f"✓ {model_name}: {training_duration}, Train Loss: {training_result.get('train_loss', 'N/A')}")
                
            except Exception as e:
                print(f"     ✗ {model_name} training failed: {e}")
                training_results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
                results.append(f"✗ {model_name} failed: {e}")
        
        # Training summary
        successful_models = [name for name, result in training_results.items() if result['status'] == 'success']
        failed_models = [name for name, result in training_results.items() if result['status'] == 'failed']
        
        print(f"\n5. Training Summary:")
        print(f"   Successful models: {len(successful_models)}")
        print(f"   Failed models: {len(failed_models)}")
        
        if successful_models:
            print(f"   Success: {', '.join(successful_models)}")
        if failed_models:
            print(f"   Failed: {', '.join(failed_models)}")
        
        results.append(f"Training Summary - Success: {len(successful_models)}, Failed: {len(failed_models)}")
        
        success = len(successful_models) > 0
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        results.append(f"✗ Error: {e}")
        success = False
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    status = "SUCCESS" if success else "ERROR"
    output_file = os.path.join(output_dir, f"step4_training_{status}_{timestamp}.txt")
    
    with open(output_file, 'w') as f:
        f.write("=== Comprehensive Medical LLM Project - Training Results ===\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"Status: {status}\n\n")
        for result in results:
            f.write(f"{result}\n")
        
        f.write(f"\n=== Detailed Training Results ===\n")
        for model_name, result in training_results.items():
            f.write(f"\n{model_name}:\n")
            for key, value in result.items():
                f.write(f"  {key}: {value}\n")
    
    print(f"\n=== Training Complete ===")
    print(f"Status: {status}")
    print(f"Results saved to: {output_file}")
    print(f"Completed at: {datetime.datetime.now()}")
    
    return output_file, success

if __name__ == "__main__":
    main() 