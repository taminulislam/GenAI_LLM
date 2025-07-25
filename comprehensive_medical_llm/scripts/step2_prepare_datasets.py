#!/usr/bin/env python3
"""
Comprehensive Medical LLM Project - Step 2: Dataset Preparation
Loads and prepares multiple medical datasets for training and evaluation.
"""

import sys
import os
import datetime

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    """Prepare datasets for training and evaluation."""
    print("=== Comprehensive Medical LLM Project - Dataset Preparation ===")
    print(f"Started at: {datetime.datetime.now()}")
    
    results = []
    
    try:
        from config import ComprehensiveConfig
        from data_loader import ComprehensiveDataLoader
        
        # Initialize configuration
        config = ComprehensiveConfig()
        print(f"\n1. Configuration loaded successfully")
        print(f"   Datasets to load: {', '.join(config.dataset_configs.keys())}")
        results.append(f"Configuration loaded: {len(config.dataset_configs)} datasets configured")
        
        # Initialize data loader
        data_loader = ComprehensiveDataLoader(config)
        print(f"\n2. Data loader initialized")
        results.append("Data loader initialized successfully")
        
        # Load datasets
        print(f"\n3. Loading datasets...")
        datasets = data_loader.load_all_datasets()
        
        print(f"\n4. Dataset Statistics:")
        total_samples = 0
        for dataset_name, dataset_info in datasets.items():
            train_size = len(dataset_info['train']) if dataset_info['train'] else 0
            val_size = len(dataset_info['validation']) if dataset_info['validation'] else 0
            test_size = len(dataset_info['test']) if dataset_info['test'] else 0
            
            print(f"   {dataset_name}:")
            print(f"     Train: {train_size:,} samples")
            print(f"     Validation: {val_size:,} samples") 
            print(f"     Test: {test_size:,} samples")
            
            total_samples += train_size + val_size + test_size
            results.append(f"{dataset_name}: Train={train_size}, Val={val_size}, Test={test_size}")
        
        print(f"\n   Total samples across all datasets: {total_samples:,}")
        results.append(f"Total samples: {total_samples:,}")
        
        # Create combined datasets
        print(f"\n5. Creating combined datasets...")
        combined_datasets = data_loader.create_combined_datasets(datasets)
        
        if combined_datasets['train']:
            train_size = len(combined_datasets['train'])
            print(f"   Combined train dataset: {train_size:,} samples")
            results.append(f"Combined train dataset: {train_size:,} samples")
        
        if combined_datasets['validation']:
            val_size = len(combined_datasets['validation'])
            print(f"   Combined validation dataset: {val_size:,} samples")
            results.append(f"Combined validation dataset: {val_size:,} samples")
        
        if combined_datasets['test']:
            test_size = len(combined_datasets['test'])
            print(f"   Combined test dataset: {test_size:,} samples")
            results.append(f"Combined test dataset: {test_size:,} samples")
        
        # Sample data inspection
        print(f"\n6. Sample Data Inspection:")
        if combined_datasets['train']:
            sample = combined_datasets['train'][0]
            print(f"   Sample input: {sample['input_text'][:100]}...")
            print(f"   Sample output: {sample['target_text'][:100]}...")
            results.append("Sample data inspection completed")
        
        success = True
        results.append("✓ All datasets loaded and prepared successfully")
        
    except Exception as e:
        print(f"\n✗ Error during dataset preparation: {e}")
        results.append(f"✗ Error: {e}")
        success = False
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    status = "SUCCESS" if success else "ERROR"
    output_file = os.path.join(output_dir, f"step2_dataset_preparation_{status}_{timestamp}.txt")
    
    with open(output_file, 'w') as f:
        f.write("=== Comprehensive Medical LLM Project - Dataset Preparation Results ===\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"Status: {status}\n\n")
        for result in results:
            f.write(f"{result}\n")
    
    print(f"\n=== Dataset Preparation Complete ===")
    print(f"Status: {status}")
    print(f"Results saved to: {output_file}")
    print(f"Completed at: {datetime.datetime.now()}")
    
    return output_file, success

if __name__ == "__main__":
    main() 