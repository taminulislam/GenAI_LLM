#!/usr/bin/env python3
"""
Comprehensive Medical LLM Project - Step 5: Model Evaluation
Comprehensively evaluates all trained models including hallucination detection.
"""

import sys
import os
import datetime
import torch

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    """Evaluate all trained models comprehensively."""
    print("=== Comprehensive Medical LLM Project - Model Evaluation ===")
    print(f"Started at: {datetime.datetime.now()}")
    
    results = []
    evaluation_results = {}
    
    try:
        from config import ComprehensiveConfig
        from data_loader import ComprehensiveDataLoader
        from evaluator import ComprehensiveEvaluator
        
        # Initialize configuration
        config = ComprehensiveConfig()
        print(f"\n1. Configuration loaded successfully")
        results.append(f"Configuration loaded: {len(config.model_configs)} models to evaluate")
        
        # Load test datasets
        print(f"\n2. Loading test datasets...")
        data_loader = ComprehensiveDataLoader(config)
        datasets = data_loader.load_all_datasets()
        combined_datasets = data_loader.create_combined_datasets(datasets)
        
        test_size = len(combined_datasets['test']) if combined_datasets['test'] else 0
        print(f"   Test samples: {test_size:,}")
        results.append(f"Test dataset loaded: {test_size:,} samples")
        
        # Initialize evaluator
        evaluator = ComprehensiveEvaluator(config)
        print(f"\n3. Evaluator initialized")
        results.append("Evaluator initialized successfully")
        
        # Evaluate each model
        print(f"\n4. Evaluating models...")
        for model_name in config.model_configs.keys():
            print(f"\n   Evaluating {model_name}...")
            try:
                # Check if model exists
                model_path = os.path.join(config.models_dir, model_name)
                if not os.path.exists(model_path):
                    print(f"     ✗ {model_name} model not found at {model_path}")
                    evaluation_results[model_name] = {
                        'status': 'model_not_found',
                        'error': f'Model not found at {model_path}'
                    }
                    results.append(f"✗ {model_name} - Model not found")
                    continue
                
                # Evaluate the model
                evaluation_start = datetime.datetime.now()
                eval_result = evaluator.evaluate_model(
                    model_name=model_name,
                    test_dataset=combined_datasets['test']
                )
                evaluation_end = datetime.datetime.now()
                evaluation_duration = evaluation_end - evaluation_start
                
                print(f"     ✓ {model_name} evaluation completed")
                print(f"     Evaluation time: {evaluation_duration}")
                print(f"     Accuracy: {eval_result.get('accuracy', 'N/A'):.4f}")
                print(f"     BLEU Score: {eval_result.get('bleu_score', 'N/A'):.4f}")
                print(f"     ROUGE-L: {eval_result.get('rouge_l', 'N/A'):.4f}")
                print(f"     Hallucination Score: {eval_result.get('hallucination_score', 'N/A'):.4f}")
                print(f"     Factual Consistency: {eval_result.get('factual_consistency', 'N/A'):.4f}")
                
                evaluation_results[model_name] = {
                    'duration': str(evaluation_duration),
                    'accuracy': eval_result.get('accuracy', 'N/A'),
                    'bleu_score': eval_result.get('bleu_score', 'N/A'),
                    'rouge_l': eval_result.get('rouge_l', 'N/A'),
                    'hallucination_score': eval_result.get('hallucination_score', 'N/A'),
                    'factual_consistency': eval_result.get('factual_consistency', 'N/A'),
                    'status': 'success'
                }
                
                results.append(f"✓ {model_name}: Acc={eval_result.get('accuracy', 'N/A'):.4f}, BLEU={eval_result.get('bleu_score', 'N/A'):.4f}")
                
            except Exception as e:
                print(f"     ✗ {model_name} evaluation failed: {e}")
                evaluation_results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
                results.append(f"✗ {model_name} failed: {e}")
        
        # Evaluation summary
        successful_evals = [name for name, result in evaluation_results.items() if result['status'] == 'success']
        failed_evals = [name for name, result in evaluation_results.items() if result['status'] != 'success']
        
        print(f"\n5. Evaluation Summary:")
        print(f"   Successful evaluations: {len(successful_evals)}")
        print(f"   Failed evaluations: {len(failed_evals)}")
        
        if successful_evals:
            print(f"   Success: {', '.join(successful_evals)}")
            
            # Find best performing model
            best_model = None
            best_accuracy = 0
            for model_name in successful_evals:
                accuracy = evaluation_results[model_name].get('accuracy', 0)
                if isinstance(accuracy, (int, float)) and accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_name
            
            if best_model:
                print(f"   Best performing model: {best_model} (Accuracy: {best_accuracy:.4f})")
                results.append(f"Best model: {best_model} (Accuracy: {best_accuracy:.4f})")
        
        if failed_evals:
            print(f"   Failed: {', '.join(failed_evals)}")
        
        results.append(f"Evaluation Summary - Success: {len(successful_evals)}, Failed: {len(failed_evals)}")
        
        success = len(successful_evals) > 0
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        results.append(f"✗ Error: {e}")
        success = False
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    status = "SUCCESS" if success else "ERROR"
    output_file = os.path.join(output_dir, f"step5_evaluation_{status}_{timestamp}.txt")
    
    with open(output_file, 'w') as f:
        f.write("=== Comprehensive Medical LLM Project - Evaluation Results ===\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"Status: {status}\n\n")
        for result in results:
            f.write(f"{result}\n")
        
        f.write(f"\n=== Detailed Evaluation Results ===\n")
        for model_name, result in evaluation_results.items():
            f.write(f"\n{model_name}:\n")
            for key, value in result.items():
                f.write(f"  {key}: {value}\n")
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Status: {status}")
    print(f"Results saved to: {output_file}")
    print(f"Completed at: {datetime.datetime.now()}")
    
    return output_file, success

if __name__ == "__main__":
    main() 