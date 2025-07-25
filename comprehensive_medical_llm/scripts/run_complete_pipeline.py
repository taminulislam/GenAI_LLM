#!/usr/bin/env python3
"""
Comprehensive Medical LLM Project - Complete Pipeline Runner
Runs the entire comprehensive pipeline from setup to evaluation.
"""

import sys
import os
import datetime
import subprocess

def run_script(script_name, description):
    """Run a script and return success status."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print(f"Started at: {datetime.datetime.now()}")
    print(f"{'='*60}")
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True,
                              cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print(f"\n✓ {description} completed successfully")
            return True
        else:
            print(f"\n✗ {description} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n✗ {description} failed with error: {e}")
        return False

def main():
    """Run the complete comprehensive pipeline."""
    print("=== COMPREHENSIVE MEDICAL LLM PROJECT - COMPLETE PIPELINE ===")
    print(f"Pipeline started at: {datetime.datetime.now()}")
    
    # Get script directory
    script_dir = os.path.dirname(__file__)
    
    # Define pipeline steps
    pipeline_steps = [
        ("step1_setup_environment.py", "Environment Setup & Dependency Check"),
        ("step2_prepare_datasets.py", "Dataset Loading & Preparation"),
        ("step3_setup_models.py", "Model Setup & Configuration"),
        ("step4_train_models.py", "Model Training"),
        ("step5_evaluate_models.py", "Model Evaluation & Analysis")
    ]
    
    # Track results
    results = {}
    overall_success = True
    
    print(f"\nPipeline consists of {len(pipeline_steps)} steps:")
    for i, (script, description) in enumerate(pipeline_steps, 1):
        print(f"  {i}. {description}")
    
    # Run each step
    for i, (script_name, description) in enumerate(pipeline_steps, 1):
        script_path = os.path.join(script_dir, script_name)
        
        if not os.path.exists(script_path):
            print(f"\n✗ Script not found: {script_path}")
            results[f"Step {i}"] = {"status": "failed", "error": "Script not found"}
            overall_success = False
            continue
        
        print(f"\n\nSTEP {i}/{len(pipeline_steps)}: {description}")
        success = run_script(script_path, description)
        
        results[f"Step {i}"] = {
            "script": script_name,
            "description": description,
            "status": "success" if success else "failed"
        }
        
        if not success:
            overall_success = False
            print(f"\n⚠️  Step {i} failed. You may want to investigate before continuing.")
            
            # Ask user if they want to continue
            while True:
                try:
                    user_input = input(f"\nContinue with remaining steps? (y/n): ").lower().strip()
                    if user_input in ['y', 'yes']:
                        print("Continuing with next step...")
                        break
                    elif user_input in ['n', 'no']:
                        print("Pipeline stopped by user.")
                        overall_success = False
                        break
                    else:
                        print("Please enter 'y' or 'n'")
                except KeyboardInterrupt:
                    print("\nPipeline interrupted by user.")
                    return False
            
            if user_input in ['n', 'no']:
                break
    
    # Final summary
    print(f"\n\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Pipeline completed at: {datetime.datetime.now()}")
    print(f"Overall Status: {'SUCCESS' if overall_success else 'PARTIAL/FAILED'}")
    
    print(f"\nStep Results:")
    for step, result in results.items():
        status_icon = "✓" if result["status"] == "success" else "✗"
        print(f"  {status_icon} {step}: {result['description']} - {result['status']}")
    
    # Save summary
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    status = "SUCCESS" if overall_success else "PARTIAL"
    summary_file = os.path.join(output_dir, f"complete_pipeline_summary_{status}_{timestamp}.txt")
    
    with open(summary_file, 'w') as f:
        f.write("=== Comprehensive Medical LLM Project - Complete Pipeline Summary ===\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"Overall Status: {status}\n\n")
        
        f.write("Step Results:\n")
        for step, result in results.items():
            f.write(f"{step}: {result['description']} - {result['status']}\n")
            if 'error' in result:
                f.write(f"  Error: {result['error']}\n")
    
    print(f"\nPipeline summary saved to: {summary_file}")
    
    if overall_success:
        print(f"\n🎉 COMPREHENSIVE PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"All models have been trained and evaluated.")
        print(f"Check the outputs directory for detailed results.")
    else:
        print(f"\n⚠️  Pipeline completed with some issues.")
        print(f"Please check the individual step outputs for details.")
    
    return overall_success

if __name__ == "__main__":
    main() 