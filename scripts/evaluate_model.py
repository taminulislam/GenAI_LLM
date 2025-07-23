#!/usr/bin/env python3
"""
Medical LLM Evaluation Script
Comprehensive evaluation and comparison of trained medical LLM models
"""

import os
import sys
import logging
import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import config
from src.model_setup import ModelManager
from src.evaluator import MedicalLLMEvaluator, quick_evaluate, compare_models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalLLMEvaluationSuite:
    """Comprehensive evaluation suite for medical LLMs"""
    
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg else config
        self.evaluator = MedicalLLMEvaluator(self.cfg)
        self.results_dir = Path("evaluation")
        self.results_dir.mkdir(exist_ok=True)
        
    def find_trained_models(self, experiments_dir: str = "experiments") -> List[Dict[str, str]]:
        """Find all trained models in experiments directory"""
        logger.info(f"🔍 Searching for trained models in {experiments_dir}...")
        
        experiments_path = Path(experiments_dir)
        trained_models = []
        
        if not experiments_path.exists():
            logger.warning(f"Experiments directory not found: {experiments_dir}")
            return trained_models
        
        for experiment_dir in experiments_path.iterdir():
            if experiment_dir.is_dir():
                # Look for final_model or models subdirectory
                model_paths = [
                    experiment_dir / "final_model",
                    experiment_dir / "models" / "final_model",
                    experiment_dir / "models"
                ]
                
                for model_path in model_paths:
                    if model_path.exists():
                        # Check if it's a valid model directory
                        if (model_path / "adapter_config.json").exists() or \
                           (model_path / "config.json").exists():
                            
                            trained_models.append({
                                "experiment_name": experiment_dir.name,
                                "model_path": str(model_path),
                                "experiment_dir": str(experiment_dir)
                            })
                            logger.info(f"✅ Found model: {experiment_dir.name}")
                            break
        
        logger.info(f"Found {len(trained_models)} trained models")
        return trained_models
    
    def load_experiment_config(self, experiment_dir: str) -> Dict[str, Any]:
        """Load experiment configuration"""
        config_file = Path(experiment_dir) / "experiment_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def evaluate_single_model(self, 
                             model_path: str, 
                             model_name: str = None,
                             save_results: bool = True) -> Dict[str, Any]:
        """Evaluate a single model comprehensively"""
        if model_name is None:
            model_name = Path(model_path).parent.name
        
        logger.info(f"📊 Evaluating model: {model_name}")
        logger.info(f"Model path: {model_path}")
        
        try:
            # Run comprehensive evaluation
            results = self.evaluator.run_comprehensive_evaluation(model_path)
            
            # Add model metadata
            results["model_metadata"] = {
                "model_name": model_name,
                "model_path": model_path,
                "evaluation_date": datetime.now().isoformat()
            }
            
            if save_results:
                # Save detailed results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = self.results_dir / f"evaluation_{model_name}_{timestamp}.json"
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                # Save human-readable report
                report = self.evaluator.create_evaluation_report(results)
                report_file = self.results_dir / f"report_{model_name}_{timestamp}.txt"
                with open(report_file, 'w') as f:
                    f.write(report)
                
                logger.info(f"💾 Results saved to: {results_file}")
                logger.info(f"📄 Report saved to: {report_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed for {model_name}: {e}")
            return {
                "model_metadata": {
                    "model_name": model_name,
                    "model_path": model_path,
                    "evaluation_date": datetime.now().isoformat()
                },
                "error": str(e),
                "success": False
            }
    
    def compare_multiple_models(self, 
                               model_info_list: List[Dict[str, str]],
                               save_comparison: bool = True) -> pd.DataFrame:
        """Compare multiple models and generate comparison report"""
        logger.info(f"🔄 Comparing {len(model_info_list)} models...")
        
        comparison_results = []
        detailed_results = []
        
        for model_info in model_info_list:
            model_name = model_info.get("experiment_name", "Unknown")
            model_path = model_info["model_path"]
            experiment_dir = model_info.get("experiment_dir", "")
            
            logger.info(f"Evaluating {model_name}...")
            
            # Get experiment config
            exp_config = self.load_experiment_config(experiment_dir) if experiment_dir else {}
            
            # Run evaluation
            results = self.evaluate_single_model(model_path, model_name, save_results=False)
            detailed_results.append(results)
            
            if "error" not in results:
                summary = results.get("summary", {})
                model_config = exp_config.get("model_config", {})
                training_config = exp_config.get("training_config", {})
                
                comparison_row = {
                    "Model_Name": model_name,
                    "Overall_Accuracy": summary.get("overall_accuracy", 0),
                    "Total_Questions": summary.get("total_questions", 0),
                    "Correct_Answers": summary.get("total_correct", 0),
                    "Benchmarks_Evaluated": summary.get("benchmarks_evaluated", 0),
                    "Base_Model": model_config.get("base_model", "Unknown"),
                    "LoRA_R": model_config.get("lora_r", "N/A"),
                    "Training_Epochs": training_config.get("num_epochs", "N/A"),
                    "Learning_Rate": training_config.get("learning_rate", "N/A"),
                    "Batch_Size": training_config.get("batch_size", "N/A"),
                    "Model_Path": model_path
                }
                
                # Add benchmark-specific scores
                benchmark_results = results.get("benchmark_results", {})
                for benchmark_name, benchmark_data in benchmark_results.items():
                    if "error" not in benchmark_data:
                        accuracy = benchmark_data.get("accuracy", 0)
                        comparison_row[f"{benchmark_name}_accuracy"] = accuracy
                
                comparison_results.append(comparison_row)
            else:
                # Handle failed evaluations
                comparison_results.append({
                    "Model_Name": model_name,
                    "Overall_Accuracy": 0,
                    "Total_Questions": 0,
                    "Correct_Answers": 0,
                    "Error": results.get("error", "Unknown error"),
                    "Model_Path": model_path
                })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        
        if save_comparison:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save comparison table
            csv_file = self.results_dir / f"model_comparison_{timestamp}.csv"
            comparison_df.to_csv(csv_file, index=False)
            
            # Save detailed results
            detailed_file = self.results_dir / f"detailed_comparison_{timestamp}.json"
            with open(detailed_file, 'w') as f:
                json.dump(detailed_results, f, indent=2, default=str)
            
            # Generate comparison report
            report = self.generate_comparison_report(comparison_df, detailed_results)
            report_file = self.results_dir / f"comparison_report_{timestamp}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            logger.info(f"💾 Comparison saved to: {csv_file}")
            logger.info(f"📄 Report saved to: {report_file}")
        
        return comparison_df
    
    def generate_comparison_report(self, 
                                  comparison_df: pd.DataFrame, 
                                  detailed_results: List[Dict]) -> str:
        """Generate a comprehensive comparison report"""
        report = []
        report.append("=" * 80)
        report.append("MEDICAL LLM MODEL COMPARISON REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Models Evaluated: {len(comparison_df)}")
        report.append("")
        
        # Summary statistics
        if len(comparison_df) > 0 and "Overall_Accuracy" in comparison_df.columns:
            successful_models = comparison_df[comparison_df["Overall_Accuracy"] > 0]
            
            if len(successful_models) > 0:
                report.append("SUMMARY STATISTICS:")
                report.append(f"  Best Overall Accuracy: {successful_models['Overall_Accuracy'].max():.3f}")
                report.append(f"  Average Accuracy: {successful_models['Overall_Accuracy'].mean():.3f}")
                report.append(f"  Worst Accuracy: {successful_models['Overall_Accuracy'].min():.3f}")
                report.append(f"  Standard Deviation: {successful_models['Overall_Accuracy'].std():.3f}")
                
                # Best performing model
                best_model_idx = successful_models['Overall_Accuracy'].idxmax()
                best_model = successful_models.loc[best_model_idx]
                report.append(f"  Best Model: {best_model['Model_Name']} ({best_model['Overall_Accuracy']:.3f})")
                report.append("")
        
        # Individual model performance
        report.append("INDIVIDUAL MODEL PERFORMANCE:")
        report.append("-" * 80)
        
        # Sort by accuracy (descending)
        if "Overall_Accuracy" in comparison_df.columns:
            sorted_df = comparison_df.sort_values("Overall_Accuracy", ascending=False)
        else:
            sorted_df = comparison_df
        
        for _, row in sorted_df.iterrows():
            report.append(f"Model: {row['Model_Name']}")
            if "Error" in row and pd.notna(row.get("Error")):
                report.append(f"  Status: FAILED - {row['Error']}")
            else:
                report.append(f"  Overall Accuracy: {row.get('Overall_Accuracy', 'N/A'):.3f}")
                report.append(f"  Questions Answered: {row.get('Total_Questions', 'N/A')}")
                report.append(f"  Correct Answers: {row.get('Correct_Answers', 'N/A')}")
                
                if "Base_Model" in row:
                    report.append(f"  Base Model: {row.get('Base_Model', 'N/A')}")
                if "Training_Epochs" in row:
                    report.append(f"  Training Epochs: {row.get('Training_Epochs', 'N/A')}")
                if "Learning_Rate" in row:
                    report.append(f"  Learning Rate: {row.get('Learning_Rate', 'N/A')}")
                
                # Benchmark-specific performance
                benchmark_cols = [col for col in row.index if col.endswith('_accuracy')]
                if benchmark_cols:
                    report.append("  Benchmark Performance:")
                    for col in benchmark_cols:
                        benchmark_name = col.replace('_accuracy', '')
                        accuracy = row[col]
                        if pd.notna(accuracy):
                            report.append(f"    {benchmark_name}: {accuracy:.3f}")
            
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 80)
        
        if len(successful_models) > 0:
            best_model = successful_models.loc[successful_models['Overall_Accuracy'].idxmax()]
            report.append(f"• Best performing model: {best_model['Model_Name']}")
            report.append(f"• Consider using this model for production deployment")
            
            if len(successful_models) > 1:
                worst_model = successful_models.loc[successful_models['Overall_Accuracy'].idxmin()]
                improvement = best_model['Overall_Accuracy'] - worst_model['Overall_Accuracy']
                report.append(f"• Performance gap: {improvement:.3f} accuracy points")
                
                if improvement > 0.1:
                    report.append("• Significant performance differences observed")
                    report.append("• Review training configurations and hyperparameters")
        else:
            report.append("• No successful evaluations - check model paths and configurations")
            report.append("• Verify that models were trained properly")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def benchmark_against_baselines(self, model_path: str, model_name: str = None) -> Dict[str, Any]:
        """Benchmark a model against baseline models"""
        logger.info(f"🎯 Benchmarking model against baselines...")
        
        # Evaluate the target model
        target_results = self.evaluate_single_model(model_path, model_name, save_results=False)
        
        # Create baseline comparison
        baseline_info = {
            "target_model": {
                "name": model_name or "Target Model",
                "path": model_path,
                "results": target_results
            },
            "baselines": {
                "random_baseline": {"accuracy": 0.25, "description": "Random chance (25% for 4-choice questions)"},
                "dummy_baseline": {"accuracy": 0.20, "description": "Simple pattern matching"}
            }
        }
        
        # Generate benchmark report
        if "error" not in target_results:
            target_accuracy = target_results.get("summary", {}).get("overall_accuracy", 0)
            
            improvement_over_random = target_accuracy - 0.25
            improvement_over_dummy = target_accuracy - 0.20
            
            baseline_info["comparison"] = {
                "target_accuracy": target_accuracy,
                "improvement_over_random": improvement_over_random,
                "improvement_over_dummy": improvement_over_dummy,
                "performance_level": self._classify_performance(target_accuracy)
            }
        
        return baseline_info
    
    def _classify_performance(self, accuracy: float) -> str:
        """Classify model performance level"""
        if accuracy >= 0.8:
            return "Excellent"
        elif accuracy >= 0.6:
            return "Good"
        elif accuracy >= 0.4:
            return "Fair"
        elif accuracy >= 0.25:
            return "Poor"
        else:
            return "Very Poor"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate medical LLM models")
    
    # Evaluation modes
    parser.add_argument("--model-path", type=str, help="Path to specific model to evaluate")
    parser.add_argument("--experiment-name", type=str, help="Name of specific experiment to evaluate")
    parser.add_argument("--compare-all", action="store_true", help="Compare all models in experiments directory")
    parser.add_argument("--experiments-dir", type=str, default="experiments", help="Directory containing experiments")
    
    # Evaluation options
    parser.add_argument("--quick-eval", action="store_true", help="Run quick evaluation with dummy data")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark against baselines")
    parser.add_argument("--save-results", action="store_true", default=True, help="Save evaluation results")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="evaluation", help="Directory to save results")
    parser.add_argument("--report-only", action="store_true", help="Generate report from existing results")
    
    return parser.parse_args()

def main():
    """Main evaluation function"""
    args = parse_arguments()
    
    logger.info("📊 Medical LLM Evaluation Suite")
    logger.info("=" * 60)
    
    # Initialize evaluation suite
    evaluation_suite = MedicalLLMEvaluationSuite()
    evaluation_suite.results_dir = Path(args.output_dir)
    evaluation_suite.results_dir.mkdir(exist_ok=True)
    
    try:
        if args.model_path:
            # Evaluate specific model
            logger.info(f"Evaluating specific model: {args.model_path}")
            
            if not Path(args.model_path).exists():
                logger.error(f"Model path not found: {args.model_path}")
                sys.exit(1)
            
            model_name = args.experiment_name or Path(args.model_path).parent.name
            results = evaluation_suite.evaluate_single_model(
                args.model_path, 
                model_name, 
                save_results=args.save_results
            )
            
            # Print summary
            if "error" not in results:
                summary = results.get("summary", {})
                logger.info(f"✅ Evaluation completed!")
                logger.info(f"   Overall Accuracy: {summary.get('overall_accuracy', 0):.3f}")
                logger.info(f"   Questions: {summary.get('total_questions', 0)}")
                logger.info(f"   Correct: {summary.get('total_correct', 0)}")
            
            # Benchmark if requested
            if args.benchmark:
                baseline_results = evaluation_suite.benchmark_against_baselines(
                    args.model_path, model_name
                )
                logger.info(f"🎯 Benchmark Results:")
                comparison = baseline_results.get("comparison", {})
                logger.info(f"   Performance Level: {comparison.get('performance_level', 'Unknown')}")
                logger.info(f"   Improvement over Random: +{comparison.get('improvement_over_random', 0):.3f}")
        
        elif args.experiment_name:
            # Evaluate specific experiment
            experiment_path = Path(args.experiments_dir) / args.experiment_name
            if not experiment_path.exists():
                logger.error(f"Experiment not found: {experiment_path}")
                sys.exit(1)
            
            # Find model in experiment
            model_paths = [
                experiment_path / "final_model",
                experiment_path / "models" / "final_model",
            ]
            
            model_path = None
            for path in model_paths:
                if path.exists():
                    model_path = str(path)
                    break
            
            if not model_path:
                logger.error(f"No model found in experiment: {experiment_path}")
                sys.exit(1)
            
            results = evaluation_suite.evaluate_single_model(
                model_path, 
                args.experiment_name, 
                save_results=args.save_results
            )
        
        elif args.compare_all:
            # Compare all models
            trained_models = evaluation_suite.find_trained_models(args.experiments_dir)
            
            if not trained_models:
                logger.warning("No trained models found for comparison")
                logger.info("Make sure you have run training experiments first")
                sys.exit(1)
            
            logger.info(f"Found {len(trained_models)} models to compare")
            comparison_df = evaluation_suite.compare_multiple_models(
                trained_models, 
                save_comparison=args.save_results
            )
            
            # Print comparison summary
            logger.info("\n📈 Comparison Summary:")
            logger.info("=" * 60)
            
            if len(comparison_df) > 0 and "Overall_Accuracy" in comparison_df.columns:
                successful_models = comparison_df[comparison_df["Overall_Accuracy"] > 0]
                if len(successful_models) > 0:
                    logger.info(f"Best Model: {successful_models.loc[successful_models['Overall_Accuracy'].idxmax(), 'Model_Name']}")
                    logger.info(f"Best Accuracy: {successful_models['Overall_Accuracy'].max():.3f}")
                    logger.info(f"Average Accuracy: {successful_models['Overall_Accuracy'].mean():.3f}")
                    logger.info(f"Models Evaluated: {len(successful_models)}")
                else:
                    logger.warning("No successful evaluations found")
            
        else:
            # Default: quick evaluation with dummy data
            logger.info("Running quick evaluation with dummy data...")
            results = quick_evaluate(use_dummy_data=True)
            
            summary = results.get("summary", {})
            logger.info(f"✅ Quick evaluation completed!")
            logger.info(f"   Overall Accuracy: {summary.get('overall_accuracy', 0):.3f}")
            logger.info(f"   Questions: {summary.get('total_questions', 0)}")
        
        logger.info("\n🎯 Evaluation completed successfully!")
        logger.info(f"📁 Results saved to: {evaluation_suite.results_dir}")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 