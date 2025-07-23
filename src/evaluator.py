"""
Medical LLM Evaluator Module
Handles model evaluation, benchmarking, and performance metrics
"""

import os
import json
import logging
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datasets import Dataset, load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re
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

class MedicalLLMEvaluator:
    """Comprehensive evaluator for medical LLMs"""
    
    def __init__(self, cfg=None):
        """
        Initialize evaluator with configuration
        
        Args:
            cfg: Configuration object, defaults to global config
        """
        self.cfg = cfg if cfg else config
        self.model_manager = None
        self.evaluation_results = {}
        self.benchmark_datasets = {}
        
    def load_benchmark_datasets(self) -> Dict[str, Dataset]:
        """
        Load standard medical benchmark datasets for evaluation
        
        Returns:
            Dictionary of benchmark datasets
        """
        logger.info("Loading medical benchmark datasets...")
        
        benchmarks = {}
        
        try:
            # MedQA dataset
            logger.info("Loading MedQA dataset...")
            medqa = load_dataset("openlifescienceai/medmcqa", split="validation[:500]")
            benchmarks['medqa'] = medqa
            logger.info(f"✅ MedQA loaded: {len(medqa)} samples")
        except Exception as e:
            logger.warning(f"⚠️ Could not load MedQA: {e}")
        
        try:
            # PubMedQA dataset
            logger.info("Loading PubMedQA dataset...")
            pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="test[:300]")
            benchmarks['pubmedqa'] = pubmedqa
            logger.info(f"✅ PubMedQA loaded: {len(pubmedqa)} samples")
        except Exception as e:
            logger.warning(f"⚠️ Could not load PubMedQA: {e}")
        
        # Create dummy medical benchmarks if real ones fail
        if not benchmarks:
            logger.info("Creating dummy medical benchmark for testing...")
            dummy_data = self._create_dummy_medical_benchmark()
            benchmarks['dummy_medical'] = dummy_data
        
        self.benchmark_datasets = benchmarks
        logger.info(f"Benchmark datasets loaded: {list(benchmarks.keys())}")
        return benchmarks
    
    def _create_dummy_medical_benchmark(self) -> Dataset:
        """Create dummy medical benchmark for testing"""
        dummy_questions = [
            {
                "question": "What is the primary function of the heart?",
                "choices": ["A) Digestion", "B) Circulation", "C) Respiration", "D) Excretion"],
                "answer": "B",
                "context": "The heart is a muscular organ that pumps blood throughout the body."
            },
            {
                "question": "Which vitamin deficiency causes scurvy?",
                "choices": ["A) Vitamin A", "B) Vitamin B", "C) Vitamin C", "D) Vitamin D"],
                "answer": "C",
                "context": "Scurvy is caused by vitamin C deficiency, leading to collagen problems."
            },
            {
                "question": "What is the normal range for human body temperature?",
                "choices": ["A) 35-36°C", "B) 36-37°C", "C) 37-38°C", "D) 38-39°C"],
                "answer": "B",
                "context": "Normal body temperature ranges from 36.1°C to 37.2°C (97°F to 99°F)."
            },
            {
                "question": "Which organ produces insulin?",
                "choices": ["A) Liver", "B) Kidney", "C) Pancreas", "D) Spleen"],
                "answer": "C",
                "context": "Insulin is produced by beta cells in the pancreas to regulate blood sugar."
            },
            {
                "question": "What is hypertension?",
                "choices": ["A) Low blood pressure", "B) High blood pressure", "C) Fast heart rate", "D) Slow heart rate"],
                "answer": "B",
                "context": "Hypertension refers to persistently high blood pressure readings."
            }
        ]
        
        from datasets import Dataset
        return Dataset.from_list(dummy_questions)
    
    def setup_model_for_evaluation(self, model_path: str = None, model_manager: ModelManager = None):
        """
        Setup model for evaluation
        
        Args:
            model_path: Path to trained model
            model_manager: Pre-configured ModelManager instance
        """
        if model_manager:
            self.model_manager = model_manager
            logger.info("Using provided ModelManager")
        elif model_path and os.path.exists(model_path):
            logger.info(f"Loading model from: {model_path}")
            self.model_manager = ModelManager(self.cfg)
            self.model_manager.load_trained_model(model_path)
        else:
            logger.info("Setting up base model for evaluation")
            self.model_manager = ModelManager(self.cfg)
            self.model_manager.setup_model_and_tokenizer()
        
        logger.info("Model ready for evaluation")
    
    def generate_response(self, prompt: str, max_length: int = 256) -> str:
        """
        Generate response from the model
        
        Args:
            prompt: Input prompt
            max_length: Maximum response length
            
        Returns:
            Generated response
        """
        if not self.model_manager or not self.model_manager.model:
            raise ValueError("Model not loaded. Call setup_model_for_evaluation() first.")
        
        # Create generation pipeline
        generator = pipeline(
            "text-generation",
            model=self.model_manager.model,
            tokenizer=self.model_manager.tokenizer,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        try:
            # Generate response
            result = generator(
                prompt,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.model_manager.tokenizer.eos_token_id
            )
            
            # Extract generated text (remove the input prompt)
            generated_text = result[0]['generated_text']
            response = generated_text[len(prompt):].strip()
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
    
    def evaluate_multiple_choice(self, dataset: Dataset, dataset_name: str) -> Dict[str, Any]:
        """
        Evaluate model on multiple choice questions
        
        Args:
            dataset: Dataset with multiple choice questions
            dataset_name: Name of the dataset for logging
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating on {dataset_name} ({len(dataset)} samples)...")
        
        correct_answers = 0
        total_questions = 0
        detailed_results = []
        
        for i, sample in enumerate(dataset):
            if i >= 50:  # Limit for quick evaluation
                break
                
            # Format question
            question = sample.get('question', '')
            choices = sample.get('choices', [])
            correct_answer = sample.get('answer', 'A')
            
            # Create prompt
            if isinstance(choices, list):
                choices_text = '\n'.join(choices)
            else:
                choices_text = str(choices)
            
            prompt = f"""Answer the following medical question by selecting the correct choice.

Question: {question}

{choices_text}

Answer:"""
            
            # Generate response
            response = self.generate_response(prompt, max_length=len(prompt) + 50)
            
            # Extract predicted answer
            predicted_answer = self._extract_choice_from_response(response)
            
            # Check if correct
            is_correct = predicted_answer.upper() == correct_answer.upper()
            if is_correct:
                correct_answers += 1
            total_questions += 1
            
            # Store detailed result
            detailed_results.append({
                'question_id': i,
                'question': question,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'response': response[:200] + "..." if len(response) > 200 else response
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{min(len(dataset), 50)} questions")
        
        # Calculate metrics
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        results = {
            'dataset_name': dataset_name,
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'detailed_results': detailed_results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ {dataset_name} Evaluation Complete:")
        logger.info(f"   Accuracy: {accuracy:.3f} ({correct_answers}/{total_questions})")
        
        return results
    
    def _extract_choice_from_response(self, response: str) -> str:
        """
        Extract choice (A, B, C, D) from model response
        
        Args:
            response: Model response text
            
        Returns:
            Extracted choice or 'Unknown'
        """
        # Look for patterns like "A)", "A.", "A:", "(A)", or just "A"
        patterns = [
            r'\b([ABCD])\)',
            r'\b([ABCD])\.',
            r'\b([ABCD]):',
            r'\(([ABCD])\)',
            r'\b([ABCD])\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.upper())
            if match:
                return match.group(1)
        
        # Fallback: look for the first occurrence of A, B, C, or D
        for char in ['A', 'B', 'C', 'D']:
            if char in response.upper():
                return char
        
        return 'Unknown'
    
    def evaluate_text_generation(self, prompts: List[str], references: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate text generation quality
        
        Args:
            prompts: List of input prompts
            references: Optional reference responses
            
        Returns:
            Dictionary with generation quality metrics
        """
        logger.info(f"Evaluating text generation on {len(prompts)} prompts...")
        
        generated_responses = []
        generation_stats = {
            'avg_length': 0,
            'min_length': float('inf'),
            'max_length': 0,
            'total_tokens': 0
        }
        
        for i, prompt in enumerate(prompts):
            response = self.generate_response(prompt, max_length=512)
            generated_responses.append(response)
            
            # Calculate stats
            response_length = len(response.split())
            generation_stats['total_tokens'] += response_length
            generation_stats['min_length'] = min(generation_stats['min_length'], response_length)
            generation_stats['max_length'] = max(generation_stats['max_length'], response_length)
            
            if (i + 1) % 5 == 0:
                logger.info(f"Generated {i + 1}/{len(prompts)} responses")
        
        generation_stats['avg_length'] = generation_stats['total_tokens'] / len(prompts)
        generation_stats['min_length'] = generation_stats['min_length'] if generation_stats['min_length'] != float('inf') else 0
        
        results = {
            'num_prompts': len(prompts),
            'generated_responses': generated_responses,
            'generation_stats': generation_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add quality metrics if references provided
        if references and len(references) == len(prompts):
            # Simple overlap-based metrics (could be enhanced with BLEU, ROUGE, etc.)
            overlap_scores = []
            for gen, ref in zip(generated_responses, references):
                gen_words = set(gen.lower().split())
                ref_words = set(ref.lower().split())
                if len(ref_words) > 0:
                    overlap = len(gen_words.intersection(ref_words)) / len(ref_words)
                    overlap_scores.append(overlap)
            
            results['avg_word_overlap'] = np.mean(overlap_scores) if overlap_scores else 0
        
        logger.info(f"✅ Text Generation Evaluation Complete:")
        logger.info(f"   Average response length: {generation_stats['avg_length']:.1f} words")
        
        return results
    
    def run_comprehensive_evaluation(self, model_path: str = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on all benchmarks
        
        Args:
            model_path: Path to trained model (optional)
            
        Returns:
            Complete evaluation results
        """
        logger.info("🚀 Starting Comprehensive Medical LLM Evaluation...")
        
        # Setup model
        self.setup_model_for_evaluation(model_path)
        
        # Load benchmark datasets
        self.load_benchmark_datasets()
        
        # Run evaluations
        all_results = {
            'model_info': {
                'model_name': self.cfg.model.base_model_name,
                'model_path': model_path,
                'evaluation_timestamp': datetime.now().isoformat()
            },
            'benchmark_results': {},
            'summary': {}
        }
        
        # Evaluate on each benchmark
        total_questions = 0
        total_correct = 0
        
        for dataset_name, dataset in self.benchmark_datasets.items():
            try:
                results = self.evaluate_multiple_choice(dataset, dataset_name)
                all_results['benchmark_results'][dataset_name] = results
                
                total_questions += results['total_questions']
                total_correct += results['correct_answers']
                
            except Exception as e:
                logger.error(f"Error evaluating {dataset_name}: {e}")
                all_results['benchmark_results'][dataset_name] = {'error': str(e)}
        
        # Calculate overall summary
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
        all_results['summary'] = {
            'overall_accuracy': overall_accuracy,
            'total_questions': total_questions,
            'total_correct': total_correct,
            'benchmarks_evaluated': len(self.benchmark_datasets),
            'memory_usage': self._get_memory_usage()
        }
        
        # Save results
        self.evaluation_results = all_results
        self._save_evaluation_results(all_results)
        
        logger.info("🎉 Comprehensive Evaluation Complete!")
        logger.info(f"📊 Overall Accuracy: {overall_accuracy:.3f} ({total_correct}/{total_questions})")
        
        return all_results
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        if torch.cuda.is_available():
            return {
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3
            }
        return {}
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
        filepath = os.path.join("evaluation", filename)
        
        os.makedirs("evaluation", exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Evaluation results saved to: {filepath}")
    
    def create_evaluation_report(self, results: Dict[str, Any] = None) -> str:
        """
        Create a formatted evaluation report
        
        Args:
            results: Evaluation results (optional, uses last evaluation if None)
            
        Returns:
            Formatted report string
        """
        if results is None:
            results = self.evaluation_results
        
        if not results:
            return "No evaluation results available."
        
        report = []
        report.append("=" * 60)
        report.append("MEDICAL LLM EVALUATION REPORT")
        report.append("=" * 60)
        
        # Model info
        model_info = results.get('model_info', {})
        report.append(f"Model: {model_info.get('model_name', 'Unknown')}")
        report.append(f"Evaluation Date: {model_info.get('evaluation_timestamp', 'Unknown')}")
        report.append("")
        
        # Summary
        summary = results.get('summary', {})
        report.append("OVERALL RESULTS:")
        report.append(f"  Overall Accuracy: {summary.get('overall_accuracy', 0):.3f}")
        report.append(f"  Total Questions: {summary.get('total_questions', 0)}")
        report.append(f"  Correct Answers: {summary.get('total_correct', 0)}")
        report.append(f"  Benchmarks Evaluated: {summary.get('benchmarks_evaluated', 0)}")
        report.append("")
        
        # Benchmark details
        benchmark_results = results.get('benchmark_results', {})
        report.append("BENCHMARK DETAILS:")
        for dataset_name, dataset_results in benchmark_results.items():
            if 'error' in dataset_results:
                report.append(f"  {dataset_name}: ERROR - {dataset_results['error']}")
            else:
                accuracy = dataset_results.get('accuracy', 0)
                total = dataset_results.get('total_questions', 0)
                correct = dataset_results.get('correct_answers', 0)
                report.append(f"  {dataset_name}: {accuracy:.3f} ({correct}/{total})")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)

# Convenience functions
def quick_evaluate(model_path: str = None, use_dummy_data: bool = True) -> Dict[str, Any]:
    """
    Quick evaluation function for testing
    
    Args:
        model_path: Path to trained model
        use_dummy_data: Whether to use dummy data for quick testing
        
    Returns:
        Evaluation results dictionary
    """
    evaluator = MedicalLLMEvaluator()
    
    if use_dummy_data:
        # Use dummy benchmark for quick testing
        dummy_data = evaluator._create_dummy_medical_benchmark()
        evaluator.benchmark_datasets = {'dummy_medical': dummy_data}
    
    return evaluator.run_comprehensive_evaluation(model_path)

def compare_models(model_paths: List[str], model_names: List[str] = None) -> pd.DataFrame:
    """
    Compare multiple models on medical benchmarks
    
    Args:
        model_paths: List of paths to trained models
        model_names: Optional list of model names for display
        
    Returns:
        DataFrame with comparison results
    """
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(model_paths))]
    
    comparison_results = []
    
    for model_path, model_name in zip(model_paths, model_names):
        logger.info(f"Evaluating {model_name}...")
        results = quick_evaluate(model_path, use_dummy_data=True)
        
        comparison_results.append({
            'Model': model_name,
            'Overall_Accuracy': results['summary']['overall_accuracy'],
            'Total_Questions': results['summary']['total_questions'],
            'Correct_Answers': results['summary']['total_correct'],
        })
    
    return pd.DataFrame(comparison_results) 