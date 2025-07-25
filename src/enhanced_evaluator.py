"""
Enhanced Medical LLM Evaluator Module
Comprehensive evaluation with multiple metrics, hallucination detection, and factual consistency probing
"""

import os
import json
import logging
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import accuracy_score
from datasets import Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import re
from datetime import datetime
import math

# Import evaluation metrics
try:
    from evaluate import load
    bleu_metric = load("bleu")
    rouge_metric = load("rouge")
except ImportError:
    bleu_metric = None
    rouge_metric = None
    logging.warning("Evaluate library not found. Some metrics may not be available.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMedicalEvaluator:
    """Enhanced evaluator with comprehensive metrics and hallucination detection"""
    
    def __init__(self):
        """Initialize Enhanced Evaluator"""
        self.model_manager = None
        self.generator = None
        self.tokenizer = None
        self.model = None
        self.evaluation_results = {}
        
        # Medical knowledge keywords for hallucination detection
        self.medical_keywords = {
            'anatomy': ['heart', 'lung', 'liver', 'kidney', 'brain', 'muscle', 'bone', 'blood'],
            'symptoms': ['fever', 'pain', 'cough', 'nausea', 'headache', 'fatigue', 'swelling'],
            'treatments': ['medication', 'surgery', 'therapy', 'rest', 'exercise', 'diet'],
            'conditions': ['diabetes', 'hypertension', 'cancer', 'infection', 'asthma', 'arthritis']
        }
        
    def setup_model(self, model_path: str = None, model_manager=None):
        """
        Setup model for evaluation
        
        Args:
            model_path: Path to trained model
            model_manager: Pre-configured ModelManager instance
        """
        if model_manager:
            self.model_manager = model_manager
            self.model = model_manager.model
            self.tokenizer = model_manager.tokenizer
        elif model_path and os.path.exists(model_path):
            logger.info(f"Loading model from: {model_path}")
            # Load model directly
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        else:
            raise ValueError("Either model_path or model_manager must be provided")
        
        # Setup generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        logger.info("✅ Model ready for enhanced evaluation")
    
    def generate_response(self, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Generate response from the model
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            Generated response
        """
        try:
            result = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            response = result[0]['generated_text'].strip()
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
    
    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of generated text
        
        Args:
            text: Text to calculate perplexity for
            
        Returns:
            Perplexity score
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            return perplexity
        except Exception as e:
            logger.warning(f"Could not calculate perplexity: {e}")
            return float('inf')
    
    def calculate_bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU score"""
        if bleu_metric is None:
            return self._simple_bleu(predictions, references)
        
        try:
            # Format references for BLEU (each reference should be a list)
            formatted_refs = [[ref] for ref in references]
            result = bleu_metric.compute(predictions=predictions, references=formatted_refs)
            return result['bleu']
        except Exception as e:
            logger.warning(f"Could not calculate BLEU: {e}")
            return self._simple_bleu(predictions, references)
    
    def _simple_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Simple BLEU approximation using word overlap"""
        scores = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            if ref_words:
                overlap = len(pred_words.intersection(ref_words))
                score = overlap / len(ref_words)
                scores.append(score)
            else:
                scores.append(0.0)
        return np.mean(scores) if scores else 0.0
    
    def calculate_rouge_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        if rouge_metric is None:
            return self._simple_rouge(predictions, references)
        
        try:
            result = rouge_metric.compute(predictions=predictions, references=references)
            return {
                'rouge1': result['rouge1'],
                'rouge2': result['rouge2'],
                'rougeL': result['rougeL']
            }
        except Exception as e:
            logger.warning(f"Could not calculate ROUGE: {e}")
            return self._simple_rouge(predictions, references)
    
    def _simple_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Simple ROUGE approximation"""
        rouge_scores = []
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            if ref_words:
                # Simple longest common subsequence approximation
                common = set(pred_words).intersection(set(ref_words))
                rouge_l = len(common) / len(ref_words)
                rouge_scores.append(rouge_l)
            else:
                rouge_scores.append(0.0)
        
        avg_rouge = np.mean(rouge_scores) if rouge_scores else 0.0
        return {
            'rouge1': avg_rouge,
            'rouge2': avg_rouge * 0.8,  # Approximation
            'rougeL': avg_rouge
        }
    
    def detect_hallucinations(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """
        Detect potential hallucinations in model responses
        
        Args:
            predictions: Model predictions
            references: Reference answers
            
        Returns:
            Dictionary with hallucination detection results
        """
        logger.info("🔍 Performing hallucination detection...")
        
        hallucination_flags = []
        hallucination_reasons = []
        severity_scores = []
        
        for pred, ref in zip(predictions, references):
            flags, reasons, severity = self._analyze_single_response(pred, ref)
            hallucination_flags.append(flags)
            hallucination_reasons.append(reasons)
            severity_scores.append(severity)
        
        # Calculate overall statistics
        total_hallucinations = sum(len(flags) for flags in hallucination_flags)
        hallucination_rate = total_hallucinations / len(predictions) if predictions else 0
        avg_severity = np.mean([np.mean(scores) if scores else 0 for scores in severity_scores])
        
        # Categorize hallucination types
        all_reasons = [reason for reasons in hallucination_reasons for reason in reasons]
        reason_counts = {reason: all_reasons.count(reason) for reason in set(all_reasons)}
        
        return {
            'hallucination_rate': hallucination_rate,
            'total_hallucinations': total_hallucinations,
            'avg_severity': avg_severity,
            'hallucination_types': reason_counts,
            'detailed_flags': hallucination_flags,
            'detailed_reasons': hallucination_reasons,
            'severity_scores': severity_scores
        }
    
    def _analyze_single_response(self, prediction: str, reference: str) -> Tuple[List[str], List[str], List[float]]:
        """Analyze a single response for hallucinations"""
        flags = []
        reasons = []
        severity_scores = []
        
        pred_lower = prediction.lower()
        ref_lower = reference.lower()
        
        # Check 1: Excessive length (potential rambling/hallucination)
        pred_words = prediction.split()
        ref_words = reference.split()
        
        if len(pred_words) > len(ref_words) * 3 and len(pred_words) > 20:
            flags.append("excessive_length")
            reasons.append("Response significantly longer than expected")
            severity_scores.append(0.7)
        
        # Check 2: Introduction of new medical terms not in reference
        pred_medical_terms = self._extract_medical_terms(pred_lower)
        ref_medical_terms = self._extract_medical_terms(ref_lower)
        
        new_terms = pred_medical_terms - ref_medical_terms
        if new_terms and len(new_terms) > 2:
            flags.append("new_medical_terms")
            reasons.append(f"Introduced new medical terms: {list(new_terms)[:3]}")
            severity_scores.append(0.6)
        
        # Check 3: Contradictory statements
        if self._contains_contradictions(prediction):
            flags.append("contradictions")
            reasons.append("Contains contradictory statements")
            severity_scores.append(0.8)
        
        # Check 4: Overly specific claims without basis
        if self._contains_specific_claims(prediction, reference):
            flags.append("unsupported_specificity")
            reasons.append("Makes overly specific claims not supported by context")
            severity_scores.append(0.5)
        
        # Check 5: Repetitive content
        if self._is_repetitive(prediction):
            flags.append("repetitive_content")
            reasons.append("Contains repetitive or circular content")
            severity_scores.append(0.4)
        
        return flags, reasons, severity_scores
    
    def _extract_medical_terms(self, text: str) -> set:
        """Extract medical terms from text"""
        medical_terms = set()
        for category, terms in self.medical_keywords.items():
            for term in terms:
                if term in text:
                    medical_terms.add(term)
        return medical_terms
    
    def _contains_contradictions(self, text: str) -> bool:
        """Check for contradictory statements"""
        contradictory_patterns = [
            (r'not.*but.*is', r'is.*but.*not'),
            (r'never.*always', r'always.*never'),
            (r'safe.*dangerous', r'dangerous.*safe'),
            (r'increase.*decrease', r'decrease.*increase')
        ]
        
        text_lower = text.lower()
        for pattern1, pattern2 in contradictory_patterns:
            if re.search(pattern1, text_lower) and re.search(pattern2, text_lower):
                return True
        return False
    
    def _contains_specific_claims(self, prediction: str, reference: str) -> bool:
        """Check for overly specific claims"""
        # Look for specific numbers, percentages, or precise measurements
        specific_patterns = [
            r'\d+\.\d+%',  # Specific percentages
            r'\d+\.\d+ (mg|ml|grams?|liters?)',  # Specific measurements
            r'exactly \d+',  # Exact numbers
            r'precisely',  # Precision claims
        ]
        
        pred_specific = sum(len(re.findall(pattern, prediction.lower())) for pattern in specific_patterns)
        ref_specific = sum(len(re.findall(pattern, reference.lower())) for pattern in specific_patterns)
        
        return pred_specific > ref_specific * 2 and pred_specific > 2
    
    def _is_repetitive(self, text: str) -> bool:
        """Check for repetitive content"""
        sentences = text.split('.')
        if len(sentences) < 3:
            return False
        
        # Check for repeated sentences
        unique_sentences = set(s.strip().lower() for s in sentences if s.strip())
        repetition_ratio = len(unique_sentences) / len([s for s in sentences if s.strip()])
        
        return repetition_ratio < 0.7
    
    def calculate_factual_consistency(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """
        Calculate factual consistency between predictions and references
        
        Args:
            predictions: Model predictions
            references: Reference answers
            
        Returns:
            Dictionary with factual consistency metrics
        """
        logger.info("📊 Calculating factual consistency...")
        
        consistency_scores = []
        medical_accuracy_scores = []
        semantic_similarity_scores = []
        
        for pred, ref in zip(predictions, references):
            # Token overlap consistency
            token_consistency = self._calculate_token_overlap(pred, ref)
            consistency_scores.append(token_consistency)
            
            # Medical fact accuracy
            medical_accuracy = self._calculate_medical_accuracy(pred, ref)
            medical_accuracy_scores.append(medical_accuracy)
            
            # Semantic similarity approximation
            semantic_sim = self._calculate_semantic_similarity(pred, ref)
            semantic_similarity_scores.append(semantic_sim)
        
        return {
            'avg_token_consistency': np.mean(consistency_scores),
            'avg_medical_accuracy': np.mean(medical_accuracy_scores),
            'avg_semantic_similarity': np.mean(semantic_similarity_scores),
            'consistency_distribution': {
                'high_consistency': sum(1 for s in consistency_scores if s > 0.7),
                'medium_consistency': sum(1 for s in consistency_scores if 0.3 <= s <= 0.7),
                'low_consistency': sum(1 for s in consistency_scores if s < 0.3)
            }
        }
    
    def _calculate_token_overlap(self, prediction: str, reference: str) -> float:
        """Calculate token overlap between prediction and reference"""
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if not ref_tokens:
            return 0.0
        
        overlap = len(pred_tokens.intersection(ref_tokens))
        return overlap / len(ref_tokens)
    
    def _calculate_medical_accuracy(self, prediction: str, reference: str) -> float:
        """Calculate medical fact accuracy"""
        pred_medical = self._extract_medical_terms(prediction.lower())
        ref_medical = self._extract_medical_terms(reference.lower())
        
        if not ref_medical:
            return 1.0 if not pred_medical else 0.5
        
        correct_medical = len(pred_medical.intersection(ref_medical))
        return correct_medical / len(ref_medical)
    
    def _calculate_semantic_similarity(self, prediction: str, reference: str) -> float:
        """Simple semantic similarity based on word order and context"""
        pred_words = prediction.lower().split()
        ref_words = reference.lower().split()
        
        # Simple bag-of-words similarity with position weighting
        similarity = 0.0
        for i, ref_word in enumerate(ref_words):
            if ref_word in pred_words:
                # Give higher weight to words in similar positions
                pred_positions = [j for j, pred_word in enumerate(pred_words) if pred_word == ref_word]
                if pred_positions:
                    min_distance = min(abs(i - pos) for pos in pred_positions)
                    position_weight = 1.0 / (1.0 + min_distance * 0.1)
                    similarity += position_weight
        
        return similarity / len(ref_words) if ref_words else 0.0
    
    def run_comprehensive_evaluation(self, eval_dataset: Dataset, num_samples: int = 100) -> Dict[str, Any]:
        """
        Run comprehensive evaluation with all metrics
        
        Args:
            eval_dataset: Evaluation dataset
            num_samples: Number of samples to evaluate (default 100)
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"🚀 Starting comprehensive evaluation on {num_samples} samples...")
        
        # Limit evaluation to specified number of samples
        eval_subset = eval_dataset.select(range(min(num_samples, len(eval_dataset))))
        
        predictions = []
        references = []
        prompts = []
        perplexity_scores = []
        
        # Generate predictions
        logger.info("Generating predictions...")
        for i, sample in enumerate(eval_subset):
            prompt = sample.get('prompt', '')
            reference = sample.get('completion', sample.get('output', ''))
            
            # Generate prediction
            prediction = self.generate_response(prompt, max_new_tokens=200)
            
            # Calculate perplexity
            perplexity = self.calculate_perplexity(prediction)
            
            predictions.append(prediction)
            references.append(reference)
            prompts.append(prompt)
            perplexity_scores.append(perplexity)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(eval_subset)} samples")
        
        # Calculate all metrics
        logger.info("Calculating metrics...")
        
        # Basic accuracy (exact match)
        exact_matches = [pred.strip().lower() == ref.strip().lower() for pred, ref in zip(predictions, references)]
        accuracy = np.mean(exact_matches)
        
        # BLEU score
        bleu_score = self.calculate_bleu_score(predictions, references)
        
        # ROUGE scores
        rouge_scores = self.calculate_rouge_score(predictions, references)
        
        # Perplexity
        avg_perplexity = np.mean(perplexity_scores)
        
        # Hallucination detection
        hallucination_results = self.detect_hallucinations(predictions, references)
        
        # Factual consistency
        factual_consistency = self.calculate_factual_consistency(predictions, references)
        
        # Compile results
        results = {
            'evaluation_info': {
                'num_samples': len(eval_subset),
                'model_type': type(self.model).__name__ if self.model else 'Unknown',
                'timestamp': datetime.now().isoformat()
            },
            'basic_metrics': {
                'accuracy': accuracy,
                'exact_matches': sum(exact_matches),
                'total_samples': len(predictions)
            },
            'language_metrics': {
                'bleu_score': bleu_score,
                'rouge_scores': rouge_scores,
                'avg_perplexity': avg_perplexity,
                'perplexity_distribution': {
                    'low_perplexity': sum(1 for p in perplexity_scores if p < 50),
                    'medium_perplexity': sum(1 for p in perplexity_scores if 50 <= p <= 200),
                    'high_perplexity': sum(1 for p in perplexity_scores if p > 200)
                }
            },
            'hallucination_analysis': hallucination_results,
            'factual_consistency': factual_consistency,
            'sample_results': [
                {
                    'prompt': prompts[i][:100] + "..." if len(prompts[i]) > 100 else prompts[i],
                    'reference': references[i],
                    'prediction': predictions[i],
                    'exact_match': exact_matches[i],
                    'perplexity': perplexity_scores[i]
                }
                for i in range(min(5, len(predictions)))  # Show first 5 samples
            ]
        }
        
        # Calculate overall quality score
        quality_score = self._calculate_overall_quality_score(results)
        results['overall_quality_score'] = quality_score
        
        # Save results
        self._save_results(results)
        
        logger.info("✅ Comprehensive evaluation completed!")
        self._print_summary(results)
        
        return results
    
    def _calculate_overall_quality_score(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall quality score based on all metrics"""
        weights = {
            'accuracy': 0.3,
            'bleu': 0.2,
            'rouge': 0.2,
            'factual_consistency': 0.2,
            'hallucination_penalty': 0.1
        }
        
        accuracy = results['basic_metrics']['accuracy']
        bleu = results['language_metrics']['bleu_score']
        rouge = results['language_metrics']['rouge_scores']['rougeL']
        factual = results['factual_consistency']['avg_token_consistency']
        hallucination_rate = results['hallucination_analysis']['hallucination_rate']
        
        # Calculate weighted score
        quality_score = (
            weights['accuracy'] * accuracy +
            weights['bleu'] * bleu +
            weights['rouge'] * rouge +
            weights['factual_consistency'] * factual +
            weights['hallucination_penalty'] * max(0, 1 - hallucination_rate)
        )
        
        return {
            'overall_score': quality_score,
            'components': {
                'accuracy_score': accuracy * weights['accuracy'],
                'bleu_score': bleu * weights['bleu'],
                'rouge_score': rouge * weights['rouge'],
                'factual_score': factual * weights['factual_consistency'],
                'hallucination_score': max(0, 1 - hallucination_rate) * weights['hallucination_penalty']
            }
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_evaluation_results_{timestamp}.json"
        os.makedirs("evaluation", exist_ok=True)
        filepath = os.path.join("evaluation", filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to: {filepath}")
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("ENHANCED MEDICAL LLM EVALUATION SUMMARY")
        print("="*60)
        
        basic = results['basic_metrics']
        language = results['language_metrics']
        hallucination = results['hallucination_analysis']
        factual = results['factual_consistency']
        quality = results['overall_quality_score']
        
        print(f"📊 Basic Metrics:")
        print(f"   Accuracy: {basic['accuracy']:.4f}")
        print(f"   Exact Matches: {basic['exact_matches']}/{basic['total_samples']}")
        
        print(f"\n📝 Language Quality:")
        print(f"   BLEU Score: {language['bleu_score']:.4f}")
        print(f"   ROUGE-L: {language['rouge_scores']['rougeL']:.4f}")
        print(f"   Average Perplexity: {language['avg_perplexity']:.2f}")
        
        print(f"\n🔍 Hallucination Analysis:")
        print(f"   Hallucination Rate: {hallucination['hallucination_rate']:.4f}")
        print(f"   Average Severity: {hallucination['avg_severity']:.4f}")
        print(f"   Total Flags: {hallucination['total_hallucinations']}")
        
        print(f"\n✅ Factual Consistency:")
        print(f"   Token Consistency: {factual['avg_token_consistency']:.4f}")
        print(f"   Medical Accuracy: {factual['avg_medical_accuracy']:.4f}")
        print(f"   Semantic Similarity: {factual['avg_semantic_similarity']:.4f}")
        
        print(f"\n🎯 Overall Quality Score: {quality['overall_score']:.4f}")
        print("="*60)

# Convenience function
def run_enhanced_evaluation(model_path: str, eval_dataset: Dataset, num_samples: int = 100):
    """Run enhanced evaluation with a single function call"""
    evaluator = EnhancedMedicalEvaluator()
    evaluator.setup_model(model_path=model_path)
    return evaluator.run_comprehensive_evaluation(eval_dataset, num_samples) 