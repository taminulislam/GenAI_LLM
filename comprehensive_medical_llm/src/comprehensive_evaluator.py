"""
Comprehensive Medical Evaluator with Hallucination Detection
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    """Comprehensive evaluator with multiple metrics and hallucination detection"""
    
    def __init__(self, cfg=None):
        self.cfg = cfg
        try:
            import evaluate
            self.bleu = evaluate.load("bleu")
            self.rouge = evaluate.load("rouge")
        except:
            self.bleu = None
            self.rouge = None
            logger.warning("Could not load evaluation metrics")
        
    def evaluate_model(self, model_name: str, test_dataset: Dataset) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        logger.info(f"Starting comprehensive evaluation for {model_name}")
        
        # Load model for evaluation
        try:
            # Handle local model paths correctly
            if model_name.startswith(('./', '../', '/')) or '\\' in model_name:
                # This is a local path - use directly
                model_path = model_name
            else:
                # This might be a HuggingFace model name
                model_path = model_name
            
            logger.info(f"Loading model from path: {model_path}")
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"Could not load model: {e}")
            return {'error': str(e)}
        
        # Generate predictions
        predictions = []
        references = []
        
        # Sample subset for evaluation (to manage time)
        eval_samples = min(100, len(test_dataset))
        test_subset = test_dataset.shuffle(seed=42).select(range(eval_samples))
        
        logger.info(f"Evaluating on {eval_samples} samples")
        
        for i, example in enumerate(test_subset):
            if i % 20 == 0:
                logger.info(f"Evaluating sample {i+1}/{eval_samples}")
            
            try:
                # Generate prediction
                input_text = example['input_text']
                target_text = example['target_text']
                
                # Tokenize and generate
                inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode prediction
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                prediction = generated[len(input_text):].strip()
                
                predictions.append(prediction)
                references.append(target_text)
                
            except Exception as e:
                logger.warning(f"Error evaluating sample {i}: {e}")
                predictions.append("")
                references.append(target_text)
        
        # Calculate metrics
        results = self._calculate_comprehensive_metrics(predictions, references)
        
        logger.info("Comprehensive evaluation completed")
        return results
    
    def _calculate_comprehensive_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """Calculate comprehensive metrics including hallucination detection"""
        
        # Basic accuracy (exact match)
        exact_matches = [pred.strip().lower() == ref.strip().lower() for pred, ref in zip(predictions, references)]
        accuracy = np.mean(exact_matches)
        
        # BLEU Score
        try:
            if self.bleu:
                bleu_score = self.bleu.compute(predictions=predictions, references=[[ref] for ref in references])['bleu']
            else:
                bleu_score = 0.0
        except:
            bleu_score = 0.0
        
        # ROUGE-L Score
        try:
            if self.rouge:
                rouge_score = self.rouge.compute(predictions=predictions, references=references)['rougeL']
            else:
                rouge_score = 0.0
        except:
            rouge_score = 0.0
        
        # Hallucination Detection
        hallucination_score = self._detect_hallucinations(predictions, references)
        
        # Factual Consistency
        factual_consistency = self._calculate_factual_consistency(predictions, references)
        
        return {
            'accuracy': accuracy,
            'bleu_score': bleu_score,
            'rouge_l': rouge_score,
            'hallucination_score': hallucination_score,
            'factual_consistency': factual_consistency,
            'total_samples': len(predictions),
            'exact_matches': sum(exact_matches)
        }
    
    def _detect_hallucinations(self, predictions: List[str], references: List[str]) -> float:
        """Simple hallucination detection based on content analysis"""
        hallucination_count = 0
        
        for pred, ref in zip(predictions, references):
            # Check for obvious hallucinations
            pred_lower = pred.lower()
            ref_lower = ref.lower()
            
            # Detect if prediction contains medical terms not in reference
            medical_terms = ['diagnosis', 'treatment', 'medication', 'surgery', 'therapy', 'disease', 'condition']
            
            pred_has_medical = any(term in pred_lower for term in medical_terms)
            ref_has_medical = any(term in ref_lower for term in medical_terms)
            
            # If prediction introduces medical terms not in reference, potential hallucination
            if pred_has_medical and not ref_has_medical and len(pred.split()) > 5:
                hallucination_count += 1
            
            # Check for very long responses that might contain hallucinations
            if len(pred.split()) > len(ref.split()) * 3 and len(pred.split()) > 10:
                hallucination_count += 1
        
        return hallucination_count / len(predictions) if predictions else 0.0
    
    def _calculate_factual_consistency(self, predictions: List[str], references: List[str]) -> float:
        """Calculate factual consistency between predictions and references"""
        consistency_scores = []
        
        for pred, ref in zip(predictions, references):
            # Simple token overlap for factual consistency
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            
            if ref_tokens:
                overlap = len(pred_tokens.intersection(ref_tokens))
                consistency = overlap / len(ref_tokens)
                consistency_scores.append(min(consistency, 1.0))
            else:
                consistency_scores.append(0.0)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0 