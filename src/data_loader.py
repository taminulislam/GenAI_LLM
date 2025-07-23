"""
Medical LLM Data Loader Module
Handles dataset loading, preprocessing, and formatting for medical training data
"""

import os
import logging
from typing import Optional, Dict, List, Union, Tuple
from datasets import Dataset, load_dataset
import pandas as pd
from .config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalDataLoader:
    """Handles loading and preprocessing of medical datasets"""
    
    def __init__(self, cfg=None):
        """
        Initialize DataLoader with configuration
        
        Args:
            cfg: Configuration object, defaults to global config
        """
        self.cfg = cfg if cfg else config
        self.dataset = None
        self.processed_dataset = None
        
    def load_medical_dataset(self, use_dummy: Optional[bool] = None) -> Dataset:
        """
        Load medical dataset (real or dummy data)
        
        Args:
            use_dummy: Whether to use dummy data, defaults to config
            
        Returns:
            Dataset object
        """
        if use_dummy is None:
            use_dummy = self.cfg.data.use_dummy_data
            
        if use_dummy:
            logger.info("Creating dummy medical dataset for testing...")
            dataset = self._create_dummy_dataset()
        else:
            logger.info("Loading real medical datasets...")
            dataset = self._load_real_dataset()
            
        self.dataset = dataset
        logger.info(f"📚 Dataset loaded with {len(dataset)} samples")
        
        return dataset
    
    def _create_dummy_dataset(self) -> Dataset:
        """Create a dummy medical dataset for testing"""
        dummy_medical_data = {
            "instruction": [
                "Diagnose the following symptoms.",
                "What is the treatment for hypertension?",
                "Explain the risk factors for diabetes.",
                "What are the symptoms of pneumonia?",
                "How to manage chronic pain?",
                "What causes migraine headaches?",
                "Explain heart attack symptoms.",
                "What is Type 2 diabetes?",
                "How to prevent stroke?",
                "What causes asthma?",
                "Explain the symptoms of depression.",
                "What is the treatment for anxiety?",
                "How to manage high cholesterol?",
                "What are the signs of dehydration?",
                "Explain the causes of insomnia."
            ],
            "input": [
                "Patient has fever, cough, and fatigue.",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                ""
            ],
            "output": [
                "Based on the symptoms, this could indicate a viral infection such as influenza or COVID-19. Recommend rest, hydration, and monitoring symptoms.",
                "Treatment for hypertension typically includes lifestyle changes (diet, exercise, weight management) and may require antihypertensive medications such as ACE inhibitors, beta-blockers, or diuretics.",
                "Risk factors for diabetes include family history, obesity, sedentary lifestyle, age over 45, high blood pressure, and gestational diabetes history.",
                "Pneumonia symptoms include persistent cough, fever, chills, shortness of breath, chest pain, and fatigue. Severe cases may require immediate medical attention.",
                "Chronic pain management includes a combination of medications, physical therapy, lifestyle modifications, stress management, and sometimes psychological support.",
                "Migraines can be triggered by stress, certain foods, hormonal changes, lack of sleep, bright lights, or strong smells. Treatment involves identifying triggers and preventive medications.",
                "Heart attack symptoms include chest pain or pressure, shortness of breath, nausea, sweating, and pain radiating to arms, neck, or jaw. Seek immediate emergency care.",
                "Type 2 diabetes is a chronic condition where the body becomes resistant to insulin or doesn't produce enough insulin, leading to elevated blood sugar levels.",
                "Stroke prevention includes controlling blood pressure, managing cholesterol, staying physically active, eating a healthy diet, avoiding smoking, and limiting alcohol.",
                "Asthma is caused by inflammation and narrowing of airways, often triggered by allergens, respiratory infections, exercise, cold air, or stress.",
                "Depression symptoms include persistent sadness, loss of interest in activities, fatigue, sleep disturbances, appetite changes, and difficulty concentrating.",
                "Anxiety treatment may include therapy (cognitive behavioral therapy), medications (SSRIs, benzodiazepines), lifestyle changes, and stress management techniques.",
                "High cholesterol management includes dietary changes (low saturated fat), regular exercise, weight management, and possibly statin medications.",
                "Dehydration signs include thirst, dry mouth, decreased urination, dark urine, fatigue, dizziness, and in severe cases, confusion or rapid heartbeat.",
                "Insomnia causes include stress, anxiety, poor sleep habits, medical conditions, medications, caffeine, and environmental factors like noise or light."
            ]
        }
        
        # Limit to configured dummy data size
        size = min(self.cfg.data.dummy_data_size, len(dummy_medical_data["instruction"]))
        for key in dummy_medical_data:
            dummy_medical_data[key] = dummy_medical_data[key][:size]
            
        return Dataset.from_dict(dummy_medical_data)
    
    def _load_real_dataset(self) -> Dataset:
        """Load real medical dataset from HuggingFace"""
        try:
            # Load the primary medical dataset
            dataset = load_dataset(
                self.cfg.data.dataset_name,
                self.cfg.data.dataset_config,
                split="train"
            )
            
            # Limit dataset size if specified
            if self.cfg.data.max_samples > 0:
                dataset = dataset.select(range(min(self.cfg.data.max_samples, len(dataset))))
                
            return dataset
            
        except Exception as e:
            logger.error(f"❌ Error loading real dataset: {e}")
            logger.info("🔄 Falling back to dummy dataset...")
            return self._create_dummy_dataset()
    
    def preprocess_dataset(self, dataset: Optional[Dataset] = None) -> Dataset:
        """
        Preprocess dataset into instruction-following format
        
        Args:
            dataset: Dataset to preprocess, defaults to stored dataset
            
        Returns:
            Preprocessed dataset
        """
        if dataset is None:
            dataset = self.dataset
            
        if dataset is None:
            raise ValueError("No dataset available. Call load_medical_dataset first.")
        
        logger.info("🔄 Preprocessing dataset...")
        
        # Apply formatting based on dataset structure
        if self._is_medical_qa_format(dataset):
            processed = dataset.map(self._format_medical_qa)
        else:
            processed = dataset.map(self._format_generic_medical)
            
        self.processed_dataset = processed
        logger.info("✅ Dataset preprocessing completed")
        
        return processed
    
    def _is_medical_qa_format(self, dataset: Dataset) -> bool:
        """Check if dataset is in medical QA format"""
        sample = dataset[0]
        return all(key in sample for key in ["instruction", "input", "output"])
    
    def _format_medical_qa(self, example: Dict) -> Dict:
        """Format medical QA example into instruction-following format"""
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output_text = example.get('output', '')
        
        # Combine instruction and input
        if input_text and input_text.strip():
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
        else:
            prompt = f"Instruction: {instruction}\nResponse:"
        
        # Create the full text for training
        full_text = f"{prompt} {output_text}"
        
        return {
            "text": full_text,
            "prompt": prompt,
            "completion": output_text,  # Changed from response to completion for SFTTrainer
            "instruction": instruction,
            "input": input_text
        }
    
    def _format_generic_medical(self, example: Dict) -> Dict:
        """Format generic medical example"""
        # Handle different possible field names
        question = example.get('question', example.get('query', ''))
        answer = example.get('answer', example.get('response', ''))
        
        if not question or not answer:
            # Try other common field names
            question = example.get('text', '')[:100] + "..."  # Truncate if too long
            answer = example.get('text', '')[100:]  # Rest as answer
            
        prompt = f"Medical Question: {question}\nAnswer:"
        full_text = f"{prompt} {answer}"
        
        return {
            "text": full_text,
            "prompt": prompt,
            "completion": answer,  # Changed from response to completion for SFTTrainer
            "question": question
        }
    
    def split_dataset(self, dataset: Optional[Dataset] = None, 
                     train_ratio: Optional[float] = None) -> Tuple[Dataset, Dataset]:
        """
        Split dataset into train and validation sets
        
        Args:
            dataset: Dataset to split, defaults to processed dataset
            train_ratio: Ratio for training split, defaults to config
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if dataset is None:
            dataset = self.processed_dataset or self.dataset
            
        if dataset is None:
            raise ValueError("No dataset available")
            
        if train_ratio is None:
            train_ratio = self.cfg.data.train_split_ratio
            
        # Calculate split indices
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        
        # Split dataset
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, total_size))
        
        logger.info(f"📊 Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation")
        
        return train_dataset, val_dataset
    
    def get_dataset_info(self, dataset: Optional[Dataset] = None) -> Dict:
        """Get information about the dataset"""
        if dataset is None:
            dataset = self.processed_dataset or self.dataset
            
        if dataset is None:
            return {"status": "No dataset loaded"}
        
        try:
            sample = dataset[0]
            
            return {
                "status": "Dataset loaded",
                "size": len(dataset),
                "columns": list(sample.keys()),
                "sample_text_length": len(sample.get("text", "")),
                "has_instruction_format": "instruction" in sample,
                "text_field": self.cfg.data.text_field
            }
        except Exception as e:
            return {"status": f"Error getting dataset info: {e}"}
    
    def validate_dataset(self, dataset: Optional[Dataset] = None) -> Dict:
        """Validate dataset format and content"""
        if dataset is None:
            dataset = self.processed_dataset or self.dataset
            
        if dataset is None:
            return {"valid": False, "error": "No dataset available"}
        
        try:
            # Check required field exists
            sample = dataset[0]
            text_field = self.cfg.data.text_field
            
            if text_field not in sample:
                return {
                    "valid": False,
                    "error": f"Required field '{text_field}' not found",
                    "available_fields": list(sample.keys())
                }
            
            # Check text content
            text_lengths = [len(item[text_field]) for item in dataset.select(range(min(100, len(dataset))))]
            avg_length = sum(text_lengths) / len(text_lengths)
            
            # Basic validation checks
            checks = {
                "has_text_field": text_field in sample,
                "non_empty_texts": all(len(item[text_field]) > 0 for item in dataset.select(range(min(10, len(dataset))))),
                "reasonable_length": 10 < avg_length < 10000,
                "size_check": len(dataset) > 0
            }
            
            return {
                "valid": all(checks.values()),
                "checks": checks,
                "average_text_length": avg_length,
                "dataset_size": len(dataset)
            }
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {e}"}
    
    def save_dataset(self, dataset: Optional[Dataset] = None, save_path: Optional[str] = None):
        """Save dataset to disk"""
        if dataset is None:
            dataset = self.processed_dataset or self.dataset
            
        if dataset is None:
            raise ValueError("No dataset to save")
            
        if save_path is None:
            save_path = os.path.join(self.cfg.system.data_dir, "processed_medical_dataset")
            
        try:
            dataset.save_to_disk(save_path)
            logger.info(f"💾 Dataset saved to {save_path}")
        except Exception as e:
            logger.error(f"❌ Error saving dataset: {e}")
            raise
    
    def load_saved_dataset(self, load_path: str) -> Dataset:
        """Load dataset from disk"""
        try:
            dataset = Dataset.load_from_disk(load_path)
            self.dataset = dataset
            logger.info(f"✅ Dataset loaded from {load_path}")
            return dataset
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            raise

# Convenience functions
def load_medical_data(use_dummy: bool = True) -> Dataset:
    """
    Convenience function to load medical dataset
    
    Args:
        use_dummy: Whether to use dummy data
        
    Returns:
        Dataset object
    """
    loader = MedicalDataLoader()
    return loader.load_medical_dataset(use_dummy=use_dummy)

def preprocess_medical_data(dataset: Dataset) -> Dataset:
    """
    Convenience function to preprocess medical dataset
    
    Args:
        dataset: Dataset to preprocess
        
    Returns:
        Preprocessed dataset
    """
    loader = MedicalDataLoader()
    return loader.preprocess_dataset(dataset)

def get_sample_data(size: int = 5) -> Dict:
    """Get sample data for testing"""
    loader = MedicalDataLoader()
    dataset = loader.load_medical_dataset(use_dummy=True)
    processed = loader.preprocess_dataset(dataset)
    
    sample_size = min(size, len(processed))
    samples = processed.select(range(sample_size))
    
    return {
        "raw_samples": [dataset[i] for i in range(sample_size)],
        "processed_samples": [processed[i] for i in range(sample_size)],
        "info": loader.get_dataset_info(processed)
    } 