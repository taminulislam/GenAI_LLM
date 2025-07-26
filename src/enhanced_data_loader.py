"""
Enhanced Medical LLM Data Loader Module
Handles loading and combining multiple medical datasets for comprehensive training
"""

import os
import logging
from typing import Optional, Dict, List, Union, Tuple
from datasets import Dataset, load_dataset, concatenate_datasets
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMedicalDataLoader:
    """Enhanced data loader for multiple medical datasets with comprehensive preprocessing"""
    
    def __init__(self):
        """Initialize Enhanced Data Loader"""
        self.datasets = {}
        self.combined_dataset = None
        self.processed_dataset = None
        self.train_dataset = None
        self.eval_dataset = None
        
    def load_multiple_medical_datasets(self) -> Dict[str, Dataset]:
        """
        Load multiple medical datasets for training
        
        Returns:
            Dictionary of loaded datasets
        """
        logger.info("Loading multiple medical datasets...")
        
        # Dataset 1: MedMCQA (Medical Multiple Choice QA)
        try:
            logger.info("Loading MedMCQA dataset...")
            medmcqa = load_dataset("medmcqa", split="train[:3000]")  # Reduced for better balance
            self.datasets['medmcqa'] = medmcqa
            logger.info(f"✅ MedMCQA loaded: {len(medmcqa)} samples")
        except Exception as e:
            logger.warning(f"⚠️ Could not load MedMCQA: {e}")
            self.datasets['medmcqa'] = self._create_dummy_mcqa_dataset()
            
        # Dataset 2: Medical QA from lavita (similar domain)
        try:
            logger.info("Loading Medical QA dataset...")
            medical_qa = load_dataset("lavita/medical-qa-datasets", "all-processed", split="train[:2000]")
            self.datasets['medical_qa'] = medical_qa
            logger.info(f"✅ Medical QA loaded: {len(medical_qa)} samples")
        except Exception as e:
            logger.warning(f"⚠️ Could not load Medical QA: {e}")
            # Create larger dummy dataset for better balance
            self.datasets['medical_qa'] = self._create_large_dummy_qa_dataset()
        
        logger.info(f"Successfully loaded {len(self.datasets)} datasets")
        return self.datasets
    
    def _create_dummy_mcqa_dataset(self) -> Dataset:
        """Create dummy MCQA dataset for fallback"""
        dummy_mcqa = {
            "question": [
                "What is the primary function of the heart?",
                "Which vitamin deficiency causes scurvy?",
                "What is the normal range for human body temperature?",
                "Which organ produces insulin?",
                "What is hypertension?",
                "What causes Type 2 diabetes?",
                "Which hormone regulates blood sugar?",
                "What is the function of red blood cells?",
                "Which vitamin is produced by sunlight exposure?",
                "What is the main symptom of anemia?"
            ],
            "opa": ["Pumping blood", "Fighting infection", "Producing hormones", "Filtering toxins"] * 10,
            "opb": ["Digestion", "Vitamin C", "36-37°C", "Liver", "High blood pressure"] * 2,
            "opc": ["Circulation", "Vitamin A", "35-36°C", "Pancreas", "Low blood pressure"] * 2,
            "opd": ["Respiration", "Vitamin D", "37-38°C", "Kidney", "Fast heart rate"] * 2,
            "cop": [1, 2, 2, 3, 2, 1, 2, 1, 3, 2],  # Correct option indices
            "subject_name": ["Anatomy"] * 10,
            "topic_name": ["Cardiovascular"] * 10
        }
        return Dataset.from_dict(dummy_mcqa)
    
    def _create_dummy_qa_dataset(self) -> Dataset:
        """Create dummy QA dataset for fallback"""
        dummy_qa = {
            "instruction": [
                "Answer the following medical question.",
                "Provide medical advice for the given condition.",
                "Explain the treatment for the described symptoms.",
                "Diagnose based on the patient's description.",
                "Provide information about the medical condition."
            ] * 4,
            "input": [
                "What are the symptoms of pneumonia?",
                "How to manage chronic pain?",
                "Patient has chest pain and shortness of breath.",
                "What causes migraine headaches?",
                "Explain heart attack symptoms."
            ] * 4,
            "output": [
                "Pneumonia symptoms include persistent cough, fever, chills, shortness of breath, and chest pain.",
                "Chronic pain management includes medications, physical therapy, lifestyle changes, and psychological support.",
                "Chest pain with shortness of breath may indicate heart problems. Seek immediate medical attention.",
                "Migraines can be triggered by stress, certain foods, hormonal changes, or environmental factors.",
                "Heart attack symptoms include chest pressure, arm pain, nausea, and sweating. Call emergency services."
            ] * 4
        }
        return Dataset.from_dict(dummy_qa)
    
    def _create_large_dummy_qa_dataset(self) -> Dataset:
        """Create larger dummy QA dataset for better balance"""
        medical_questions = [
            "What are the symptoms of pneumonia?",
            "How to manage chronic pain?",
            "Patient has chest pain and shortness of breath.",
            "What causes migraine headaches?",
            "Explain heart attack symptoms.",
            "What is Type 2 diabetes?",
            "How to prevent stroke?",
            "What causes asthma?",
            "Explain the symptoms of depression.",
            "What is the treatment for anxiety?",
            "How to manage high cholesterol?",
            "What are the signs of dehydration?",
            "Explain the causes of insomnia.",
            "What is hypertension?",
            "How to treat bacterial infections?",
            "What are the symptoms of liver disease?",
            "How to manage arthritis pain?",
            "What causes kidney stones?",
            "Explain thyroid disorders.",
            "What is the treatment for allergies?",
        ]
        
        medical_answers = [
            "Pneumonia symptoms include persistent cough, fever, chills, shortness of breath, and chest pain.",
            "Chronic pain management includes medications, physical therapy, lifestyle changes, and psychological support.",
            "Chest pain with shortness of breath may indicate heart problems. Seek immediate medical attention.",
            "Migraines can be triggered by stress, certain foods, hormonal changes, or environmental factors.",
            "Heart attack symptoms include chest pressure, arm pain, nausea, and sweating. Call emergency services.",
            "Type 2 diabetes is a chronic condition where the body becomes resistant to insulin.",
            "Stroke prevention includes controlling blood pressure, managing cholesterol, and staying active.",
            "Asthma is caused by inflammation and narrowing of airways, often triggered by allergens.",
            "Depression symptoms include persistent sadness, loss of interest, fatigue, and sleep disturbances.",
            "Anxiety treatment may include therapy, medications, lifestyle changes, and stress management.",
            "High cholesterol management includes dietary changes, regular exercise, and possibly medications.",
            "Dehydration signs include thirst, dry mouth, decreased urination, and fatigue.",
            "Insomnia causes include stress, anxiety, poor sleep habits, and medical conditions.",
            "Hypertension is high blood pressure that can damage arteries and organs over time.",
            "Bacterial infections are treated with appropriate antibiotics based on the specific bacteria.",
            "Liver disease symptoms include jaundice, abdominal pain, swelling, and fatigue.",
            "Arthritis pain management includes medications, exercise, heat/cold therapy, and lifestyle changes.",
            "Kidney stones form from minerals and salts that crystallize in concentrated urine.",
            "Thyroid disorders affect metabolism and can cause weight changes, fatigue, and mood issues.",
            "Allergy treatment includes antihistamines, avoiding triggers, and sometimes immunotherapy.",
        ]
        
        # Repeat patterns to create more samples
        repetitions = 100  # Creates 2000 samples
        dummy_qa = {
            "instruction": ["Answer the following medical question."] * (len(medical_questions) * repetitions),
            "input": medical_questions * repetitions,
            "output": medical_answers * repetitions
        }
        return Dataset.from_dict(dummy_qa)
    
    def combine_datasets(self) -> Dataset:
        """
        Combine multiple datasets into a unified format
        
        Returns:
            Combined dataset
        """
        logger.info("Combining datasets into unified format...")
        
        combined_samples = []
        
        # Process MedMCQA format
        if 'medmcqa' in self.datasets:
            mcqa_data = self.datasets['medmcqa']
            for sample in mcqa_data:
                # Convert MCQA to instruction format
                question = sample.get('question', '')
                options = [
                    sample.get('opa', ''),
                    sample.get('opb', ''),
                    sample.get('opc', ''),
                    sample.get('opd', '')
                ]
                
                # Create options text
                options_text = '\n'.join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options) if opt])
                
                # Get correct answer
                cop = sample.get('cop', 1)  # Default to A if not found
                correct_letter = chr(64 + cop) if 1 <= cop <= 4 else 'A'
                
                instruction = "Answer the following medical multiple choice question by selecting the correct option."
                input_text = f"Question: {question}\n\nOptions:\n{options_text}"
                output_text = f"The correct answer is {correct_letter}."
                
                combined_samples.append({
                    'instruction': instruction,
                    'input': input_text,
                    'output': output_text,
                    'dataset_source': 'medmcqa',
                    'question_type': 'multiple_choice'
                })
        
        # Process Medical QA format
        if 'medical_qa' in self.datasets:
            qa_data = self.datasets['medical_qa']
            for sample in qa_data:
                instruction = sample.get('instruction', 'Answer the medical question.')
                input_text = sample.get('input', '')
                output_text = sample.get('output', '')
                
                combined_samples.append({
                    'instruction': instruction,
                    'input': input_text,
                    'output': output_text,
                    'dataset_source': 'medical_qa',
                    'question_type': 'open_ended'
                })
        
        # Create combined dataset
        self.combined_dataset = Dataset.from_list(combined_samples)
        logger.info(f"✅ Combined dataset created with {len(self.combined_dataset)} samples")
        
        return self.combined_dataset
    
    def preprocess_combined_dataset(self) -> Dataset:
        """
        Preprocess combined dataset for training
        
        Returns:
            Preprocessed dataset
        """
        if self.combined_dataset is None:
            raise ValueError("No combined dataset available. Run combine_datasets() first.")
        
        logger.info("🔄 Preprocessing combined dataset...")
        
        def format_sample(example):
            instruction = example['instruction']
            input_text = example['input']
            output_text = example['output']
            
            # Create training format
            if input_text and input_text.strip():
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:"
            
            full_text = f"{prompt}\n{output_text}"
            
            return {
                'text': full_text,
                'prompt': prompt,
                'completion': output_text,
                'instruction': instruction,
                'input': input_text,
                'dataset_source': example['dataset_source'],
                'question_type': example['question_type']
            }
        
        self.processed_dataset = self.combined_dataset.map(format_sample)
        logger.info("✅ Dataset preprocessing completed")
        
        return self.processed_dataset
    
    def create_train_eval_split(self, train_ratio: float = 0.9) -> Tuple[Dataset, Dataset]:
        """
        Split processed dataset into train and evaluation sets
        
        Args:
            train_ratio: Ratio of data for training
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        if self.processed_dataset is None:
            raise ValueError("No processed dataset available. Run preprocess_combined_dataset() first.")
        
        # Shuffle dataset
        shuffled_dataset = self.processed_dataset.shuffle(seed=42)
        
        # Calculate split
        total_size = len(shuffled_dataset)
        train_size = int(total_size * train_ratio)
        
        # Split
        self.train_dataset = shuffled_dataset.select(range(train_size))
        self.eval_dataset = shuffled_dataset.select(range(train_size, total_size))
        
        logger.info(f"📊 Dataset split: {len(self.train_dataset)} train, {len(self.eval_dataset)} eval")
        
        return self.train_dataset, self.eval_dataset
    
    def get_evaluation_subset(self, size: int = 100) -> Dataset:
        """
        Get subset of evaluation data for testing
        
        Args:
            size: Number of samples for evaluation
            
        Returns:
            Evaluation subset
        """
        if self.eval_dataset is None:
            # Create split if not done yet
            self.create_train_eval_split()
        
        eval_size = min(size, len(self.eval_dataset))
        eval_subset = self.eval_dataset.select(range(eval_size))
        
        logger.info(f"Created evaluation subset with {len(eval_subset)} samples")
        return eval_subset
    
    def get_dataset_statistics(self) -> Dict:
        """Get comprehensive dataset statistics"""
        stats = {}
        
        if self.combined_dataset:
            # Dataset source distribution
            sources = [sample['dataset_source'] for sample in self.combined_dataset]
            source_counts = {source: sources.count(source) for source in set(sources)}
            
            # Question type distribution
            q_types = [sample['question_type'] for sample in self.combined_dataset]
            type_counts = {qtype: q_types.count(qtype) for qtype in set(q_types)}
            
            # Text length statistics
            text_lengths = [len(sample['output'].split()) for sample in self.combined_dataset]
            
            stats = {
                'total_samples': len(self.combined_dataset),
                'source_distribution': source_counts,
                'question_type_distribution': type_counts,
                'avg_output_length': np.mean(text_lengths),
                'min_output_length': np.min(text_lengths),
                'max_output_length': np.max(text_lengths),
                'median_output_length': np.median(text_lengths)
            }
        
        return stats
    
    def save_datasets(self, save_dir: str = "data/processed"):
        """Save processed datasets"""
        os.makedirs(save_dir, exist_ok=True)
        
        if self.combined_dataset:
            self.combined_dataset.save_to_disk(os.path.join(save_dir, "combined_dataset"))
            
        if self.processed_dataset:
            self.processed_dataset.save_to_disk(os.path.join(save_dir, "processed_dataset"))
            
        if self.train_dataset and self.eval_dataset:
            self.train_dataset.save_to_disk(os.path.join(save_dir, "train_dataset"))
            self.eval_dataset.save_to_disk(os.path.join(save_dir, "eval_dataset"))
            
        logger.info(f"💾 Datasets saved to {save_dir}")

# Main execution function
def load_and_prepare_datasets():
    """Load and prepare all datasets"""
    loader = EnhancedMedicalDataLoader()
    
    # Load datasets
    datasets = loader.load_multiple_medical_datasets()
    
    # Combine datasets
    combined = loader.combine_datasets()
    
    # Preprocess
    processed = loader.preprocess_combined_dataset()
    
    # Create splits
    train, eval_data = loader.create_train_eval_split()
    
    # Get statistics
    stats = loader.get_dataset_statistics()
    
    logger.info("Dataset preparation completed!")
    logger.info(f"Statistics: {stats}")
    
    return loader

if __name__ == "__main__":
    load_and_prepare_datasets() 