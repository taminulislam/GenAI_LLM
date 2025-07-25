"""
Comprehensive Medical Data Loader
Handles multiple medical datasets for comprehensive training and evaluation
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datasets import Dataset, load_dataset, concatenate_datasets
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveDataLoader:
    """Comprehensive data loader for multiple medical datasets"""
    
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.datasets = {}
        self.combined_datasets = {}
        
    def load_all_datasets(self) -> Dict[str, Dict[str, Dataset]]:
        """Load all configured medical datasets"""
        logger.info("Loading comprehensive medical datasets...")
        
        # Load MedMCQA dataset
        try:
            logger.info("Loading MedMCQA dataset...")
            medmcqa = load_dataset("medmcqa", split={"train": "train[:5000]", "validation": "validation[:500]", "test": "validation[500:1000]"})
            self.datasets['medmcqa'] = {
                'train': medmcqa['train'],
                'validation': medmcqa['validation'], 
                'test': medmcqa['test']
            }
            logger.info(f"MedMCQA loaded: {len(medmcqa['train'])} train, {len(medmcqa['validation'])} val, {len(medmcqa['test'])} test")
        except Exception as e:
            logger.warning(f"Could not load MedMCQA: {e}")
            self.datasets['medmcqa'] = {'train': None, 'validation': None, 'test': None}
        
        # Load PubMedQA dataset
        try:
            logger.info("Loading PubMedQA dataset...")
            pubmedqa = load_dataset("pubmed_qa", "pqa_labeled", split={"train": "train[:3000]", "test": "train[3000:3500]"})
            # Split train into train/validation
            train_split = pubmedqa['train'].train_test_split(test_size=0.2, seed=42)
            self.datasets['pubmedqa'] = {
                'train': train_split['train'],
                'validation': train_split['test'],
                'test': pubmedqa['test']
            }
            logger.info(f"PubMedQA loaded: {len(train_split['train'])} train, {len(train_split['test'])} val, {len(pubmedqa['test'])} test")
        except Exception as e:
            logger.warning(f"Could not load PubMedQA: {e}")
            self.datasets['pubmedqa'] = {'train': None, 'validation': None, 'test': None}
        
        # Load Medical Meadow Flashcards
        try:
            logger.info("Loading Medical Meadow Flashcards...")
            meadow = load_dataset("medalpaca/medical_meadow_medical_flashcards", split={"train": "train[:2000]", "test": "train[2000:2300]"})
            train_split = meadow['train'].train_test_split(test_size=0.2, seed=42)
            self.datasets['medical_flashcards'] = {
                'train': train_split['train'],
                'validation': train_split['test'],
                'test': meadow['test']
            }
            logger.info(f"Medical Flashcards loaded: {len(train_split['train'])} train, {len(train_split['test'])} val, {len(meadow['test'])} test")
        except Exception as e:
            logger.warning(f"Could not load Medical Flashcards: {e}")
            self.datasets['medical_flashcards'] = {'train': None, 'validation': None, 'test': None}
        
        return self.datasets
    
    def create_combined_datasets(self, datasets: Dict[str, Dict[str, Dataset]]) -> Dict[str, Dataset]:
        """Combine multiple datasets into unified train/val/test splits"""
        logger.info("Combining datasets...")
        
        combined = {'train': [], 'validation': [], 'test': []}
        
        for dataset_name, splits in datasets.items():
            for split_name, dataset in splits.items():
                if dataset is not None:
                    # Convert to standard format
                    formatted = self._format_dataset(dataset, dataset_name)
                    if formatted:
                        combined[split_name].append(formatted)
        
        # Concatenate datasets
        final_combined = {}
        for split_name, dataset_list in combined.items():
            if dataset_list:
                final_combined[split_name] = concatenate_datasets(dataset_list)
                logger.info(f"Combined {split_name}: {len(final_combined[split_name])} samples")
            else:
                final_combined[split_name] = None
        
        self.combined_datasets = final_combined
        return final_combined
    
    def _format_dataset(self, dataset: Dataset, dataset_name: str) -> Optional[Dataset]:
        """Format dataset to standard input/output format"""
        try:
            if dataset_name == 'medmcqa':
                def format_medmcqa(example):
                    question = example['question']
                    choices = [example['opa'], example['opb'], example['opc'], example['opd']]
                    choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
                    
                    input_text = f"Medical Question: {question}\n\nChoices:\n{choices_text}\n\nAnswer:"
                    
                    # Convert answer index to letter
                    answer_idx = example['cop']
                    target_text = chr(65 + answer_idx) if 0 <= answer_idx <= 3 else "A"
                    
                    return {
                        'input_text': input_text,
                        'target_text': target_text,
                        'dataset_source': 'medmcqa'
                    }
                
                return dataset.map(format_medmcqa)
            
            elif dataset_name == 'pubmedqa':
                def format_pubmedqa(example):
                    context = example.get('context', {})
                    contexts = context.get('contexts', []) if isinstance(context, dict) else []
                    context_text = " ".join(contexts[:2]) if contexts else ""  # Limit context length
                    
                    question = example['question']
                    input_text = f"Medical Context: {context_text}\n\nQuestion: {question}\n\nAnswer:"
                    target_text = example['final_decision']
                    
                    return {
                        'input_text': input_text,
                        'target_text': target_text,
                        'dataset_source': 'pubmedqa'
                    }
                
                return dataset.map(format_pubmedqa)
            
            elif dataset_name == 'medical_flashcards':
                def format_flashcards(example):
                    input_text = f"Medical Question: {example['input']}\n\nAnswer:"
                    target_text = example['output']
                    
                    return {
                        'input_text': input_text,
                        'target_text': target_text,
                        'dataset_source': 'medical_flashcards'
                    }
                
                return dataset.map(format_flashcards)
            
        except Exception as e:
            logger.error(f"Error formatting {dataset_name}: {e}")
            return None
        
        return None 